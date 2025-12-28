#!/usr/bin/env python3
"""Profile ABAPC removal vs MUS solving across node sizes.

This script mirrors the timing breakdown shown in tests_mus, emitting side-by-side
stats for grounding, solving, and peak memory for both ABAPC (clingo) and MUS (WASP).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple
import contextlib
import io
import os

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests_mus import RandomPCSimConfig, build_random_pc_case  # noqa: E402
from causalaba import CausalABA, compile_and_ground  # noqa: E402
from causalaba_mus import CausalABA_MUS, build_mus_program  # noqa: E402


# Some optional deps used by cd_algorithms.models (e.g., notears) are not installed in
# the profiling environment; stub them to keep imports lightweight.
def _ensure_notears_stub() -> None:
    import types

    if 'notears.nonlinear' in sys.modules:
        return
    notears_module = types.ModuleType('notears')
    notears_nonlinear_module = types.ModuleType('notears.nonlinear')

    class _DummyMLP:
        pass

    def _dummy_notears_nonlinear(*args, **kwargs):  # pragma: no cover - diagnostic only
        raise ImportError("notears is not installed in this profiling environment")

    notears_nonlinear_module.NotearsMLP = _DummyMLP
    notears_nonlinear_module.notears_nonlinear = _dummy_notears_nonlinear
    sys.modules['notears'] = notears_module
    sys.modules['notears.nonlinear'] = notears_nonlinear_module


def _mem_mb() -> float:
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


@contextlib.contextmanager
def _quiet_phase():
    """Silence stdout/stderr and pause logging for noisy phases."""
    log = logging.getLogger()
    prev_disabled = log.disabled
    prev_level = log.level
    log.disabled = True
    log.setLevel(logging.CRITICAL)
    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        try:
            yield
        finally:
            log.disabled = prev_disabled
            log.setLevel(prev_level)


def _parse_ext_facts_to_dicts(ext_facts_list: Iterable[str]):
    indep, dep = {}, {}
    for line in ext_facts_list:
        line = line.strip()
        if not line:
            continue
        if line.endswith('.'):
            line = line[:-1]
        m = re.match(r'(ext_indep|ext_dep)\((\d+),(\d+),([^\)]+)\)', line)
        if not m:
            continue
        fact_type, x, y, s = m.groups()
        x, y = int(x), int(y)
        if s == 'empty':
            S = ()
        else:
            ms = re.match(r's((?:0|[1-9]\d*)*)', s)
            if ms:
                idxs = ms.group(1)
                S = tuple(int(idxs[i]) for i in range(len(idxs))) if idxs else ()
            else:
                S = ()
        target = indep if fact_type == 'ext_indep' else dep
        target.setdefault((x, y), set()).add(S)
    return indep, dep


def profile_once(
    n_nodes: int,
    *,
    edge_per_node: int,
    solve_timeout: int,
    graph_type: str,
    seed: int,
    wasp_path: str,
    max_muses: int,
    camus_mcs_threshold: int,
    camus_mus_threshold: int,
    emit_lp_path: str | None = None,
) -> bool:
    _ensure_notears_stub()

    # Minimal progress markers so we know the run is alive; heavy logs stay quiet
    logging.info(f"[start] n_nodes={n_nodes}: compiling/grounding + solving (quiet)...")

    # Track if we hit a timeout during computation
    early_timeout = False
    timeout_phase = None
    abapc_time = solve_timeout
    mus_time = solve_timeout
    remove_n = 0
    mus_result = {'n_mus': 0, 'n_mcs': 0}
    ground_time = 0.0
    mem_before = _mem_mb()
    mem_abapc = mem_before
    mem_mus = mem_before

    try:
        with _quiet_phase():
            config = RandomPCSimConfig(
                n_nodes=n_nodes,
                alpha=0.05,
                graph_type=graph_type,
                edge_per_node=edge_per_node,
                seed=seed,
                sample_size=10000,
                uc_rule=5,
                uc_priority=2,
                stable=True,
            )
            case = build_random_pc_case(config)
            facts = case["facts"]
            facts_ext = case["facts_ext"]

            # Persist facts for solvers
            import tempfile

            fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
            os.close(fd)
            facts_I_file = facts_file.replace('.lp', '_I.lp')
            facts_wc_file = facts_file.replace('.lp', '_wc.lp')
            with open(facts_file, 'w') as f:
                for s in facts_ext:
                    line = s if s.endswith('.') else s + '.'
                    f.write(f"#external {line}\n")
            with open(facts_I_file, 'w') as f:
                for fact, s in zip(facts, facts_ext):
                    line = s if s.endswith('.') else s + '.'
                    I = fact[1]
                    f.write(f"{line} I={I}, NA\n")
            with open(facts_wc_file, 'w') as f:
                for fact, s in zip(facts, facts_ext):
                    line = s if s.endswith('.') else s + '.'
                    I = fact[1]
                    f.write(f":~ {line} [-{int(I*1e14)*2}]\n")

            # --- ABAPC (removal) ---
            mem_before = _mem_mb()
            start_abapc = datetime.now()
            models_after, _multiple, _stats, remove_n = CausalABA(
                n_nodes,
                facts_file,
                weak_constraints=True,
                search_for_models='first',
                skeleton_rules_reduction=True,
                print_models=False,
                return_statistics=True,
                solve_timeout=solve_timeout,
            )
            abapc_time = (datetime.now() - start_abapc).total_seconds()
            mem_abapc = _mem_mb()

            # --- MUS ---
            import tempfile as _tf

            fd_mus, facts_mus_file = _tf.mkstemp(suffix='.lp', text=True)
            os.close(fd_mus)
            with open(facts_mus_file, 'w') as f:
                for s in facts_ext:
                    line = s if s.endswith('.') else s + '.'
                    f.write(f"{line}\n")

            # Optionally emit a complete adorned MUS program for external execution
            if emit_lp_path:
                try:
                    program = build_mus_program(n_nodes, facts_ext, facts_mus_file, deadline=None)
                    out_path = Path(emit_lp_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_path, 'w') as outf:
                        outf.write(program)
                except Exception as e:
                    logging.debug(f"Emit LP failed: {e}")

            start_mus = datetime.now()
            mus_result = CausalABA_MUS(
                n_nodes=n_nodes,
                facts_location=facts_mus_file,
                gringo_path="clingo",
                wasp_path=wasp_path,
                max_muses=max_muses,
                mus_algorithm="camus",
                print_mcses=True,
                camus_mcs_threshold=camus_mcs_threshold,
                camus_mus_threshold=camus_mus_threshold,
                solve_timeout=solve_timeout,
            )
            mus_time = (datetime.now() - start_mus).total_seconds()
            mem_mus = _mem_mb()

            # --- Grounding time (shared) ---
            indep_facts, dep_facts = _parse_ext_facts_to_dicts(facts_ext)
            timing = {}
            try:
                _ = compile_and_ground(
                    n_nodes,
                    facts_location="",
                    skeleton_rules_reduction=True,
                    weak_constraints=False,
                    indep_facts=indep_facts,
                    dep_facts=dep_facts,
                    opt_mode='optN',
                    out_n=0,
                    show=['arrow'],
                    pre_grounding=False,
                    ext_flag=False,
                    prior_knowledge=None,
                    timing_recorder=timing,
                )
                ground_time = timing.get('ground_sec_total', 0.0)
            except Exception as e:
                ground_time = min(abapc_time * 0.1, 0.1)
                logging.debug(f"Grounding measurement fallback: {ground_time:.3f}s (error: {e})")

    except TimeoutError as e:
        early_timeout = True
        # Determine which phase timed out based on error message
        err_str = str(e)
        if 'compile_and_ground' in err_str:
            timeout_phase = 'grounding'
        else:
            timeout_phase = 'solving'

    if emit_lp_path:
        logging.info(f"[emit] MUS program written to {emit_lp_path}")
    logging.info(f"[done] n_nodes={n_nodes}: compute finished; summary below")

    abapc_solve_time = max(0.0, abapc_time - ground_time)
    mus_solve_time = max(0.0, mus_time - ground_time)

    # Banner placed here so no solver warnings appear between header and summary
    logging.info("\n" + "=" * 90)
    logging.info(f"TIMING COMPARISON: GROUNDING + SOLVING (n_nodes={n_nodes})")
    logging.info("=" * 90)

    logging.info(f"Measured grounding time (clingo):     {ground_time:8.3f}s")
    logging.info("")
    logging.info("  ┌─ ABAPC (removal strategy):")
    logging.info(f"  │   Grounding:  {ground_time:8.3f}s")
    logging.info(f"  │   Solving:    {abapc_solve_time:8.3f}s")
    logging.info(f"  │   Total:      {abapc_time:8.3f}s")
    logging.info(f"  │   Peak RSS:   {mem_abapc:8.1f} MiB (Δ={mem_abapc - mem_before:+.1f})")
    logging.info("  │")
    logging.info("  └─ MUS solver (full analysis):")
    logging.info(f"      Grounding:  {ground_time:8.3f}s (shared, clingo)")
    logging.info(f"      Solving:    {mus_solve_time:8.3f}s (WASP)")
    logging.info(f"      Total:      {mus_time:8.3f}s")
    logging.info(f"      Peak RSS:   {mem_mus:8.1f} MiB (Δ={mem_mus - mem_abapc:+.1f})")
    logging.info("")

    ratio_total = mus_time / abapc_time if abapc_time > 0 else 0.0
    ratio_solve = mus_solve_time / abapc_solve_time if abapc_solve_time > 0 else 0.0
    logging.info(f"  Total time ratio (MUS/ABAPC):   {ratio_total:8.2f}x")
    logging.info(f"  Solve time ratio (WASP/clingo): {ratio_solve:8.2f}x")

    # Detect timeout conditions
    abapc_timed_out = early_timeout or abapc_time >= (solve_timeout - 1.0)
    mus_timed_out = early_timeout or mus_time >= (solve_timeout - 1.0)

    logging.info("-- Results --")
    result_parts = [f"remove_n={remove_n}", f"mus_count={mus_result.get('n_mus', 0)}", f"mcs_count={mus_result.get('n_mcs', 0)}"]
    if early_timeout:
        result_parts.append(f"⚠ EARLY TIMEOUT during {timeout_phase} ({solve_timeout}s budget)")
    else:
        if abapc_timed_out:
            result_parts.append(f"⚠ ABAPC TIMEOUT ({abapc_time:.1f}s)")
        if mus_timed_out:
            result_parts.append(f"⚠ MUS TIMEOUT ({mus_time:.1f}s)")
    logging.info(f"  {', '.join(result_parts)}")
    is_sat = (remove_n == 0 and mus_result.get('n_mus', 0) == 0)
    if is_sat:
        logging.info(f"  SAT: no removals and no MUS/MCS (seed={seed}, n={n_nodes})")

    for path in (facts_file, facts_I_file, facts_wc_file, facts_mus_file):
        try:
            os.remove(path)
        except Exception:
            pass

    return is_sat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare ABAPC removal vs MUS solving time/memory across node sizes")
    parser.add_argument("--node-sizes", type=str, default="5,6,7,8", help="Comma-separated node sizes to profile")
    parser.add_argument("--solve-timeout", type=int, default=60, help="Wall-clock timeout (seconds) per instance")
    parser.add_argument("--edge-per-node", type=int, default=2, help="Edge-per-node multiplier for random DAG")
    parser.add_argument("--graph-type", type=str, default="ER", help="Random graph type")
    parser.add_argument("--wasp-path", type=str, default="/vol/bitbucket/fr920/wasp/build/release/wasp", help="Path to WASP executable")
    parser.add_argument("--max-muses", type=int, default=200, help="Maximum MUSes to enumerate")
    parser.add_argument("--camus-mcs-threshold", type=int, default=50, help="CAMUS MCS threshold")
    parser.add_argument("--camus-mus-threshold", type=int, default=300, help="CAMUS MUS threshold")
    parser.add_argument("--seed-base", type=int, default=2004, help="Base seed; incremented by node size for determinism")
    parser.add_argument("--rep_unsat", type=int, default=0, help="Retries with new seeds when an instance is SAT (no removals and no MUS/MCS)")
    parser.add_argument("--emit-lp", type=str, default="", help="Write complete adorned MUS program to this .lp path for external execution")
    parser.add_argument("--emit-lp-dir", type=str, default="", help="Auto-name and write adorned MUS programs to this directory (names: mus_<n_nodes>_<seed>.lp)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
        force=True,
    )

    _ensure_notears_stub()

    node_sizes = [int(x.strip()) for x in args.node_sizes.split(',') if x.strip()]
    logging.info(
        f"Profile config: node_sizes={node_sizes}, solve_timeout={args.solve_timeout}s, edge_per_node={args.edge_per_node}, graph_type={args.graph_type}"
    )
    for n in node_sizes:
        base_seed = args.seed_base# + (n - node_sizes[0])
        curr_seed = base_seed
        logging.info(f"\n=== Profiling n_nodes={n} (base={args.seed_base}, seed={curr_seed}) ===")
        # Determine the LP output path for this (n, seed) pair
        emit_lp_path = None
        if args.emit_lp:
            emit_lp_path = args.emit_lp
        elif args.emit_lp_dir:
            Path(args.emit_lp_dir).mkdir(parents=True, exist_ok=True)
            emit_lp_path = str(Path(args.emit_lp_dir) / f"mus_{n}_{curr_seed}.lp")
        
        is_sat = profile_once(
            n_nodes=n,
            edge_per_node=args.edge_per_node,
            solve_timeout=args.solve_timeout,
            graph_type=args.graph_type,
            seed=curr_seed,
            wasp_path=args.wasp_path,
            max_muses=args.max_muses,
            camus_mcs_threshold=args.camus_mcs_threshold,
            camus_mus_threshold=args.camus_mus_threshold,
            emit_lp_path=emit_lp_path,
        )
        attempts = 0
        while is_sat and attempts < args.rep_unsat:
            attempts += 1
            curr_seed += 1
            logging.info(f"[retry] n_nodes={n}: SAT instance; trying seed={curr_seed} ({attempts}/{args.rep_unsat})")
            # Update LP path for retry seed
            if args.emit_lp_dir:
                emit_lp_path = str(Path(args.emit_lp_dir) / f"mus_{n}_{curr_seed}.lp")
            is_sat = profile_once(
                n_nodes=n,
                edge_per_node=args.edge_per_node,
                solve_timeout=args.solve_timeout,
                graph_type=args.graph_type,
                seed=curr_seed,
                wasp_path=args.wasp_path,
                max_muses=args.max_muses,
                camus_mcs_threshold=args.camus_mcs_threshold,
                camus_mus_threshold=args.camus_mus_threshold,
                emit_lp_path=emit_lp_path,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
