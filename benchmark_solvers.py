#!/usr/bin/env python
"""
Benchmark baseline vs. incremental CausalABA on synthetic BNLearn-like tasks.

For each n in --n-nodes-list and each run, we:
  - Sample a random DAG
  - Simulate discrete data
  - Run PC to obtain sepsets and build facts (like ABAPC)
  - Run both solvers with the same facts, collecting time, memory, SID, and model count.

Example:
  python benchmark_solvers.py --n-nodes-list 5 7 10 --runs 3 --fact-pct 0.3
"""

from __future__ import annotations

import argparse
import tempfile
import time
import tracemalloc
import resource
import os
import csv
from pathlib import Path
from itertools import combinations
import logging

import numpy as np
import pandas as pd

import json

# Support running as a package module (preferred) as well as a standalone script.
try:
    from .cd_algorithms.PC import pc
    from .causalaba import CausalABA as CausalABA_Base
    from .causalaba_increm import CausalABA as CausalABA_Incr
    from .causalaba_binsearch import CausalABA as CausalABA_Binsearch
    from .causalaba_weakc import CausalABA as CausalABA_WeakC
    from .utils.data_utils import simulate_dag, simulate_discrete_data
    from .utils.graph_utils import initial_strength, set_of_models_to_set_of_graphs, DAGMetrics
    from .utils.helpers import random_stability
except ImportError:  # pragma: no cover
    from cd_algorithms.PC import pc
    from causalaba import CausalABA as CausalABA_Base
    from causalaba_increm import CausalABA as CausalABA_Incr
    from causalaba_binsearch import CausalABA as CausalABA_Binsearch
    from causalaba_weakc import CausalABA as CausalABA_WeakC
    from utils.data_utils import simulate_dag, simulate_discrete_data
    from utils.graph_utils import initial_strength, set_of_models_to_set_of_graphs, DAGMetrics
    from utils.helpers import random_stability
import clingo


_ROW_COLUMNS = [
    "n_nodes",
    "run",
    "solver",
    "elapsed_sec",
    "solver_sec",
    "postprocess_sec",
    "total_sec",
    "peak_mem_bytes",
    "peak_mem_mb",
    "peak_after_solver_bytes",
    "peak_after_postprocess_bytes",
    "peak_after_solver_mb",
    "peak_after_postprocess_mb",
    "peak_after_compile_bytes",
    "peak_after_ground_bytes",
    "peak_after_solve_bytes_last",
    "peak_after_solve_bytes_max",
    "peak_after_compile_mb",
    "peak_after_ground_mb",
    "peak_after_solve_mb",
    "rss_delta",
    "rss_delta_kib",
    "rss_delta_mb",
    "remove_n",
    "compile_sec_total",
    "compile_sec_last",
    "compile_calls",
    "ground_sec_total",
    "ground_sec_last",
    "ground_calls_internal",
    "solve_sec_total",
    "solve_sec_last",
    "solve_calls_internal",
    "sat_check_sec_total",
    "sat_check_sec_last",
    "final_opt_sec",
    "paths_added_total",
    "paths_added_last",
    "n_models",
    "sid",
    "solve_calls",
    "ground_calls",
    "timed_out",
    "error",
]


def _append_csv_row(out_path: Path, row: dict, *, write_header_if_needed: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ROW_COLUMNS, extrasaction="ignore")
        if write_header_if_needed and f.tell() == 0:
            writer.writeheader()
        writer.writerow({k: row.get(k, float("nan")) for k in _ROW_COLUMNS})
        f.flush()
        os.fsync(f.fileno())


def _bytes_to_mib(x: float | int) -> float:
    try:
        return float(x) / (1024.0 * 1024.0)
    except Exception:
        return float('nan')


def _kib_to_mib(x: float | int) -> float:
    try:
        return float(x) / 1024.0
    except Exception:
        return float('nan')


def build_facts(n_nodes: int, sepsets, alpha: float, facts_path: Path) -> None:
    """Write facts.lp, facts_wc.lp, and facts_I.lp given PC sepsets."""
    facts = set()
    for X, Y in combinations(range(n_nodes), 2):
        for S, p in sepsets[X, Y]:
            dep_type = "indep" if p > alpha else "dep"
            s_str = "empty" if len(S) == 0 else "s" + "y".join([str(i) for i in S])
            I = initial_strength(p, len(S), alpha, 0.5, n_nodes)
            facts.add((X, tuple(S), Y, dep_type, f"{dep_type}({X},{Y},{s_str}).", I))

    facts_wc_path = facts_path.parent / (facts_path.stem + "_wc.lp")
    facts_I_path = facts_path.parent / (facts_path.stem + "_I.lp")
    with open(facts_path, "w") as f_lp, open(facts_wc_path, "w") as f_wc, open(facts_I_path, "w") as f_I:
        for s in facts:
            f_lp.write(f"#external ext_{s[4]}\n")
            f_wc.write(f":~ ext_{s[4]} [-{int(s[5]*1000)}]\n")
            f_I.write(f"ext_{s[4]} I={s[5]}, True\n")


def arrows_to_adj(arrows: set[tuple[int, int]], n: int) -> np.ndarray:
    B = np.zeros((n, n))
    for (u, v) in arrows:
        B[u, v] = 1
    return B


def run_solver(
    label: str,
    solver,
    n_nodes: int,
    facts_path: Path,
    fact_pct: float,
    weak_constraints: bool,
    skeleton_rules_reduction: bool,
    disable_reground: bool,
    timeout: int,
    opt_mode: str = "optN",
    out_n: int = 0,
    max_path_length: int | None = None,
    max_conditioning_size: int | None = None,
    threads: int | None = None,
    tracemalloc_top: int = 0,
    tracemalloc_group: str = "lineno",
) -> dict:
    logger = logging.getLogger(__name__)

    # Monkeypatch clingo Control.solve and Control.ground (both the top-level
    # alias and clingo.control.Control) to count invocations. The solver code
    # sometimes imports Control via `from clingo.control import Control`, so
    # we patch both references to be safe.
    solve_calls = {"count": 0}
    ground_calls = {"count": 0}
    orig_solve_top = clingo.Control.solve
    orig_ground_top = clingo.Control.ground
    try:
        from clingo import control as clingo_control
        ControlClass = clingo_control.Control
    except Exception:
        ControlClass = None
    orig_solve_ctrl = ControlClass.solve if ControlClass else None
    orig_ground_ctrl = ControlClass.ground if ControlClass else None

    def _solve(self, *args, **kwargs):
        solve_calls["count"] += 1
        if self is not None and hasattr(self, "__class__") and self.__class__ is ControlClass and orig_solve_ctrl:
            return orig_solve_ctrl(self, *args, **kwargs)
        return orig_solve_top(self, *args, **kwargs)

    def _ground(self, *args, **kwargs):
        ground_calls["count"] += 1
        if self is not None and hasattr(self, "__class__") and self.__class__ is ControlClass and orig_ground_ctrl:
            return orig_ground_ctrl(self, *args, **kwargs)
        return orig_ground_top(self, *args, **kwargs)

    clingo.Control.solve = _solve
    clingo.Control.ground = _ground
    if ControlClass:
        ControlClass.solve = _solve
        ControlClass.ground = _ground

    timed_out = False
    error: str | None = None

    tracemalloc.start()
    # On Linux, ru_maxrss is reported in KiB.
    rss_before_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    try:
        logger.info(
            "[%s] Starting solve (n_nodes=%s facts=%s disable_reground=%s timeout=%ss)",
            label,
            n_nodes,
            facts_path,
            disable_reground,
            timeout,
        )
        res = solver(
            n_nodes,
            str(facts_path),
            weak_constraints=weak_constraints,
            fact_pct=fact_pct,
            opt_mode=opt_mode,
            out_n=out_n,
            search_for_models="first",
            print_models=False,
            skeleton_rules_reduction=skeleton_rules_reduction,
            disable_reground=disable_reground,
            return_statistics=True,
            solve_timeout=float(timeout) if timeout else None,
            max_path_length=max_path_length,
            max_conditioning_size=max_conditioning_size,
            threads=threads,
        )
        # Normalize solver return shapes.
        # Expected (bench-enhanced): [models, multiple, clingo_stats, remove_n, profile]
        # Legacy: [models, multiple]
        models = res[0] if isinstance(res, (list, tuple)) and len(res) > 0 else []
        _multiple = bool(res[1]) if isinstance(res, (list, tuple)) and len(res) > 1 else False
        clingo_stats = res[2] if isinstance(res, (list, tuple)) and len(res) > 2 else None
        remove_n = int(res[3]) if isinstance(res, (list, tuple)) and len(res) > 3 else 0
        solver_profile = res[4] if isinstance(res, (list, tuple)) and len(res) > 4 else {}
        if not isinstance(solver_profile, dict):
            solver_profile = {}
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        models = []
        _multiple = False
        clingo_stats = None
        remove_n = 0
        solver_profile = {}
    solver_sec = time.perf_counter() - t0

    # Track tracemalloc peak right after solver returned.
    peak_after_solver_bytes = None
    try:
        _cur, _peak = tracemalloc.get_traced_memory()
        peak_after_solver_bytes = int(_peak)
    except Exception:
        peak_after_solver_bytes = None

    # With clingo async cancellation, elapsed should generally be <= timeout.
    timed_out = bool(timeout) and solver_sec >= float(timeout)
    if error is not None and error.startswith("TimeoutError"):
        timed_out = True

    snapshot_stats = None
    if tracemalloc_top and tracemalloc_top > 0:
        try:
            snapshot = tracemalloc.take_snapshot()
            snapshot_stats = snapshot.statistics(tracemalloc_group)
        except Exception:
            snapshot_stats = None

    # Post-process: convert models -> graph arrows (can be non-trivial when many models).
    t_post0 = time.perf_counter()
    model_sets, _ = set_of_models_to_set_of_graphs(models, n_nodes, False)
    # Deterministic representative when multiple models exist.
    # (Set iteration order is arbitrary and can cause SID noise.)
    if model_sets:
        arrows = min(model_sets, key=lambda s: tuple(sorted(s)))
    else:
        arrows = set()
    postprocess_sec = time.perf_counter() - t_post0

    # Track tracemalloc peak after postprocess.
    peak_after_postprocess_bytes = None
    try:
        _cur2, _peak2 = tracemalloc.get_traced_memory()
        peak_after_postprocess_bytes = int(_peak2)
    except Exception:
        peak_after_postprocess_bytes = None

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Restore patched methods
    clingo.Control.solve = orig_solve_top
    clingo.Control.ground = orig_ground_top
    if ControlClass:
        ControlClass.solve = orig_solve_ctrl
        ControlClass.ground = orig_ground_ctrl

    logger.info(
        "[%s] Done (timed_out=%s solver=%.3fs post=%.3fs solve_calls=%s ground_calls=%s models=%s)",
        label,
        timed_out,
        solver_sec,
        postprocess_sec,
        solve_calls["count"],
        ground_calls["count"],
        len(model_sets),
    )

    # Stage profile log (if solver provided one)
    if solver_profile:
        try:
            logger.info(
                "[%s] profile compile=%.3fs (last=%.3fs, calls=%s) ground=%.3fs (last=%.3fs, calls=%s) solve=%.3fs (last=%.3fs, calls=%s)",
                label,
                float(solver_profile.get("compile_sec_total", 0.0)),
                float(solver_profile.get("compile_sec_last", 0.0)),
                int(solver_profile.get("compile_calls", 0)),
                float(solver_profile.get("ground_sec_total", 0.0)),
                float(solver_profile.get("ground_sec_last", 0.0)),
                int(solver_profile.get("ground_calls_internal", 0)),
                float(solver_profile.get("solve_sec_total", 0.0)),
                float(solver_profile.get("solve_sec_last", 0.0)),
                int(solver_profile.get("solve_calls_internal", 0)),
            )
        except Exception:
            pass

    if snapshot_stats is not None:
        logger.info(
            "[%s] tracemalloc top %s by %s (current allocations)",
            label,
            tracemalloc_top,
            tracemalloc_group,
        )
        for stat in snapshot_stats[:tracemalloc_top]:
            try:
                logger.info("[%s] %s", label, stat)
            except Exception:
                pass

    return {
        "solver": label,
        # Solver wall time only (historical metric)
        "elapsed_sec": solver_sec,
        "solver_sec": solver_sec,
        "postprocess_sec": postprocess_sec,
        "total_sec": solver_sec + postprocess_sec,
        "peak_mem_bytes": peak,
        "peak_after_solver_bytes": peak_after_solver_bytes,
        "peak_after_postprocess_bytes": peak_after_postprocess_bytes,
        "peak_after_solver_mb": _bytes_to_mib(peak_after_solver_bytes) if peak_after_solver_bytes is not None else float('nan'),
        "peak_after_postprocess_mb": _bytes_to_mib(peak_after_postprocess_bytes) if peak_after_postprocess_bytes is not None else float('nan'),
        # Backwards-compatible: keep rss_delta as KiB (matches ru_maxrss on Linux)
        "rss_delta": rss_after_kib - rss_before_kib,
        "rss_delta_kib": rss_after_kib - rss_before_kib,
        "rss_delta_mb": _kib_to_mib(rss_after_kib - rss_before_kib),
        "peak_mem_mb": _bytes_to_mib(peak),
        "n_models": len(model_sets),
        "arrows": arrows,
        "solve_calls": solve_calls["count"],
        "ground_calls": ground_calls["count"],
        "timed_out": timed_out,
        "error": error,
        "remove_n": int(remove_n or 0),
        # Solver-provided stage timings (cumulative + last)
        "compile_sec_total": float(solver_profile.get("compile_sec_total", float('nan'))),
        "compile_sec_last": float(solver_profile.get("compile_sec_last", float('nan'))),
        "compile_calls": int(solver_profile.get("compile_calls", 0)),
        "ground_sec_total": float(solver_profile.get("ground_sec_total", float('nan'))),
        "ground_sec_last": float(solver_profile.get("ground_sec_last", float('nan'))),
        "ground_calls_internal": int(solver_profile.get("ground_calls_internal", 0)),
        "solve_sec_total": float(solver_profile.get("solve_sec_total", float('nan'))),
        "solve_sec_last": float(solver_profile.get("solve_sec_last", float('nan'))),
        "solve_calls_internal": int(solver_profile.get("solve_calls_internal", 0)),
        "sat_check_sec_total": float(solver_profile.get("sat_check_sec_total", float('nan'))),
        "sat_check_sec_last": float(solver_profile.get("sat_check_sec_last", float('nan'))),
        "final_opt_sec": float(solver_profile.get("final_opt_sec", float('nan'))),
        "paths_added_total": int(solver_profile.get("paths_added_total", 0)),
        "paths_added_last": int(solver_profile.get("paths_added_last", 0)),
        "peak_after_compile_bytes": solver_profile.get("peak_after_compile_bytes"),
        "peak_after_ground_bytes": solver_profile.get("peak_after_ground_bytes"),
        "peak_after_solve_bytes_last": solver_profile.get("peak_after_solve_bytes_last"),
        "peak_after_solve_bytes_max": solver_profile.get("peak_after_solve_bytes_max"),
        "peak_after_compile_mb": _bytes_to_mib(solver_profile.get("peak_after_compile_bytes"))
        if solver_profile.get("peak_after_compile_bytes") is not None
        else float('nan'),
        "peak_after_ground_mb": _bytes_to_mib(solver_profile.get("peak_after_ground_bytes"))
        if solver_profile.get("peak_after_ground_bytes") is not None
        else float('nan'),
        "peak_after_solve_mb": _bytes_to_mib(solver_profile.get("peak_after_solve_bytes_max"))
        if solver_profile.get("peak_after_solve_bytes_max") is not None
        else float('nan'),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark baseline vs incremental CausalABA on synthetic data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--n-nodes-list", nargs="+", type=int, required=True, help="List of node counts to test")
    ap.add_argument("--runs", type=int, default=3, help="Runs per node count")
    ap.add_argument("--fact-pct", type=float, default=1, help="Fraction of facts to activate")
    ap.add_argument("--alpha", type=float, default=0.05, help="Significance level for PC")
    ap.add_argument("--sample-size", type=int, default=5000, help="Sample size for simulated data")
    ap.add_argument("--out", type=str, default=None, help="Path to save per-run results CSV")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV instead of appending (only relevant when --out is set)",
    )
    ap.add_argument("--skeleton-rules-reduction", action="store_true", help="Enable skeleton_rules_reduction in solver")
    ap.add_argument("--disable-reground", action="store_true", help="Disable regrounding (applies to both solvers)")
    ap.add_argument("--timeout", type=int, default=900, help="Per-instance timeout in seconds (default 900s)")
    ap.add_argument("--max-path-length", type=int, default=None, help="Bound max path length for baseline path rules")
    ap.add_argument("--max-conditioning-size", type=int, default=None, help="Bound |S| for conditioning sets")
    ap.add_argument("--threads", type=int, default=None, help="Threads for clingo (-t)")
    ap.add_argument(
        "--opt-mode",
        type=str,
        default="optN",
        choices=["opt", "optN", "ignore"],
        help="clingo optimization mode used by solvers",
    )
    ap.add_argument(
        "--out-n",
        type=int,
        default=1,
        help="Max models to enumerate (0 means all; 1 recommended for benchmarking)",
    )
    ap.add_argument(
        "--solvers",
        type=str,
        default="both",
        choices=["both", "baseline", "incremental", "binsearch", "weakc", "all"],
        help="Which solver(s) to run",
    )
    ap.add_argument(
        "--tracemalloc-top",
        type=int,
        default=0,
        help="Log top-N tracemalloc allocation sites (0 disables)",
    )
    ap.add_argument(
        "--tracemalloc-group",
        type=str,
        default="lineno",
        choices=["lineno", "filename"],
        help="Group tracemalloc stats by line or file",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out_path: Path | None = None
    if args.out:
        out_path = Path(args.out)
        # Fail fast if we cannot create the output directory.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.overwrite and out_path.exists():
            out_path.unlink()

    rows = []
    for n in args.n_nodes_list:
        for run in range(args.runs):
            seed = 1000 + 10 * n + run
            random_stability(seed)
            B_true = simulate_dag(d=n, s0=max(n, n - 1), graph_type="ER")
            truth_edges = set(zip(*np.where(B_true == 1)))
            data = simulate_discrete_data(num_of_nodes=n, sample_size=args.sample_size, truth_DAG_directed_edges=truth_edges, random_seed=seed)
            cg = pc(data=data, alpha=args.alpha, indep_test="gsq", uc_rule=3, uc_priority=2, stable=True, show_progress=False, verbose=False)

            with tempfile.TemporaryDirectory() as tmpdir:
                facts_path = Path(tmpdir) / "facts.lp"
                build_facts(n, cg.sepset, args.alpha, facts_path)

                results = []
                if args.solvers in ("both", "all", "baseline"):
                    results.append(
                        run_solver(
                            "baseline",
                            CausalABA_Base,
                            n,
                            facts_path,
                            args.fact_pct,
                            True,
                            args.skeleton_rules_reduction,
                            args.disable_reground,
                            args.timeout,
                            opt_mode=args.opt_mode,
                            out_n=args.out_n,
                            max_path_length=args.max_path_length,
                            max_conditioning_size=args.max_conditioning_size,
                            threads=args.threads,
                            tracemalloc_top=args.tracemalloc_top,
                            tracemalloc_group=args.tracemalloc_group,
                        )
                    )
                if args.solvers in ("both", "all", "incremental"):
                    # Run incremental with reground disabled to showcase incremental grounding
                    results.append(
                        run_solver(
                            "incremental",
                            CausalABA_Incr,
                            n,
                            facts_path,
                            args.fact_pct,
                            True,
                            args.skeleton_rules_reduction,
                            args.disable_reground,
                            args.timeout,
                            opt_mode=args.opt_mode,
                            out_n=args.out_n,
                            max_path_length=args.max_path_length,
                            max_conditioning_size=args.max_conditioning_size,
                            threads=args.threads,
                            tracemalloc_top=args.tracemalloc_top,
                            tracemalloc_group=args.tracemalloc_group,
                        )
                    )

                if args.solvers in ("all", "binsearch"):
                    results.append(
                        run_solver(
                            "binsearch",
                            CausalABA_Binsearch,
                            n,
                            facts_path,
                            args.fact_pct,
                            True,
                            args.skeleton_rules_reduction,
                            args.disable_reground,
                            args.timeout,
                            opt_mode=args.opt_mode,
                            out_n=args.out_n,
                            max_path_length=args.max_path_length,
                            max_conditioning_size=args.max_conditioning_size,
                            threads=args.threads,
                            tracemalloc_top=args.tracemalloc_top,
                            tracemalloc_group=args.tracemalloc_group,
                        )
                    )

                if args.solvers in ("all", "weakc"):
                    results.append(
                        run_solver(
                            "weakc",
                            CausalABA_WeakC,
                            n,
                            facts_path,
                            args.fact_pct,
                            True,
                            args.skeleton_rules_reduction,
                            args.disable_reground,
                            args.timeout,
                            opt_mode=args.opt_mode,
                            out_n=args.out_n,
                            max_path_length=args.max_path_length,
                            max_conditioning_size=args.max_conditioning_size,
                            threads=args.threads,
                            tracemalloc_top=args.tracemalloc_top,
                            tracemalloc_group=args.tracemalloc_group,
                        )
                    )

            for res in results:
                sid = DAGMetrics(arrows_to_adj(res["arrows"], n), B_true).metrics.get("sid", np.nan)
                row = {
                    "n_nodes": n,
                    "run": run,
                    "solver": res["solver"],
                    "elapsed_sec": res["elapsed_sec"],
                    "solver_sec": res.get("solver_sec", res["elapsed_sec"]),
                    "postprocess_sec": res.get("postprocess_sec", float('nan')),
                    "total_sec": res.get("total_sec", float('nan')),
                    "peak_mem_bytes": res["peak_mem_bytes"],
                    "peak_mem_mb": res.get("peak_mem_mb", _bytes_to_mib(res["peak_mem_bytes"])),
                    "peak_after_solver_bytes": res.get("peak_after_solver_bytes", float('nan')),
                    "peak_after_postprocess_bytes": res.get("peak_after_postprocess_bytes", float('nan')),
                    "peak_after_solver_mb": res.get("peak_after_solver_mb", float('nan')),
                    "peak_after_postprocess_mb": res.get("peak_after_postprocess_mb", float('nan')),
                    "peak_after_compile_bytes": res.get("peak_after_compile_bytes", float('nan')),
                    "peak_after_ground_bytes": res.get("peak_after_ground_bytes", float('nan')),
                    "peak_after_solve_bytes_last": res.get("peak_after_solve_bytes_last", float('nan')),
                    "peak_after_solve_bytes_max": res.get("peak_after_solve_bytes_max", float('nan')),
                    "peak_after_compile_mb": res.get("peak_after_compile_mb", float('nan')),
                    "peak_after_ground_mb": res.get("peak_after_ground_mb", float('nan')),
                    "peak_after_solve_mb": res.get("peak_after_solve_mb", float('nan')),
                    "rss_delta": res["rss_delta"],
                    "rss_delta_kib": res.get("rss_delta_kib", res["rss_delta"]),
                    "rss_delta_mb": res.get("rss_delta_mb", _kib_to_mib(res["rss_delta"])),
                    "remove_n": res.get("remove_n", 0),
                    "compile_sec_total": res.get("compile_sec_total", float('nan')),
                    "compile_sec_last": res.get("compile_sec_last", float('nan')),
                    "compile_calls": res.get("compile_calls", 0),
                    "ground_sec_total": res.get("ground_sec_total", float('nan')),
                    "ground_sec_last": res.get("ground_sec_last", float('nan')),
                    "ground_calls_internal": res.get("ground_calls_internal", 0),
                    "solve_sec_total": res.get("solve_sec_total", float('nan')),
                    "solve_sec_last": res.get("solve_sec_last", float('nan')),
                    "solve_calls_internal": res.get("solve_calls_internal", 0),
                    "sat_check_sec_total": res.get("sat_check_sec_total", float('nan')),
                    "sat_check_sec_last": res.get("sat_check_sec_last", float('nan')),
                    "final_opt_sec": res.get("final_opt_sec", float('nan')),
                    "paths_added_total": res.get("paths_added_total", 0),
                    "paths_added_last": res.get("paths_added_last", 0),
                    "n_models": res["n_models"],
                    "sid": sid,
                    "solve_calls": res["solve_calls"],
                    "ground_calls": res["ground_calls"],
                    "timed_out": res["timed_out"],
                    "error": res["error"],
                }
                rows.append(row)
                if out_path is not None:
                    _append_csv_row(out_path, row)

    df = pd.DataFrame(rows)
    if out_path is not None:
        print(f"Saved {len(df)} rows to {out_path} (incremental write enabled)")
    # Print a readable view (keep raw bytes/KiB in CSV).
    printable = df.copy()
    if "peak_mem_mb" in printable.columns:
        printable["peak_mem_mb"] = printable["peak_mem_mb"].map(lambda v: round(float(v), 3) if pd.notna(v) else v)
    if "rss_delta_mb" in printable.columns:
        printable["rss_delta_mb"] = printable["rss_delta_mb"].map(lambda v: round(float(v), 3) if pd.notna(v) else v)
    print(printable.to_csv(index=False))
    summary = (
        df.groupby(["n_nodes", "solver"], as_index=False)
        .agg(
            {
                "elapsed_sec": ["mean", "std"],
                "postprocess_sec": ["mean", "std"],
                "total_sec": ["mean", "std"],
                "compile_sec_total": ["mean", "std"],
                "ground_sec_total": ["mean", "std"],
                "solve_sec_total": ["mean", "std"],
                "sid": ["mean", "std"],
                "peak_mem_bytes": ["mean", "std"],
                "peak_mem_mb": ["mean", "std"],
                "peak_after_solver_mb": ["mean", "std"],
                "peak_after_postprocess_mb": ["mean", "std"],
                "peak_after_compile_mb": ["mean", "std"],
                "peak_after_ground_mb": ["mean", "std"],
                "peak_after_solve_mb": ["mean", "std"],
                "timed_out": ["mean"],
                "solve_calls": ["mean", "std"],
                "ground_calls": ["mean", "std"],
                "rss_delta_mb": ["mean", "std"],
            }
        )
    )
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
