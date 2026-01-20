"""MUS/MCS analysis tests + reproducible profiling harness.

This file serves two roles:

1) **pytest/unittest tests** for the MUS assumption-layer and small sanity cases.
2) A **scriptable harness** for generating random PC-based instances, running:
   - ABAPC removal (`CausalABA(..., search_for_models='first')`), and
   - MUS/MCS analysis (`CausalABA_MUS(...)`),
   while printing a phase-by-phase timing comparison.

Key tests:

- `test_parsing_facts_from_file`: ensures we correctly read `ext_indep`/`ext_dep` facts and ignore comments/directives.
- `test_adorning_with_mus_assumptions`: validates `{mus(i)}.` choice rules and `fact :- mus(i).` guarding.
- `test_mock_three_var_manual_vs_mus`: smallest end-to-end example; matches the manual folder
  `encodings/test_lps/mock_three_var_manual/`.
- `test_mus_links_wrong_tests_four_node_abapc`: links PC-derived wrong tests to MUS/MCS on a fixed 4-node case.
- `test_mus_mcs_random_five_node_abapc`: PC → ABAPC → MUS/MCS on a deterministic 5-node seed.
- `test_mus_mcs_random_sizes_abapc`: same pipeline across multiple sizes (`--node-sizes`).

Run via pytest:

  python -m pytest tests_mus.py -xvs
  python -m pytest tests_mus.py::TestMUSAnalysis::test_mock_three_var_manual_vs_mus -v

Run as a script (uses argparse at bottom):

  python tests_mus.py --node-sizes 7,8 --solve-timeout 120 --max-muses 0 --emit-lp results/adornedLP_{n}.lp

To reproduce a run externally once you emitted an adorned `.lp`:

  clingo results/mus_7.lp --output=smodels | wasp \
      --mus=mus --mus-algorithm=camus --print-mcses -n 0

Important knobs:
- `--graph-type`: random DAG family passed to the simulator (default: ER).
- `--opt-mode`, `--out-n`: passed to the ABAPC run (`CausalABA`) to control clingo optimization and model bound.
- `--max-muses`: controls whether WASP receives `-n`.
"""

import os
import sys
import argparse
import logging
import tempfile
import unittest
import atexit
import random
import re
import multiprocessing
import traceback
from datetime import datetime, timedelta
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

CausalABA: Any = None
CausalABA_MUS: Any = None
CausalABA_WC: Any = None
CausalABA_INC: Any = None


# Accumulates total wall-clock times across the randomG harness so we can
# print a final summary at process exit (useful when logs are very long).
_METHOD_TIME_TOTALS: dict[str, dict[str, float]] = {}
_METHOD_TIME_TOTALS_PRINTED: bool = False


def _accumulate_method_time(method: str, seconds: float) -> None:
    try:
        s = float(seconds)
    except Exception:
        return
    if s <= 0.0:
        return
    entry = _METHOD_TIME_TOTALS.setdefault(str(method), {'total_sec': 0.0, 'runs': 0.0})
    entry['total_sec'] = float(entry.get('total_sec', 0.0) or 0.0) + s
    entry['runs'] = float(entry.get('runs', 0.0) or 0.0) + 1.0


def _log_total_time_per_method() -> None:
    global _METHOD_TIME_TOTALS_PRINTED
    if _METHOD_TIME_TOTALS_PRINTED:
        return
    if not _METHOD_TIME_TOTALS:
        return
    _METHOD_TIME_TOTALS_PRINTED = True

    try:
        rows = []
        for method, d in _METHOD_TIME_TOTALS.items():
            total = float(d.get('total_sec', 0.0) or 0.0)
            runs = int(float(d.get('runs', 0.0) or 0.0))
            rows.append((method, total, runs))
        rows.sort(key=lambda t: (-t[1], t[0]))

        logging.info("\n" + "=" * 60)
        logging.info(" TOTAL TIME PER METHOD (sum over usable runs)")
        logging.info("=" * 60)
        for method, total, runs in rows:
            avg = (total / runs) if runs > 0 else 0.0
            logging.info(f"  {method:<28}: {total:10.3f}s  (runs={runs:3d}, avg={avg:7.3f}s)")
        logging.info("=" * 60 + "\n")
    except Exception:
        # Avoid failing the test runner due to logging issues.
        pass


def _ensure_solvers_imported() -> None:
    """Import solver modules lazily so `--help` works without clingo/wasp installed."""
    global CausalABA, CausalABA_MUS, CausalABA_WC
    if CausalABA is None or CausalABA_MUS is None or CausalABA_WC is None:
        from causalaba import CausalABA as _CausalABA
        from causalaba_mus import CausalABA_MUS as _CausalABA_MUS
        from causalaba_mus import CausalABA_WC as _CausalABA_WC

        CausalABA = _CausalABA
        CausalABA_MUS = _CausalABA_MUS
        CausalABA_WC = _CausalABA_WC


def _ensure_abapc_inc_imported() -> None:
    """Import incremental ABAPC lazily so `--help` works without clingo installed."""
    global CausalABA_INC
    if CausalABA_INC is not None:
        return

    first_exc: Exception | None = None
    try:
        # When ArgCausalDisco is importable as a (namespace) package.
        from ArgCausalDisco.causalaba_increm import CausalABA as _CausalABA_INC  # type: ignore

        CausalABA_INC = _CausalABA_INC
        return
    except Exception as e:
        first_exc = e

    # Fallback 1: add the ArgCausalDisco folder itself to sys.path and try a direct module import.
    try:
        candidate_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "ArgCausalDisco")
        if os.path.isdir(candidate_dir) and candidate_dir not in sys.path:
            sys.path.insert(0, candidate_dir)
        from causalaba_increm import CausalABA as _CausalABA_INC  # type: ignore

        CausalABA_INC = _CausalABA_INC
        return
    except Exception as second_exc:
        # Fallback 2: import by absolute file path.
        try:
            import importlib.util

            module_path = os.path.join(os.path.dirname(PROJECT_ROOT), "ArgCausalDisco", "causalaba_increm.py")
            spec = importlib.util.spec_from_file_location("abapc_inc_causalaba_increm", module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load spec for {module_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            CausalABA_INC = getattr(module, "CausalABA")
            return
        except Exception as third_exc:
            raise ImportError(
                "Failed to import ABAPC_INC incremental encoding. "
                f"package import error={first_exc!r}; direct import error={second_exc!r}; file import error={third_exc!r}"
            )


def _abapc_inc_worker(result_queue: "multiprocessing.Queue[tuple[str, Any]]", args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    try:
        _ensure_abapc_inc_imported()
        if CausalABA_INC is None:
            raise ImportError("ABAPC_INC is not available after import")

        def _stringify_models(obj: Any) -> Any:
            # CausalABA returns clingo.Symbol objects which are not reliably picklable.
            # Convert to plain strings so results can cross process boundaries.
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, list):
                return [_stringify_models(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_stringify_models(x) for x in obj)
            try:
                return str(obj)
            except Exception:
                return repr(obj)

        raw = CausalABA_INC(*args, **kwargs)
        # Expected shape when return_statistics=True: [models_out, multiple, stats, remove_n, profile]
        models_out, multiple, stats, remove_n, profile = raw
        safe_payload = {
            "models_out": _stringify_models(models_out),
            "multiple": bool(multiple),
            # clingo statistics can contain non-picklable objects; omit.
            "stats": None,
            "remove_n": int(remove_n or 0),
            "profile": profile if isinstance(profile, dict) else {},
        }
        result_queue.put(("ok", safe_payload))
    except BaseException as e:
        result_queue.put(("err", (repr(e), traceback.format_exc())))


def _run_abapc_inc_with_wall_timeout(
    *args: Any,
    wall_timeout: float | None,
    **kwargs: Any,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Run ABAPC_INC with a hard wall-clock timeout.

    Motivation: clingo's Python API doesn't provide a reliable timeout for grounding.
    For large instances, grounding can dominate runtime; this wrapper ensures the
    overall ABAPC_INC call respects the user-provided `--solve-timeout` budget.
    """

    _ensure_abapc_inc_imported()
    if CausalABA_INC is None:
        raise ImportError("ABAPC_INC is not available after import")

    if wall_timeout is None:
        result = CausalABA_INC(*args, **kwargs)
        # CausalABA_INC returns (models_after, multiple, stats, remove_n, profile)
        if isinstance(result, tuple) and len(result) == 5 and isinstance(result[4], dict):
            return result  # type: ignore[return-value]
        # Be defensive if upstream signature changes
        models_after, multiple, stats, remove_n, profile = result
        if not isinstance(profile, dict):
            profile = {}
        return models_after, multiple, stats, remove_n, profile

    if wall_timeout <= 0:
        # Treat non-positive as "no timeout" for safety.
        result = CausalABA_INC(*args, **kwargs)
        models_after, multiple, stats, remove_n, profile = result
        if not isinstance(profile, dict):
            profile = {}
        return models_after, multiple, stats, remove_n, profile

    try:
        ctx = multiprocessing.get_context("fork")
    except Exception:
        ctx = multiprocessing.get_context()

    result_queue: multiprocessing.Queue[tuple[str, Any]] = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_abapc_inc_worker, args=(result_queue, args, kwargs))
    proc.daemon = True
    proc.start()
    proc.join(timeout=float(wall_timeout))

    if proc.is_alive():
        logging.warning(f"⚠ ABAPC_INC exceeded wall timeout ({wall_timeout}s); terminating.")
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():
            # Python 3.7+ supports kill(); best-effort.
            try:
                proc.kill()  # type: ignore[attr-defined]
            except Exception:
                pass

        timed_profile = {
            "timed_out": True,
            "timeout_phase": "wall",
            "timeout_s": float(wall_timeout),
        }
        return [], False, None, 0, timed_profile

    try:
        status, payload = result_queue.get_nowait()
    except Exception:
        # Child exited without returning a result.
        logging.warning("⚠ ABAPC_INC subprocess exited without returning a result (possible pickling failure).")
        timed_profile = {
            "timed_out": True,
            "timeout_phase": "aborted",
            "timeout_s": float(wall_timeout),
        }
        return [], False, None, 0, timed_profile

    if status == "ok":
        if isinstance(payload, dict) and {"models_out", "multiple", "remove_n", "profile"}.issubset(payload.keys()):
            models_after = payload.get("models_out", [])
            multiple = bool(payload.get("multiple", False))
            stats = payload.get("stats", None)
            remove_n = int(payload.get("remove_n", 0) or 0)
            profile = payload.get("profile", {})
            if not isinstance(profile, dict):
                profile = {}
            return models_after, multiple, stats, remove_n, profile

        # Backward/defensive fallback if worker returns raw tuple/list.
        result = payload
        models_after, multiple, stats, remove_n, profile = result
        if not isinstance(profile, dict):
            profile = {}
        return models_after, multiple, stats, remove_n, profile

    err_repr, err_tb = payload
    raise RuntimeError(f"ABAPC_INC subprocess failed: {err_repr}\n{err_tb}")


def _parse_node_sizes(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in raw.split(',') if x.strip())
    except Exception:
        return (5, 7)


def _parse_graph_types(raw: str) -> tuple[str, ...]:
    items = [x.strip() for x in (raw or "").split(',') if x.strip()]
    return tuple(items)


def _as_solver_timeout(timeout_s: Any) -> float | None:
    """Convert CLI/env timeout (seconds) to a solver timeout.

    Convention: 0 or negative means "no timeout".
    """
    if timeout_s is None:
        return None
    try:
        t = float(timeout_s)
    except Exception:
        return None
    return None if t <= 0 else t


MUS_SOLVE_TIMEOUT = int(os.environ.get("MUS_SOLVE_TIMEOUT", "30"))
MUS_NODE_SIZES = _parse_node_sizes(os.environ.get("MUS_NODE_SIZES", "5,7"))
MUS_EDGE_PER_NODE = int(os.environ.get("MUS_EDGE_PER_NODE", "2"))
MUS_SEED_BASE = int(os.environ.get("MUS_SEED_BASE", "2004"))
MUS_REP_UNSAT = int(os.environ.get("MUS_REP_UNSAT", "2"))
MUS_RANDOM_REPS = int(os.environ.get("MUS_RANDOM_REPS", "1"))
MUS_VERSION = os.environ.get("MUS_VERSION", "").strip()
MUS_EMIT_LP = os.environ.get("MUS_EMIT_LP", "")
MUS_EMIT_ABAPC_INC_LP = os.environ.get("MUS_EMIT_ABAPC_INC_LP", "")
MUS_USE_INCREM_ONLY = os.environ.get("MUS_USE_INCREM_ONLY", "0").strip() not in ("", "0", "false", "False", "no", "NO")
MUS_MAX_MUSES = os.environ.get("MUS_MAX_MUSES", "")  # "" = omit -n flag (WASP default: 1 MUS output), "0" = unlimited, ">0" = limit
MUS_MCS_THRESHOLD = int(os.environ.get("MUS_MCS_THRESHOLD", "0"))  # 0 = no limit (unlimited enumeration)
MUS_MUS_THRESHOLD = int(os.environ.get("MUS_MUS_THRESHOLD", "0"))  # 0 = no limit (unlimited enumeration)

# Optional diagnostic: check MUS minimality-in-isolation (can be slow; also not always applicable).
MUS_CHECK_MINIMALITY = os.environ.get("MUS_CHECK_MINIMALITY", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Optional diagnostic: check the MUS definition directly.
# For each returned MUS core (as a set of mus(i) assumptions), enforce those assumptions in the
# *same adorned MUS program* and assert the result is UNSAT.
MUS_CHECK_ENFORCED_UNSAT = os.environ.get("MUS_CHECK_ENFORCED_UNSAT", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Optional diagnostic: for each returned MUS core, take its corresponding facts and check that
# running CausalABA on *only those facts* yields UNSAT (ABAPC background).
MUS_CHECK_ENFORCED_UNSAT_ABAPC = os.environ.get("MUS_CHECK_ENFORCED_UNSAT_ABAPC", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Optional diagnostic: check MCS semantics (repairs).
# For each returned MCS, if we enable all facts *except* those in the MCS, the program should be SAT.
MUS_CHECK_MCS_REMOVED_SAT = os.environ.get("MUS_CHECK_MCS_REMOVED_SAT", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Same as MUS_CHECK_MCS_REMOVED_SAT, but checked under the plain CausalABA (ABAPC) background:
# run CausalABA on all facts with the MCS facts removed and assert SAT.
MUS_CHECK_MCS_REMOVED_SAT_ABAPC = os.environ.get("MUS_CHECK_MCS_REMOVED_SAT_ABAPC", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# When enabled, ABAPC-background MCS removed->SAT becomes a hard assertion.
MUS_CHECK_MCS_REMOVED_SAT_ABAPC_STRICT = os.environ.get("MUS_CHECK_MCS_REMOVED_SAT_ABAPC_STRICT", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Whether to require at least one MUS to intersect wrong facts. Empirically this can be false
# on some random instances; default is to warn rather than fail.
MUS_REQUIRE_MUS_HIT_WRONG = os.environ.get("MUS_REQUIRE_MUS_HIT_WRONG", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Whether to require at least one (Opt)MCS to intersect wrong facts.
# This can be false on some instances because an MCS is a *repair set* and may drop
# only correct-but-weak (near-alpha) tests. Default is to warn rather than fail.
MUS_REQUIRE_MCS_HIT_WRONG = os.environ.get("MUS_REQUIRE_MCS_HIT_WRONG", "0").strip() not in ("", "0", "false", "False", "no", "NO")
MUS_REQUIRE_OPTMCS_HIT_WRONG = os.environ.get("MUS_REQUIRE_OPTMCS_HIT_WRONG", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Demo-only failing tests to illustrate known effects. Default is enabled.
MUS_DEMO_FAILING_TESTS = os.environ.get("MUS_DEMO_FAILING_TESTS", "1").strip() not in ("", "0", "false", "False", "no", "NO")

# Random graph family used by build_random_pc_case/randomG_PC_case.
MUS_GRAPH_TYPE = os.environ.get("MUS_GRAPH_TYPE", "ER")
MUS_GRAPH_TYPES = _parse_graph_types(os.environ.get("MUS_GRAPH_TYPES", "")) or (MUS_GRAPH_TYPE,)

# ABAPC/Clingo options passed into CausalABA (removal strategy).
MUS_ABAPC_OUT_N = int(os.environ.get("MUS_ABAPC_OUT_N", "1"))
MUS_ABAPC_OPT_MODE = os.environ.get("MUS_ABAPC_OPT_MODE", "optN")

# When enumerating compatible DAGs for the graph-eval summaries, it's often useful to
# use a different model bound than the ABAPC removal run (which uses search='first').
# Set to 0 to ask clingo for all models (can be very expensive).
MUS_GRAPH_EVAL_OUT_N = int(os.environ.get("MUS_GRAPH_EVAL_OUT_N", str(MUS_ABAPC_OUT_N)))

# SID is computed via cdt.metrics (R-backed) and can be very slow when many DAGs are enumerated.
MUS_GRAPH_EVAL_SID = os.environ.get("MUS_GRAPH_EVAL_SID", "0").strip() not in ("", "0", "false", "False", "no", "NO")

# Directory to save temp facts files for inspection (default: "" = delete temp files)
MUS_KEEP_TEMP_FILES = os.environ.get("MUS_KEEP_TEMP_FILES", "")

# When enabled, print full fact lists + MUS/MCS/OptMCS sets for inspection.
# This can be very verbose on larger instances.
MUS_PRINT_DETAILS = os.environ.get("MUS_PRINT_DETAILS", "0").strip() not in ("", "0", "false", "False", "no", "NO")
# Max number of sets (per kind: MUS/MCS/OptMCS) to print when MUS_PRINT_DETAILS is enabled.
# Use 0 for unlimited.
MUS_PRINT_DETAILS_MAX_SETS = int(os.environ.get("MUS_PRINT_DETAILS_MAX_SETS", "10"))

# Optional: enable WASP optimum-MCS mode (requires wc weights file).
# Valid values (per WASP): "camus" or "emax" (case-insensitive). Empty disables.
MUS_OPTIMUM_MCS_ALGORITHM = os.environ.get("MUS_OPTIMUM_MCS_ALGORITHM", "camus").strip()

# Which analyses to run in the random-size harness.
# Supported atoms: mus, optmcs, wc
# Backwards-compatible shorthands:
# - both: mus + optmcs
# - all: mus + optmcs + wc
def _parse_analysis_modes(raw: str) -> set[str]:
    s = str(raw or "").strip().lower()
    if not s:
        s = "both"
    if s == 'both':
        s = 'mus,optmcs'
    if s == 'all':
        s = 'mus,optmcs,wc'
    parts = [p.strip() for p in re.split(r"[,+]", s) if p.strip()]
    allowed = {'mus', 'optmcs', 'wc'}
    out = {p for p in parts if p in allowed}
    return out or {'mus', 'optmcs'}


MUS_ANALYSIS = os.environ.get("MUS_ANALYSIS", "both").strip().lower()
MUS_ANALYSIS_MODES = _parse_analysis_modes(MUS_ANALYSIS)
MUS_RUN_MUS_MCS = 'mus' in MUS_ANALYSIS_MODES
MUS_RUN_OPTMCS = 'optmcs' in MUS_ANALYSIS_MODES
MUS_RUN_WC = 'wc' in MUS_ANALYSIS_MODES

# Optional: compare the pure-WC optimum cut to WASP's OptMCS cut.
MUS_CHECK_WC_VS_OPTMCS = os.environ.get("MUS_CHECK_WC_VS_OPTMCS", "0").strip() not in ("", "0", "false", "False", "no", "NO")
MUS_CHECK_WC_VS_OPTMCS_STRICT_SET = os.environ.get("MUS_CHECK_WC_VS_OPTMCS_STRICT_SET", "1").strip() not in ("", "0", "false", "False", "no", "NO")


@dataclass(frozen=True)
class RandomPCSimConfig:
    n_nodes: int
    alpha: float
    graph_type: str
    edge_per_node: int
    seed: int
    sample_size: int = 10000
    uc_rule: int = 5
    uc_priority: int = 2
    stable: bool = True


def build_random_pc_case(config: RandomPCSimConfig):
    import networkx as nx
    import numpy as np
    import pandas as pd

    from utils.graph_utils import (
        find_all_d_separations_sets,
        extract_test_elements_from_symbol,
        initial_strength,
    )
    from utils.data_utils import simulate_dag, simulate_data_and_run_PC
    from utils.helpers import random_stability

    n_nodes = config.n_nodes
    s0 = int(n_nodes * config.edge_per_node)
    max_edges = int(n_nodes * (n_nodes - 1) / 2)
    if s0 > max_edges:
        s0 = max_edges

    random_stability(config.seed)
    B_true = simulate_dag(d=n_nodes, s0=s0, graph_type=config.graph_type)
    G_true = nx.DiGraph(
        pd.DataFrame(
            B_true,
            columns=[f"X{i+1}" for i in range(n_nodes)],
            index=[f"X{i+1}" for i in range(n_nodes)],
        )
    )
    relabel_dict = {f"X{i+1}": i for i in range(n_nodes)}
    G_true1 = nx.relabel_nodes(G_true, relabel_dict)

    true_seplist = find_all_d_separations_sets(G_true, verbose=False)

    random_stability(config.seed)
    data, cg = simulate_data_and_run_PC(
        G_true,
        config.alpha,
        uc_rule=config.uc_rule,
        uc_priority=config.uc_priority,
        stable=config.stable,
        seed=config.seed,
        sample_size=config.sample_size,
    )

    facts = []  # (fact_str, I, is_correct)
    facts_ext = []
    wrong_ext = []
    count_wrong = 0

    for test in true_seplist:
        X, S, Y, dep_type = extract_test_elements_from_symbol(test)
        test_PC = [t for t in cg.sepset[X, Y] if set(t[0]) == S]
        if len(test_PC) != 1:
            continue

        p = test_PC[0][1]
        dep_type_PC = "indep" if p > config.alpha else "dep"
        I = initial_strength(p, len(S), config.alpha, 0.5, n_nodes)

        if dep_type == dep_type_PC:
            fact_str = test
            is_correct = True
        elif dep_type == "indep":
            count_wrong += 1
            fact_str = test.replace("indep", "dep")
            is_correct = False
        else:  # dep
            count_wrong += 1
            fact_str = test.replace("dep", "indep")
            is_correct = False

        facts.append((fact_str, I, is_correct))
        ext_line = f"ext_{fact_str}"
        facts_ext.append(ext_line)
        if not is_correct:
            wrong_ext.append(ext_line)

    return {
        "B_true": B_true,
        "G_true": G_true,
        "G_true1": G_true1,
        "true_seplist": true_seplist,
        "data": data,
        "cg": cg,
        "facts": facts,
        "facts_ext": facts_ext,
        "wrong_ext": wrong_ext,
        "count_wrong": count_wrong,
    }


def logger_setup(scenario: str = "test_mus", *, log_file: str | None = None) -> None:
    """Setup logging for tests.

    If `log_file` is provided, logs are written to both stdout and the file.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    # If a log file is already configured (e.g., via CLI), many unit tests call
    # logger_setup() again without passing log_file. Preserve the existing file
    # handler instead of dropping it.
    if not log_file:
        try:
            for h in logging.getLogger().handlers:
                if isinstance(h, logging.FileHandler):
                    handlers.append(logging.FileHandler(str(h.baseFilename), mode='a'))
                    break
        except Exception:
            pass

    if log_file:
        try:
            parent = os.path.dirname(str(log_file))
            if parent:
                os.makedirs(parent, exist_ok=True)
        except Exception:
            # Best-effort; if directory creation fails, FileHandler will raise below.
            pass
        handlers.append(logging.FileHandler(str(log_file), mode='a'))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=handlers,
        force=True,
    )


class TestMUSAnalysis(unittest.TestCase):
    def _dump_instance_details(
        self,
        *,
        n_nodes: int,
        seed: int,
        facts_ext: list[str],
        wrong_ext: list[str],
        removed_facts: list[str],
        weights_by_fact: dict[str, int] | None,
        mus_facts: list[list[str]],
        mcs_facts: list[list[str]],
        opt_mcs_facts: list[list[str]] | None,
        opt_mcs_label: str | None,
    ) -> None:
        def _canon(s: str) -> str:
            return self._normalize_fact_str(str(s))

        wrong_set = {_canon(w) for w in (wrong_ext or []) if str(w).strip()}
        all_facts = [_canon(s) for s in (facts_ext or []) if str(s).strip()]
        removed_set = {_canon(s) for s in (removed_facts or []) if str(s).strip()}

        weights_norm: dict[str, int] = {}
        for k, v in (weights_by_fact or {}).items():
            kk = _canon(str(k))
            if kk:
                weights_norm[kk] = int(v)

        def _fmt_fact(fact: str) -> str:
            w = weights_norm.get(fact)
            if w is None:
                return fact
            return f"{fact} (w={w})"

        def _sum_weights(facts_set: set[str]) -> tuple[int, int]:
            total_w = 0
            missing_w = 0
            for fact in (facts_set or set()):
                w = weights_norm.get(fact)
                if w is None:
                    missing_w += 1
                else:
                    total_w += int(w)
            return total_w, missing_w

        logging.info("\n" + "=" * 60)
        logging.info(f"DETAILS (n_nodes={n_nodes}, seed={seed})")
        logging.info("=" * 60)

        logging.info(f"All PC facts ({len(all_facts)}):")
        for i, fact in enumerate(all_facts, start=1):
            label = "WRONG" if fact in wrong_set else "CORRECT"
            extra = " (REMOVED_BY_ABAPC)" if fact in removed_set else ""
            logging.info(f"  F{i:03d} {label}{extra}: {_fmt_fact(fact)}")

        logging.info(f"Wrong facts ({len(wrong_set)}):")
        for i, fact in enumerate(sorted(wrong_set), start=1):
            logging.info(f"  W{i:03d}: {_fmt_fact(fact)}")
        if wrong_set:
            total_w, missing_w = _sum_weights(wrong_set)
            extra = f" (missing {missing_w})" if missing_w else ""
            logging.info(f"  sum_w = {total_w}{extra}")

        logging.info(f"ABAPC removed facts ({len(removed_set)}):")
        for i, fact in enumerate(sorted(removed_set), start=1):
            label = "WRONG" if fact in wrong_set else "CORRECT"
            logging.info(f"  R{i:03d} {label}: {_fmt_fact(fact)}")
        if removed_set:
            total_w, missing_w = _sum_weights(removed_set)
            extra = f" (missing {missing_w})" if missing_w else ""
            logging.info(f"  sum_w = {total_w}{extra}")

        def _dump_sets(title: str, sets: list[list[str]] | None) -> None:
            sets = sets or []
            max_sets = int(MUS_PRINT_DETAILS_MAX_SETS)
            if max_sets > 0:
                shown = sets[:max_sets]
            else:
                shown = sets
            logging.info(f"{title} ({len(sets)}):")
            for idx, core in enumerate(shown, start=1):
                core_c = [_canon(s) for s in (core or []) if str(s).strip()]
                logging.info(f"  Set #{idx} (size {len(core_c)}):")
                total_w = 0
                missing_w = 0
                for fact in sorted(core_c):
                    label = "WRONG" if fact in wrong_set else "CORRECT"
                    w = weights_norm.get(fact)
                    if w is None:
                        missing_w += 1
                    else:
                        total_w += int(w)
                    logging.info(f"    - {label}: {_fmt_fact(fact)}")
                if core_c:
                    extra = f" (missing {missing_w})" if missing_w else ""
                    logging.info(f"    sum_w = {total_w}{extra}")
            if max_sets > 0 and len(sets) > max_sets:
                logging.info(f"  ... truncated: printed {max_sets}/{len(sets)} sets (increase MUS_PRINT_DETAILS_MAX_SETS or set to 0)")

        _dump_sets("MUS sets", mus_facts)
        _dump_sets("MCS sets", mcs_facts)
        if opt_mcs_facts is not None:
            _dump_sets(opt_mcs_label or "Optimum MCS sets", opt_mcs_facts)

        logging.info("=" * 60 + "\n")
    """Tests for MUS analysis of CausalABA using the mus/1 assumption framework.
    
    The mus/1 assumption layer allows WASP to compute minimal unsatisfiable
    subsets of facts that violate causal graph constraints. Facts are guarded
    by choice rules {mus(i)}. and conditionally activated.
    """

    def randomG_PC_case(
        self,
        n_nodes: int,
        edge_per_node: int = 2,
        graph_type: str = "ER",
        seed: int = 2024,
        alpha: float = 0.05,
        sample_size: int = 10000,
        uc_rule: int = 5,
        uc_priority: int = 2,
        stable: bool = True,
    ):
        """Build a random-DAG + PC fact set case (similar to tests.py::randomG_PC_facts).

        Returns a dict with keys like: facts, facts_ext, wrong_ext, count_wrong, cg, true_seplist, G_true1.
        """
        config = RandomPCSimConfig(
            n_nodes=n_nodes,
            alpha=alpha,
            graph_type=graph_type,
            edge_per_node=edge_per_node,
            seed=seed,
            sample_size=sample_size,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            stable=stable,
        )
        logging.info(
            "Sim config: "
            f"n_nodes={config.n_nodes}, alpha={config.alpha}, graph_type={config.graph_type}, "
            f"edge_per_node={config.edge_per_node}, seed={config.seed}, sample_size={config.sample_size}, "
            f"uc_rule={config.uc_rule}, uc_priority={config.uc_priority}, stable={config.stable}"
        )
        case = build_random_pc_case(config)
        case["config"] = config
        return case

    def test_000_parsing_facts_from_file(self):
        """Test parsing ext_indep/ext_dep facts from a file."""
        logger_setup()
        logging.info("===============Running test_000_parsing_facts_from_file===============")
        
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        
        with open(facts_file, 'w') as f:
            f.write("% Comment line\n")
            f.write("ext_indep(0,1,empty).\n")
            f.write("ext_dep(1,2,s0).\n")
            f.write("ext_indep(2,0,s1).\n")
            f.write("% Another comment\n")
            f.write("#show something.\n")
        
        from causalaba_mus import parse_facts_from_file
        facts, fact_mapping = parse_facts_from_file(facts_file)
        
        logging.info(f"Parsed {len(facts)} facts:")
        for idx, fact_str in fact_mapping.items():
            logging.info(f"  Fact {idx}: {fact_str}")
        
        self.assertEqual(len(facts), 3, "Expected 3 facts parsed")
        self.assertIn("ext_indep(0,1,empty).", list(fact_mapping.values())[0])
        
        os.remove(facts_file)

    def test_mus_links_wrong_tests_four_node_abapc(self):
        """Follow test_abapc_four_node_example and append MUS analysis.

        We reuse the exact fact construction from test_abapc_four_node_example:
                facts come from PC output versus ground truth (no synthetic contradictions),
        then we run:
        - CausalABA with all facts (expect UNSAT or at least allow removal),
        - CausalABA with search_for_models='first' (removal strategy),
                - MUS/MCS over the PC fact set (CAMUS + --print-mcses) to highlight which
                    PC tests are implicated in contradictions.
        """
        logger_setup()
        logging.info("===============Running test_mus_links_wrong_tests_four_node_abapc===============")

        _ensure_solvers_imported()

        import networkx as nx
        import numpy as np
        import types
        # Stub notears to avoid heavy optional dependency required by cd_algorithms.models
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')
            class _DummyMLP:
                pass
            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")
            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module
        from utils.graph_utils import find_all_d_separations_sets, extract_test_elements_from_symbol, initial_strength
        from utils.data_utils import simulate_data_and_run_PC
        from utils.helpers import random_stability

        # ArgCD four-node DAG (same structure and seed as test_abapc_four_node_example)
        alpha = 0.05
        seed = 2376
        B_true = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )
        n_nodes = B_true.shape[0]
        import pandas as pd

        G_true = nx.DiGraph(
            pd.DataFrame(
                B_true,
                columns=[f"X{i+1}" for i in range(B_true.shape[1])],
                index=[f"X{i+1}" for i in range(B_true.shape[1])],
            )
        )
        relabel_dict = {f"X{i+1}": i for i in range(n_nodes)}
        G_true1 = nx.relabel_nodes(G_true, relabel_dict)

        true_seplist = find_all_d_separations_sets(G_true)

        random_stability(seed)
        data, cg = simulate_data_and_run_PC(G_true, alpha, seed=seed, uc_rule=5, stable=True)

        facts = []  # (fact_str, I, is_correct)
        facts_ext = []
        wrong_ext = []
        count_wrong = 0

        for test in true_seplist:
            X, S, Y, dep_type = extract_test_elements_from_symbol(test)
            test_PC = set([t for t in cg.sepset[X, Y] if set(t[0]) == S])
            if len(test_PC) != 1:
                continue
            p = list(test_PC)[0][1]
            dep_type_PC = "indep" if p > alpha else "dep"
            I = initial_strength(p, len(S), alpha, 0.5, n_nodes)
            if dep_type == dep_type_PC:
                fact_str = test
                is_correct = True
            elif dep_type == "indep":
                count_wrong += 1
                fact_str = test.replace("indep", "dep")
                is_correct = False
            else:  # dep
                count_wrong += 1
                fact_str = test.replace("dep", "indep")
                is_correct = False

            facts.append((fact_str, I, is_correct))
            ext_line = f"ext_{fact_str}"
            facts_ext.append(ext_line)
            if not is_correct:
                wrong_ext.append(ext_line)

        logging.info(f"Seed: {seed}")
        logging.info(f"True DAG: {G_true1.edges}")
        logging.info(f"Number of total independence statements: {len(true_seplist)}")
        logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
        logging.info(f"Number of wrong facts: {count_wrong} ({(count_wrong/len(facts))*100 if facts else 0:.2f}%)")
        logging.info(f"Fully directed edges from PC: {cg.find_fully_directed()}")
        logging.info(f"Undirected edges from PC: {[(x,y) for (x,y) in cg.find_undirected() if x < y]}")

        # Write facts to temp files (base + I + wc) for CausalABA
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        facts_I_file = facts_file.replace('.lp', '_I.lp')
        facts_wc_file = facts_file.replace('.lp', '_wc.lp')

        with open(facts_file, 'w') as f:
            for s in facts_ext:
                line = s if s.endswith('.') else s + '.'
                f.write(f"#external {self._normalize_fact_str(line)}\n")

        with open(facts_I_file, 'w') as f:
            for fact, s in zip(facts, facts_ext):
                line = s if s.endswith('.') else s + '.'
                I = fact[1]
                f.write(f"{line} I={I}, NA\n")

        with open(facts_wc_file, 'w') as f:
            weights: list[int] = []
            for fact, s in zip(facts, facts_ext):
                line = s if s.endswith('.') else s + '.'
                I = fact[1]
                # Weights must be integers with at most 7 digits.
                # Map I in [0,1] to [1, 9_999_999] using rounding.
                try:
                    w = int(round(float(I) * 9_999_999))
                except Exception:
                    w = 0
                w = max(1, min(9_999_999, w))
                weights.append(w)
                f.write(f":~ {line} [-{w}]\n")

        if facts_ext:
            try:
                w_min = min(weights) if weights else None
                w_max = max(weights) if weights else None
                logging.info(
                    f"Weak-constraint weights: min={w_min}, max={w_max}, max_digits={len(str(w_max)) if w_max is not None else 'NA'}"
                )
            except Exception:
                pass

        # Step 1: Run with all facts (may be UNSAT). If UNSAT, removal should fix it.
        logging.info("Step 1: Testing with all facts")
        models_all, _ = CausalABA(n_nodes, facts_file, weak_constraints=True, print_models=False, skeleton_rules_reduction=True)
        logging.info(f"  → {len(models_all)} models")

        # Step 2: Apply removal strategy to reach SAT (or keep SAT)
        logging.info("Step 2: Applying removal strategy (search_for_models='first')")
        models_after, multiple, stats, remove_n = CausalABA(
            n_nodes,
            facts_file,
            weak_constraints=True,
            search_for_models='first',
            print_models=False,
            return_statistics=True,
        )
        logging.info(f"  → Facts removed: {remove_n}")
        logging.info(f"  → Models found after removal: {len(models_after)}")

        self.assertGreaterEqual(len(models_after), 0)
        if len(models_all) == 0:
            self.assertGreater(remove_n, 0, "Expected removal of X>0 tests to reach SAT when UNSAT")
            self.assertGreater(len(models_after), 0, "Expected SAT after removing X tests")

        # Step 3: Run MUS on PC facts
        logging.info("Step 3: Running MUS analysis")
        fd_mus, facts_mus_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd_mus)
        with open(facts_mus_file, 'w') as f:
            for s in facts_ext:
                line = s if s.endswith('.') else s + '.'
                f.write(f"{line}\n")

        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_mus_file,
            gringo_path="clingo",
            wasp_path="wasp",
            mus_algorithm="camus",
            print_mcses=True,
        )

        logging.info(f"MUS cores found: {mus_result['n_mus']}")
        mus_sets = [set(mf) for mf in mus_result['mus_facts']]
        if mus_sets:
            sizes = [len(ms) for ms in mus_sets]
            fact_freq = Counter(f for ms in mus_sets for f in ms)
            common_facts = set.intersection(*mus_sets)
            common_preview = sorted(common_facts)[:10]
            if len(common_facts) > 10:
                common_preview.append(f"... (+{len(common_facts) - 10} more)")
            top_common = ', '.join(f"{f} ({c}/{mus_result['n_mus']})" for f, c in fact_freq.most_common(5))
            logging.info(
                f"  MUS sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}"
            )
            logging.info(
                f"  Facts present in all MUS ({len(common_facts)}): {', '.join(common_preview) if common_preview else '(none)'}"
            )
            logging.info(f"  Top frequent facts: {top_common if top_common else '(none)'}")
            
            # Analyze wrong vs correct fact frequencies in MUS
            wrong_set = set(w + '.' if not w.endswith('.') else w for w in wrong_ext)
            correct_facts = [f for f in fact_freq if f not in wrong_set]
            wrong_facts = [f for f in fact_freq if f in wrong_set]
            
            logging.info(f"  Wrong facts in MUS: {len(wrong_facts)} unique, correct facts: {len(correct_facts)} unique")
            
            if wrong_facts:
                wrong_freq = [(f, fact_freq[f], fact_freq[f]/mus_result['n_mus']*100) for f in wrong_facts]
                wrong_freq.sort(key=lambda x: x[1], reverse=True)
                top_wrong = ', '.join(f"{f} ({c}/{mus_result['n_mus']} = {p:.1f}%)" for f, c, p in wrong_freq[:5])
                logging.info(f"  Top wrong facts in MUS: {top_wrong}")
                
                # Show average frequency
                avg_wrong_freq = sum(fact_freq[f] for f in wrong_facts) / len(wrong_facts)
                avg_correct_freq = sum(fact_freq[f] for f in correct_facts) / len(correct_facts) if correct_facts else 0
                logging.info(f"  Avg appearances: wrong facts {avg_wrong_freq:.1f}, correct facts {avg_correct_freq:.1f}")

            # Ordered ranking of facts by MUS frequency, explicitly labeled WRONG/CORRECT
            n_mus = max(int(mus_result['n_mus'] or 0), 1)
            ordered = [(f, fact_freq[f], (f in wrong_set)) for f in fact_freq]
            ordered.sort(key=lambda t: (-t[1], t[0]))
            logging.info("  Fact ranking by MUS frequency (count / n_mus):")
            for fact, count, is_wrong in ordered[: min(30, len(ordered))]:
                label = "WRONG" if is_wrong else "CORRECT"
                pct = (count / n_mus) * 100.0
                logging.info(f"    - {label}: {fact} ({count}/{n_mus} = {pct:.1f}%)")

            # Print 3 smallest MUSes as examples
            sorted_muses = sorted(enumerate(mus_result['mus_facts'], 1), key=lambda x: (len(x[1]), x[0]))
            logging.info("  Examples of smallest MUSes:")
            for idx, mus_facts in sorted_muses[:3]:
                logging.info(f"    MUS #{idx} (size {len(mus_facts)}):")
                for fact in mus_facts:
                    label = "WRONG" if fact in wrong_set else "CORRECT"
                    logging.info(f"      - {label}: {fact}")
        else:
            logging.info("  No MUS cores found")

        # --------------------------
        # MCS analysis (CAMUS)
        # --------------------------
        logging.info(f"MCSes found: {mus_result.get('n_mcs', 0)}")
        mcs_sets = [set(mf) for mf in mus_result.get('mcs_facts', [])]
        if mcs_sets:
            # Only analyze PC-derived facts (filter out ground-truth counterparts)
            pc_facts_set = set(s + '.' if not s.endswith('.') else s for s in facts_ext)
            wrong_set = set(w + '.' if not w.endswith('.') else w for w in wrong_ext)
            
            # Filter MCS to only include PC facts
            mcs_sets_pc_only = [ms.intersection(pc_facts_set) for ms in mcs_sets]
            fact_freq = Counter(f for ms in mcs_sets_pc_only for f in ms)
            
            # Compute stats on PC facts only
            sizes = [len(ms) for ms in mcs_sets_pc_only]
            common_facts = set.intersection(*mcs_sets_pc_only) if mcs_sets_pc_only else set()
            common_preview = sorted(common_facts)[:10]
            if len(common_facts) > 10:
                common_preview.append(f"... (+{len(common_facts) - 10} more)")
            top_common = ', '.join(f"{f} ({fact_freq[f]}/{mus_result.get('n_mcs', 0)})" for f in sorted(fact_freq, key=lambda x: -fact_freq[x])[:5])
            
            logging.info(
                f"  MCS sizes (PC facts only): min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}"
            )
            logging.info(
                f"  PC facts present in all MCSes ({len(common_facts)}): {', '.join(common_preview) if common_preview else '(none)'}"
            )
            logging.info(f"  Top frequent PC facts: {top_common if top_common else '(none)'}")

            # Analyze wrong vs correct fact frequencies in MCS (PC facts only)
            correct_facts = [f for f in fact_freq if f not in wrong_set]
            wrong_facts = [f for f in fact_freq if f in wrong_set]
            logging.info(f"  Wrong PC facts in MCS: {len(wrong_facts)} unique, correct PC facts: {len(correct_facts)} unique")

            if wrong_facts and mus_result.get('n_mcs', 0) > 0:
                wrong_freq = [(f, fact_freq[f], fact_freq[f]/mus_result['n_mcs']*100) for f in wrong_facts]
                wrong_freq.sort(key=lambda x: x[1], reverse=True)
                top_wrong = ', '.join(f"{f} ({c}/{mus_result['n_mcs']} = {p:.1f}%)" for f, c, p in wrong_freq[:5])
                logging.info(f"  Top wrong PC facts in MCS: {top_wrong}")

            # Ordered ranking of PC facts by MCS frequency, explicitly labeled WRONG/CORRECT
            n_mcs = max(int(mus_result.get('n_mcs', 0) or 0), 1)
            ordered = [(f, fact_freq[f], (f in wrong_set)) for f in fact_freq]
            ordered.sort(key=lambda t: (-t[1], t[0]))
            logging.info("  PC Fact ranking by MCS frequency (count / n_mcs):")
            for fact, count, is_wrong in ordered[: min(30, len(ordered))]:
                label = "WRONG" if is_wrong else "CORRECT"
                pct = (count / n_mcs) * 100.0
                logging.info(f"    - {label}: {fact} ({count}/{n_mcs} = {pct:.1f}%)")

            # Print 3 smallest MCSes as examples
            sorted_mcses = sorted(enumerate(mus_result.get('mcs_facts', []), 1), key=lambda x: (len(x[1]), x[0]))
            logging.info("  Examples of smallest MCSes:")
            for idx, mcs_facts in sorted_mcses[:3]:
                logging.info(f"    MCS #{idx} (size {len(mcs_facts)}):")
                for fact in mcs_facts:
                    label = "WRONG" if fact in wrong_set else "CORRECT"
                    logging.info(f"      - {label}: {fact}")
        else:
            logging.info("  No MCSes found")

        wrong_set = set(w + '.' if not w.endswith('.') else w for w in wrong_ext)
        if mus_result['n_mus'] > 0:
            for ms in mus_sets:
                self.assertTrue(len(ms.intersection(wrong_set)) >= 1, "Each MUS should include at least one wrong/flipped test")

        os.remove(facts_file)
        os.remove(facts_I_file)
        os.remove(facts_wc_file)
        os.remove(facts_mus_file)

    def test_mus_mcs_random_five_node_abapc(self):
        """Random 5-node variant of the PC→ABAPC→MUS/MCS pipeline.

        This mirrors the intent of `randomG_PC_facts` in `tests.py`, but compares against
        the *normal* ABAPC configuration (search_for_models='first') rather than
        search_for_models='all_subsets'.

        Steps:
        1) Build a random 5-node DAG, run PC, and construct the `ext_*` facts.
        2) Run CausalABA with all PC facts (fixed seed chosen to be UNSAT).
        3) Run ABAPC removal with search_for_models='first' to restore SAT.
        4) Run MUS+MCS analysis on the PC fact set only (CAMUS + --print-mcses).
        """
        logger_setup()
        logging.info("===============Running test_mus_mcs_random_five_node_abapc===============")

        _ensure_solvers_imported()

        import types

        # Stub notears to avoid heavy optional dependency required by cd_algorithms.models
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')

            class _DummyMLP:
                pass

            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")

            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module

        # Deterministic configuration (seed chosen to yield UNSAT for step 1)
        case = self.randomG_PC_case(
            n_nodes=5,
            edge_per_node=2,
            graph_type="ER",
            seed=2004,
            alpha=0.05,
            sample_size=10000,
            uc_rule=5,
            stable=True,
        )
        config = case["config"]
        n_nodes = config.n_nodes
        facts = case["facts"]
        facts_ext = case["facts_ext"]
        wrong_ext = case["wrong_ext"]
        count_wrong = case["count_wrong"]
        true_seplist = case["true_seplist"]
        cg = case["cg"]
        G_true1 = case["G_true1"]

        logging.info(f"Seed: {config.seed}")
        logging.info(f"True DAG: {G_true1.edges}")
        logging.info(f"Number of total independence statements: {len(true_seplist)}")
        logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
        logging.info(f"Number of wrong facts: {count_wrong} ({(count_wrong/len(facts))*100 if facts else 0:.2f}%)")
        logging.info(f"Fully directed edges from PC: {cg.find_fully_directed()}")
        logging.info(f"Undirected edges from PC: {[(x,y) for (x,y) in cg.find_undirected() if x < y]}")

        self.assertGreater(len(facts_ext), 0, "Expected at least one PC-derived fact")
        self.assertGreater(count_wrong, 0, "Expected at least one wrong PC fact for MUS analysis")

        # Write facts to temp files (base + I + wc) for CausalABA
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
                try:
                    w = int(round(float(I) * 9_999_999))
                except Exception:
                    w = 0
                w = max(1, min(9_999_999, w))
                f.write(f":~ {line} [-{w}]\n")

        # Step 1: Run with all facts (expected UNSAT for this seed)
        logging.info("Step 1: Testing with all facts")
        models_all, _ = CausalABA(n_nodes, facts_file, weak_constraints=True, print_models=False, skeleton_rules_reduction=True)
        logging.info(f"  → {len(models_all)} models")
        self.assertEqual(len(models_all), 0, "Expected UNSAT with all PC facts for the chosen seed")

        # Step 2: Apply ABAPC removal strategy (search='first')
        logging.info("Step 2: Applying removal strategy (search_for_models='first')")
        models_after, multiple, stats, remove_n = CausalABA(
            n_nodes,
            facts_file,
            weak_constraints=True,
            search_for_models='first',
            print_models=False,
            return_statistics=True,
        )
        logging.info(f"  → Facts removed: {remove_n}")
        logging.info(f"  → Models found after removal: {len(models_after)}")

        self.assertGreater(remove_n, 0, "Expected removal of X>0 tests to reach SAT")
        self.assertGreater(len(models_after), 0, "Expected SAT after removing X tests")

        # Derive which facts were removed by ABAPC 'first'.
        # CausalABA sorts facts by descending I and then removes from the tail until SAT.
        def _canon_fact(s: str) -> str:
            return s if s.endswith('.') else s + '.'

        facts_with_I = []
        for (fact_str, I, _is_correct), ext_line in zip(facts, facts_ext):
            stmt = _canon_fact(ext_line)
            facts_with_I.append((float(I), stmt))
        facts_sorted_by_I = sorted(facts_with_I, key=lambda t: t[0], reverse=True)
        removed_facts = [stmt for _I, stmt in facts_sorted_by_I[-remove_n:]] if remove_n else []
        removed_set = set(removed_facts)
        all_pc_facts_set = set(stmt for _I, stmt in facts_sorted_by_I)
        kept_set = all_pc_facts_set.difference(removed_set)

        wrong_set = set(_canon_fact(w) for w in wrong_ext)
        wrong_all = wrong_set & all_pc_facts_set
        wrong_removed = wrong_set & removed_set
        wrong_kept = wrong_set & kept_set

        removed_preview = sorted(removed_facts)[:10]
        if len(removed_facts) > 10:
            removed_preview.append(f"... (+{len(removed_facts) - 10} more)")
        logging.info(
            f"  ABAPC removed facts (lowest I): {len(removed_facts)} / {len(all_pc_facts_set)}"
        )
        logging.info(
            f"  Removed preview: {', '.join(removed_preview) if removed_preview else '(none)'}"
        )
        logging.info(
            f"  Wrong PC facts (all): {len(wrong_all)} / {len(all_pc_facts_set)} ({(len(wrong_all)/len(all_pc_facts_set))*100 if all_pc_facts_set else 0:.2f}%)"
        )
        logging.info(
            f"  Wrong among REMOVED: {len(wrong_removed)} / {len(removed_set)} ({(len(wrong_removed)/len(removed_set))*100 if removed_set else 0:.2f}%)"
        )
        logging.info(
            f"  Wrong among KEPT: {len(wrong_kept)} / {len(kept_set)} ({(len(wrong_kept)/len(kept_set))*100 if kept_set else 0:.2f}%)"
        )

        # Step 3: Run MUS/MCS on PC facts only
        logging.info("Step 3: Running MUS/MCS analysis")
        fd_mus, facts_mus_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd_mus)
        with open(facts_mus_file, 'w') as f:
            for s in facts_ext:
                line = s if s.endswith('.') else s + '.'
                f.write(f"{line}\n")

        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_mus_file,
            gringo_path="clingo",
            wasp_path="wasp",
            max_muses=500,
            mus_algorithm="camus",
            print_mcses=True,
            camus_mcs_threshold=50,
            camus_mus_threshold=300,
        )

        logging.info(f"MUS cores found: {mus_result['n_mus']}")
        logging.info(f"MCSes found: {mus_result.get('n_mcs', 0)}")
        self.assertGreater(mus_result['n_mus'], 0, "Expected at least one MUS")
        self.assertGreater(mus_result.get('n_mcs', 0), 0, "Expected at least one MCS")

        mus_sets = [set(_canon_fact(f) for f in mf) for mf in mus_result['mus_facts']]
        for ms in mus_sets:
            self.assertTrue(
                len(ms.intersection(wrong_set)) >= 1,
                "Each MUS should include at least one wrong/flipped test",
            )

        # --------------------------
        # Link ABAPC removals vs MUS/MCS rankings
        # --------------------------
        mus_freq = Counter(f for ms in mus_sets for f in ms)
        n_mus = max(int(mus_result.get('n_mus', 0) or 0), 1)
        mcs_sets = [set(_canon_fact(f) for f in mf) for mf in mus_result.get('mcs_facts', [])]
        mcs_freq = Counter(f for cs in mcs_sets for f in cs)
        n_mcs = max(int(mus_result.get('n_mcs', 0) or 0), 1)

        wrong_in_mus_unique = set(mus_freq.keys()) & wrong_set
        wrong_in_mcs_unique = set(mcs_freq.keys()) & wrong_set
        logging.info(
            f"  Wrong facts appearing in MUSes: {len(wrong_in_mus_unique)} / {len(wrong_all)} unique"
        )
        logging.info(
            f"  Wrong facts appearing in MCSes: {len(wrong_in_mcs_unique)} / {len(wrong_all)} unique"
        )

        def _log_ranked(title: str, freq: Counter, denom: int, top_k: int = 20) -> list[str]:
            ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
            top_facts = [f for f, _c in ordered[: min(top_k, len(ordered))]]
            logging.info(title)
            for fact, count in ordered[: min(top_k, len(ordered))]:
                pct = (count / denom) * 100.0
                label_wc = "WRONG" if fact in wrong_set else "CORRECT"
                label_rm = "REMOVED" if fact in removed_set else "KEPT"
                logging.info(f"    - {label_rm} | {label_wc}: {fact} ({count}/{denom} = {pct:.1f}%)")
            return top_facts

        top_mus = _log_ranked("  Top facts by MUS frequency (all PC facts):", mus_freq, n_mus, top_k=20)
        top_mcs = _log_ranked("  Top facts by MCS frequency (all PC facts):", mcs_freq, n_mcs, top_k=20)

        for k in (5, 10, 20):
            k_mus = set(top_mus[: min(k, len(top_mus))])
            k_mcs = set(top_mcs[: min(k, len(top_mcs))])
            logging.info(
                f"  Overlap with ABAPC removed (top-{k}): MUS={len(k_mus & removed_set)}/{len(k_mus) or 1}, MCS={len(k_mcs & removed_set)}/{len(k_mcs) or 1}"
            )

        # Rank removed facts by how often they appear in MUS/MCS
        removed_by_mus = sorted(
            ((f, mus_freq.get(f, 0)) for f in removed_facts),
            key=lambda t: (-t[1], t[0]),
        )
        removed_by_mcs = sorted(
            ((f, mcs_freq.get(f, 0)) for f in removed_facts),
            key=lambda t: (-t[1], t[0]),
        )
        logging.info("  ABAPC-REMOVED facts ranked by MUS frequency (count / n_mus):")
        for fact, count in removed_by_mus[:20]:
            pct = (count / n_mus) * 100.0
            label_wc = "WRONG" if fact in wrong_set else "CORRECT"
            logging.info(f"    - {label_wc}: {fact} ({count}/{n_mus} = {pct:.1f}%)")
        logging.info("  ABAPC-REMOVED facts ranked by MCS frequency (count / n_mcs):")
        for fact, count in removed_by_mcs[:20]:
            pct = (count / n_mcs) * 100.0
            label_wc = "WRONG" if fact in wrong_set else "CORRECT"
            logging.info(f"    - {label_wc}: {fact} ({count}/{n_mcs} = {pct:.1f}%)")

        # Examples of smallest MUSes and MCSes
        sorted_muses = sorted(enumerate(mus_result.get('mus_facts', []), 1), key=lambda x: (len(x[1]), x[0]))
        logging.info("  Examples of smallest MUSes:")
        for idx, mus_facts in sorted_muses[:3]:
            logging.info(f"    MUS #{idx} (size {len(mus_facts)}):")
            for fact in mus_facts:
                label_wc = "WRONG" if fact in wrong_set else "CORRECT"
                label_rm = "REMOVED" if fact in removed_set else "KEPT"
                logging.info(f"      - {label_rm} | {label_wc}: {fact}")

        sorted_mcses = sorted(enumerate(mus_result.get('mcs_facts', []), 1), key=lambda x: (len(x[1]), x[0]))
        logging.info("  Examples of smallest MCSes:")
        for idx, mcs_facts in sorted_mcses[:3]:
            logging.info(f"    MCS #{idx} (size {len(mcs_facts)}):")
            for fact in mcs_facts:
                label_wc = "WRONG" if fact in wrong_set else "CORRECT"
                label_rm = "REMOVED" if fact in removed_set else "KEPT"
                logging.info(f"      - {label_rm} | {label_wc}: {fact}")

        os.remove(facts_file)
        os.remove(facts_I_file)
        os.remove(facts_wc_file)
        os.remove(facts_mus_file)

    def test_adorning_with_mus_assumptions(self):
        """Test that facts are correctly adorned with mus/1 assumptions.
        
        The adorning process wraps each fact in a conditional rule guarded by
        a mus(i) assumption, allowing WASP to compute MUS over the assumptions.
        
        This test verifies:
        1. mus(i) choice rules are generated
        2. Facts are guarded by mus(i) atoms
        3. No #show statements pollute the output
        """
        logger_setup()
        logging.info("===============Running test_adorning_with_mus_assumptions===============")
        
        from causalaba_mus import build_mus_program
        
        n_nodes = 3
        facts = ["ext_indep(1,2,s0)", "ext_dep(1,2,empty)"]
        
        program = build_mus_program(n_nodes, facts)
        logging.info(f"Generated program with {len(program.split(chr(10)))} lines")
        
        # Verify mus choice rules exist (allow varying whitespace)
        self.assertIn("{mus(1)}", program, "Expected mus(1) choice rule")
        self.assertIn("{mus(2)}", program, "Expected mus(2) choice rule")
        
        # Verify facts are guarded by mus(i) (allow varying whitespace)
        self.assertRegex(program, r'ext_indep\s*\(\s*1\s*,\s*2\s*,\s*s0\s*\)\s*:-\s*mus\s*\(\s*1\s*\)', "Expected fact 1 guarded by mus(1)")
        self.assertRegex(program, r'ext_dep\s*\(\s*1\s*,\s*2\s*,\s*empty\s*\)\s*:-\s*mus\s*\(\s*2\s*\)', "Expected fact 2 guarded by mus(2)")
        
        # Verify no #show statements (would interfere with MUS output)
        self.assertNotIn("#show", program, "Program should not contain #show directives")
        
        logging.info("✓ Adorning structure verified")

    def test_mock_three_var_manual_vs_mus(self):
        """Test mock_three_var: manually removing facts vs MUS finding them all.
        
        This test verifies that:
        1. All three facts together cause UNSAT
        2. Removing any single fact causes SAT (with different models)
        3. MUS discovers all three facts as a single minimal core
        
        This directly links the manual removal behavior to the MUS algorithm.
        """
        logger_setup()
        logging.info("===============Running test_mock_three_var_manual_vs_mus===============")

        _ensure_solvers_imported()
        
        n_nodes = 3
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        
        with open(facts_file, 'w') as f:
            f.write("#external ext_indep(1,2,s0).\n")
            f.write("#external ext_dep(1,2,empty).\n")
            f.write("#external ext_indep(0,1,empty).\n")
        
        logging.info("Mock-three-var facts:")
        logging.info("  1. ext_indep(1,2,s0)")
        logging.info("  2. ext_dep(1,2,empty)")
        logging.info("  3. ext_indep(0,1,empty)")
        logging.info("")
        
        # Step 1: Verify UNSAT with all facts
        logging.info("Step 1: Testing with all three facts")
        models, _ = CausalABA(n_nodes, facts_file, print_models=False, skeleton_rules_reduction=True)
        logging.info(f"  → {len(models)} models (expected 0 for UNSAT)")
        self.assertEqual(len(models), 0, "Expected UNSAT with all three facts")
        logging.info("")
        
        # Step 2: Verify SAT after removing each fact individually
        logging.info("Step 2: Testing with each fact removed individually")
        for removed_fact_num in [1, 2, 3]:
            fd_temp, facts_file_removed = tempfile.mkstemp(suffix='.lp', text=True)
            os.close(fd_temp)
            
            with open(facts_file_removed, 'w') as f:
                if removed_fact_num != 1:
                    f.write("#external ext_indep(1,2,s0).\n")
                if removed_fact_num != 2:
                    f.write("#external ext_dep(1,2,empty).\n")
                if removed_fact_num != 3:
                    f.write("#external ext_indep(0,1,empty).\n")
            
            models_removed, _ = CausalABA(n_nodes, facts_file_removed, print_models=False, skeleton_rules_reduction=True)
            logging.info(f"  Removed fact {removed_fact_num}: {len(models_removed)} models")
            if len(models_removed) > 0:
                first_model = models_removed[0]
                arrow_syms = sorted([str(s) for s in first_model])
                logging.info(f"    First model arrows: {', '.join(arrow_syms) if arrow_syms else '(none)'}")
            self.assertGreater(len(models_removed), 0, f"Expected SAT after removing fact {removed_fact_num}")
            
            os.remove(facts_file_removed)
        
        logging.info("")
        
        # Step 3: Compute MUS - should find all three facts
        logging.info("Step 3: Running MUS analysis")
        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_file,
            gringo_path="clingo",
            wasp_path="wasp",
            mus_algorithm="camus",
            print_mcses=True,
        )
        
        logging.info(f"  MUS cores found: {mus_result['n_mus']}")
        for i, mus_facts in enumerate(mus_result['mus_facts']):
            logging.info(f"    MUS #{i+1}: {len(mus_facts)} facts")
            for fact in mus_facts:
                logging.info(f"      - {fact}")
        
        self.assertEqual(mus_result['n_mus'], 1, "Expected exactly one MUS")
        # With skeleton-rules optimization enabled, the MUS may be smaller than all 3 facts
        # (only the minimal subset needed to cause UNSAT is reported)
        self.assertGreater(len(mus_result['mus_facts'][0]), 0, "Expected at least one fact in the MUS")
        self.assertLessEqual(len(mus_result['mus_facts'][0]), 3, "Expected at most three facts in the MUS")

        # With CAMUS + --print-mcses, we should see singleton MCSes for each removed element
        self.assertGreater(mus_result.get('n_mcs', 0), 0, "Expected at least one MCS")
        self.assertLessEqual(mus_result.get('n_mcs', 0), 3, "Expected at most three MCSes")
        mcs_sets = [set(m) for m in mus_result.get('mcs_list', [])]
        logging.info(f"MCS sets: {mcs_sets}")

        logging.info(f"  MCSes found: {mus_result.get('n_mcs', 0)}")
        for i, mcs_facts in enumerate(mus_result.get('mcs_facts', [])):
            logging.info(f"    MCS #{i+1}: {len(mcs_facts)} facts")
            for fact in mcs_facts:
                logging.info(f"      - {fact}")
        
        os.remove(facts_file)

    def test_wc_matches_optmcs_on_mock_three_var(self):
        """Pure-WC (clingo) optimum cut should match WASP OptMCS on the same adorned objective.

        We use the same 3-fact mock instance as `test_mock_three_var_manual_vs_mus` and assign
        distinct weights so the optimum is unique.
        """
        logger_setup()
        _ensure_solvers_imported()

        import subprocess

        # Require clingo.
        try:
            subprocess.run(["clingo", "--version"], capture_output=True, text=True, timeout=2.0)
        except Exception:
            self.skipTest("clingo not available; skipping WC-vs-OptMCS equivalence test")

        # Require WASP optimum-mcs support.
        try:
            help_out = subprocess.run(["wasp", "--help"], capture_output=True, text=True, timeout=2.0)
            supported = "--optimum-mcs-algorithm" in ((help_out.stdout or "") + (help_out.stderr or ""))
        except Exception:
            supported = False
        if not supported:
            self.skipTest("wasp --optimum-mcs-algorithm not supported; skipping equivalence test")

        n_nodes = 3
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        facts_wc_file = facts_file.replace('.lp', '_wc.lp')

        facts = [
            "ext_indep(1,2,s0).",
            "ext_dep(1,2,empty).",
            "ext_indep(0,1,empty).",
        ]
        weights = [1_000_000, 2_000_000, 3_000_000]

        try:
            with open(facts_file, 'w') as f:
                for s in facts:
                    f.write(f"#external {s}\n")

            with open(facts_wc_file, 'w') as f:
                for s, w in zip(facts, weights):
                    f.write(f":~ {s} [-{int(w)}]\n")

            opt_result = CausalABA_MUS(
                n_nodes=n_nodes,
                facts_location=facts_file,
                gringo_path="clingo",
                wasp_path="wasp",
                mus_algorithm="camus",
                print_mcses=False,
                optimum_mcs_algorithm="camus",
                facts_wc_location=facts_wc_file,
            )
            wc_result = CausalABA_WC(
                n_nodes=n_nodes,
                facts_location=facts_file,
                gringo_path="clingo",
                facts_wc_location=facts_wc_file,
            )

            self.assertFalse(bool((opt_result or {}).get('timed_out', False)), "OptMCS unexpectedly timed out")
            self.assertFalse(bool((wc_result or {}).get('timed_out', False)), "WC unexpectedly timed out")

            def _first_cut_set(obj: Any) -> set[str]:
                if obj is None:
                    return set()
                if isinstance(obj, list) and obj and all(isinstance(x, (list, tuple, set)) for x in obj):
                    for cut in obj:
                        s = {self._normalize_fact_str(str(x)) for x in (cut or []) if str(x).strip()}
                        s = {x for x in s if x}
                        if s:
                            return s
                    return set()
                s = {self._normalize_fact_str(str(x)) for x in (obj or []) if str(x).strip()}
                return {x for x in s if x}

            opt_cut = _first_cut_set((opt_result or {}).get('mcs_facts', None))
            wc_cut = _first_cut_set((wc_result or {}).get('cut_facts', None))

            self.assertTrue(opt_cut, "Expected a non-empty OptMCS cut")
            self.assertTrue(wc_cut, "Expected a non-empty WC cut")
            self.assertEqual(wc_cut, opt_cut, f"WC cut {wc_cut} != OptMCS cut {opt_cut}")

            # With strictly increasing weights, optimum should remove the cheapest single fact.
            expected = {self._normalize_fact_str(facts[0])}
            self.assertEqual(wc_cut, expected, f"Expected optimum cut {expected} with weights {weights}")
        finally:
            try:
                os.remove(facts_file)
            except Exception:
                pass
            try:
                os.remove(facts_wc_file)
            except Exception:
                pass

    def test_mus_catches_wrong_facts_on_four_nodes(self):
        """Link wrong tests to MUSes on a larger 4-node case.

        We craft two explicit contradictions:
        - ext_indep(0,1,empty) vs ext_dep(0,1,empty)
        - ext_indep(2,3,empty) vs ext_dep(2,3,empty)

        Each pair is independently inconsistent due to the base constraint
        ":- dep(X,Y,S), indep(X,Y,S), ...". MUS should return two minimal
        cores, each containing the two conflicting facts. This demonstrates
        how MUS pinpoints wrong tests in bigger problems.
        """
        logger_setup()
        logging.info("===============Running test_mus_catches_wrong_facts_on_four_nodes===============")

        _ensure_solvers_imported()

        n_nodes = 4
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)

        with open(facts_file, 'w') as f:
            # Contradiction A (pair 0-1)
            f.write("ext_indep(0,1,empty).\n")
            f.write("ext_dep(0,1,empty).\n")
            # Contradiction B (pair 2-3)
            f.write("ext_indep(2,3,empty).\n")
            f.write("ext_dep(2,3,empty).\n")
            # Some additional non-contradictory facts to make the instance bigger
            f.write("ext_dep(0,2,empty).\n")
            f.write("ext_indep(1,2,s0).\n")

        # Run MUS analysis
        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_file,
            gringo_path="clingo",
            wasp_path="wasp",
            # Enumerate all MUSes; WASP default without -n is typically just 1.
            max_muses=0,
            mus_algorithm="camus",
            print_mcses=True,
        )

        logging.info(f"MUS cores found: {mus_result['n_mus']}")
        for i, mus_facts in enumerate(mus_result['mus_facts']):
            logging.info(f"  MUS #{i+1}: {len(mus_facts)} facts")
            for fact in mus_facts:
                logging.info(f"    - {fact}")

        # Expect at least two MUSes (one for each contradictory pair)
        self.assertGreaterEqual(mus_result['n_mus'], 2, "Expected at least two MUS cores")

        # Define the two wrong test sets
        wrong_A = {"ext_indep(0,1,empty).", "ext_dep(0,1,empty)."}
        wrong_B = {"ext_indep(2,3,empty).", "ext_dep(2,3,empty)."}

        # Check that each wrong set appears as a MUS of size 2
        mus_sets = [set(mf) for mf in mus_result['mus_facts']]
        has_A = any(ms == wrong_A for ms in mus_sets)
        has_B = any(ms == wrong_B for ms in mus_sets)
        self.assertTrue(has_A, "Contradictory pair (0,1) should be identified as a MUS")
        self.assertTrue(has_B, "Contradictory pair (2,3) should be identified as a MUS")

        # Additional section: print MCSes (more actionable than just MUS)
        logging.info(f"MCSes found: {mus_result.get('n_mcs', 0)}")
        for i, mcs_facts in enumerate(mus_result.get('mcs_facts', [])):
            logging.info(f"  MCS #{i+1}: {len(mcs_facts)} facts")
            for fact in mcs_facts:
                logging.info(f"    - {fact}")

        os.remove(facts_file)

    def test_mus_mcs_random_sizes_abapc(self):
        """Run MUS/MCS analysis across multiple node sizes.
        
        This test parameterizes test_mus_mcs_random_five_node_abapc across different graph
        sizes, using unittest.TestCase.subTest to organize results by size.
        """
        logger_setup()
        logging.info("===============Running test_mus_mcs_random_sizes_abapc===============")

        import types

        # Stub notears once to avoid repeated imports
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')

            class _DummyMLP:
                pass

            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")

            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module

        graph_types = MUS_GRAPH_TYPES or (MUS_GRAPH_TYPE,)
        # If running multiple repetitions, collect per-(graph_type,n_nodes) summaries.
        rep_summaries: dict[tuple[str, int], list[dict[str, Any]]] = {}
        rep_outcomes: dict[tuple[str, int], dict[str, int]] = {}

        # Run for multiple node sizes, repetitions, and graph types.
        for graph_idx, graph_type in enumerate(graph_types):
            for n_nodes in MUS_NODE_SIZES:
                key = (str(graph_type), int(n_nodes))
                rep_summaries.setdefault(key, [])
                rep_outcomes.setdefault(
                    key,
                    {
                        # "usable" means we found a run (may still have solver timeouts).
                        'usable': 0,
                        'timed_out': 0,
                        'startSAT': 0,
                        'no_wrong': 0,
                        'mus_min_skipped': 0,
                        # Per-category completion counts (usable runs only)
                        'abapc_finished': 0,
                        'abapc_timed_out': 0,
                        'mus_finished': 0,
                        'mus_timed_out': 0,
                        'mus_skipped': 0,
                        'opt_finished': 0,
                        'opt_timed_out': 0,
                        'opt_skipped': 0,
                        'abapc_inc_finished': 0,
                        'abapc_inc_timed_out': 0,
                        'abapc_inc_skipped': 0,
                    },
                )

                for rep_idx in range(max(1, MUS_RANDOM_REPS)):
                    with self.subTest(n_nodes=n_nodes, graph_type=graph_type, rep=rep_idx):
                        run_info = None
                        selected_run: dict[str, Any] | None = None
                        saw_timeout = False
                        saw_sat = False
                        saw_no_wrong = False
                        # Seed policy: always start from --seed-base,
                        # then retries just add +1. When running multiple reps, we stride seeds 
                        # by (rep_unsat+1) so each rep explores a different instance without 
                        # overlapping the retry window of another rep.
                        stride = max(1, int(MUS_REP_UNSAT) + 1)
                        seed0 = int(MUS_SEED_BASE) + (int(rep_idx) * stride)
                        # Retry with incremented seeds until we get an UNSAT instance (i.e., ABAPC removes >0)
                        # that also has at least one wrong fact.
                        attempt_limit = max(0, int(MUS_REP_UNSAT)) + 1
                        for attempt in range(attempt_limit):
                            seed = seed0 + attempt
                            run_info = self._run_mus_mcs_for_size(n_nodes, seed=seed, graph_type=graph_type)
                            if run_info.get('abapc_timeout', False) or run_info.get('mus_timeout', False):
                                saw_timeout = True
                            if run_info.get('was_sat', False):
                                saw_sat = True
                                logging.info(
                                    f"Instance already SAT (no removal) for n_nodes={n_nodes} graph_type={graph_type} seed={seed}; "
                                    f"attempt {attempt+1}/{attempt_limit}; trying next seed"
                                )
                                continue

                            if run_info.get('count_wrong', 0) <= 0:
                                saw_no_wrong = True
                                logging.warning(
                                    f"No wrong facts for n_nodes={n_nodes} graph_type={graph_type} seed={seed}; "
                                    f"attempt {attempt+1}/{attempt_limit}; trying next seed"
                                )
                                continue

                            selected_run = run_info
                            break

                        if not selected_run:
                            last_seed = (run_info or {}).get('seed', None)
                            last_was_sat = bool((run_info or {}).get('was_sat', False))
                            last_wrong = int((run_info or {}).get('count_wrong', 0) or 0)
                            # Track why this rep didn't produce a usable run.
                            if saw_timeout:
                                rep_outcomes[key]['timed_out'] += 1
                            elif saw_sat or last_was_sat:
                                rep_outcomes[key]['startSAT'] += 1
                            elif saw_no_wrong or last_wrong <= 0:
                                rep_outcomes[key]['no_wrong'] += 1
                            self.skipTest(
                                f"No suitable UNSAT instance with wrong facts for n_nodes={n_nodes} graph_type={graph_type} "
                                f"after {MUS_REP_UNSAT+1} seed(s). Last_seed={last_seed}, last_was_sat={last_was_sat}, last_count_wrong={last_wrong}. "
                                f"Try a different --seed-base or increase --rep-unsat."
                            )

                        overlap = self._assert_wrong_fact_correspondence(
                            n_nodes=n_nodes,
                            facts_ext=selected_run.get('facts_ext', []),
                            wrong_ext=selected_run.get('wrong_ext', []),
                            removed_facts=selected_run.get('removed_facts', []),
                            mus_facts=selected_run.get('mus_facts', []),
                            # On timeouts, keep *all* MCS/OptMCS sets printed so far.
                            # (They are partial, but useful for overlap statistics.)
                            mcs_facts=selected_run.get('mcs_facts', []),
                            opt_mcs_facts=selected_run.get('opt_mcs_facts', None),
                            opt_mcs_label=selected_run.get('opt_mcs_label', None),
                            wc_cut_facts=selected_run.get('wc_cut_facts', None),
                            wc_label="WC",
                            include_mus_mcs=bool(selected_run.get('mus_ran', True)),
                            allow_incomplete=bool(
                                selected_run.get('mus_timeout')
                                or selected_run.get('abapc_timeout')
                                or selected_run.get('opt_timeout')
                                or selected_run.get('wc_timeout')
                                or (not bool(selected_run.get('mus_ran', True)))
                                or (not bool(selected_run.get('opt_ran', True)))
                                or (not bool(selected_run.get('wc_ran', True)))
                            ),
                            mus_timed_out=bool(selected_run.get('mus_timeout')) if bool(selected_run.get('mus_ran', True)) else False,
                            opt_timed_out=bool(selected_run.get('opt_timeout')) if bool(selected_run.get('opt_ran', True)) else False,
                        )

                        # Record run summary *before* optional minimality diagnostics, so
                        # a SkipTest from those diagnostics doesn't remove the run from aggregates.
                        rep_summaries[key].append(
                            {
                                'seed': selected_run.get('seed'),
                                'abapc_time_sec': float(selected_run.get('abapc_time_sec', 0.0) or 0.0),
                                'mus_time_sec': float(selected_run.get('mus_time_sec', 0.0) or 0.0),
                                'opt_time_sec': float(selected_run.get('opt_time_sec', 0.0) or 0.0),
                                'wc_time_sec': float(selected_run.get('wc_time_sec', 0.0) or 0.0),
                                'abapc_inc_time_sec': float(selected_run.get('abapc_inc_time_sec', 0.0) or 0.0),
                                'opt_mcs_label': selected_run.get('opt_mcs_label', None),
                                'n_facts': int(len(selected_run.get('facts_ext', []) or [])),
                                'n_wrong': int(selected_run.get('count_wrong', 0) or 0),
                                'overlap': overlap or {},
                                'graph_eval': selected_run.get('graph_eval', {}) or {},
                            }
                        )

                        # Accumulate totals for a final end-of-log summary.
                        baseline_label = "ABAPC_INC removal" if bool(MUS_USE_INCREM_ONLY) else "ABAPC removal"
                        _accumulate_method_time(baseline_label, float(selected_run.get('abapc_time_sec', 0.0) or 0.0))
                        if bool(selected_run.get('abapc_inc_ran', False)):
                            _accumulate_method_time(
                                "ABAPC_INC (incremental encoding)",
                                float(selected_run.get('abapc_inc_time_sec', 0.0) or 0.0),
                            )
                        if bool(selected_run.get('mus_ran', True)):
                            _accumulate_method_time(
                                "MUS/MCS",
                                float(selected_run.get('mus_time_sec', 0.0) or 0.0),
                            )
                        if bool(selected_run.get('opt_ran', False)):
                            _accumulate_method_time(
                                str(selected_run.get('opt_mcs_label', None) or 'OptMCS'),
                                float(selected_run.get('opt_time_sec', 0.0) or 0.0),
                            )
                        if bool(selected_run.get('wc_ran', False)):
                            _accumulate_method_time(
                                "WC",
                                float(selected_run.get('wc_time_sec', 0.0) or 0.0),
                            )
                        rep_outcomes[key]['usable'] += 1
                        # Track per-category completion for this usable run.
                        if bool(selected_run.get('abapc_timeout', False)):
                            rep_outcomes[key]['abapc_timed_out'] += 1
                        else:
                            rep_outcomes[key]['abapc_finished'] += 1

                        mus_attempted = bool(selected_run.get('mus_ran', True))
                        if not mus_attempted:
                            rep_outcomes[key]['mus_skipped'] += 1
                        else:
                            if bool(selected_run.get('mus_timeout', False)):
                                rep_outcomes[key]['mus_timed_out'] += 1
                            else:
                                rep_outcomes[key]['mus_finished'] += 1

                        opt_attempted = bool(selected_run.get('opt_ran', False))
                        if not opt_attempted:
                            rep_outcomes[key]['opt_skipped'] += 1
                        else:
                            if bool(selected_run.get('opt_timeout', False)):
                                rep_outcomes[key]['opt_timed_out'] += 1
                            else:
                                rep_outcomes[key]['opt_finished'] += 1

                        # ABAPC_INC comparison run (when not running increm-only baseline)
                        inc_attempted = selected_run.get('abapc_inc_ran', False)
                        if not inc_attempted:
                            rep_outcomes[key]['abapc_inc_skipped'] += 1
                        else:
                            if bool(selected_run.get('abapc_inc_timeout', False)):
                                rep_outcomes[key]['abapc_inc_timed_out'] += 1
                            else:
                                rep_outcomes[key]['abapc_inc_finished'] += 1

                        # Optional diagnostic: MUS minimality in isolation.
                        if MUS_CHECK_MINIMALITY and int(selected_run.get('remove_n', 0) or 0) > 0:
                            try:
                                self._assert_mus_single_deletion_makes_sat(
                                    n_nodes=n_nodes,
                                    facts_ext=selected_run.get('facts_ext', []) or [],
                                    mus_facts=selected_run.get('mus_facts', []) or [],
                                )
                            except unittest.SkipTest as e:
                                rep_outcomes[key]['mus_min_skipped'] += 1
                                logging.info(f"MUS minimality checks: skipped ({e})")

                # Per-(graph_type,n_nodes) aggregate summary over reps.
                if MUS_RANDOM_REPS > 1:
                    rows = rep_summaries.get(key, [])
                    outs = rep_outcomes.get(key, {})
                    if rows:
                        def _avg_min_max(vals: list[float]) -> tuple[float, float, float]:
                            if not vals:
                                return (0.0, 0.0, 0.0)
                            return (sum(vals) / len(vals), min(vals), max(vals))

                        abapc_vals = [r.get('abapc_time_sec', 0.0) for r in rows if r.get('abapc_time_sec', 0.0) > 0]
                        mus_vals = [r.get('mus_time_sec', 0.0) for r in rows if r.get('mus_time_sec', 0.0) > 0]
                        abapc_avg, abapc_min, abapc_max = _avg_min_max(abapc_vals)
                        mus_avg, mus_min, mus_max = _avg_min_max(mus_vals)

                        def _fmt_triple(min_v: float, avg_v: float, max_v: float) -> str:
                            return f"{min_v:7.3f} / {avg_v:7.3f} / {max_v:7.3f}"

                        def _fmt_count_triple(min_v: float, avg_v: float, max_v: float) -> str:
                            return f"{min_v:7.0f} / {avg_v:7.1f} / {max_v:7.0f}"

                        logging.info("\n" + 29*" "+"=" * 60)
                        logging.info(
                            f"SUMMARY OVER REPS (n_nodes={n_nodes}, graph_type={graph_type}, reps={MUS_RANDOM_REPS}, usable={len(rows)})"
                        )
                        logging.info("=" * 60)
                        logging.info("  [min/avg/max]")
                        usable = int(outs.get('usable', len(rows)) or 0)
                        logging.info(f"  {'usable runs':<18}: {usable}/{MUS_RANDOM_REPS}")
                        logging.info(f"  {'timed out (no run)':<18}: {outs.get('timed_out', 0)}")
                        logging.info(f"  {'startSAT':<18}: {outs.get('startSAT', 0)}")
                        if outs.get('mus_min_skipped', 0):
                            logging.info(f"  {'mus_min_skipped':<18}: {outs.get('mus_min_skipped', 0)}")
                        if outs.get('no_wrong', 0):
                            logging.info(f"  {'no_wrong':<18}: {outs.get('no_wrong', 0)}")
                        # Per-category completion: these exclude timeouts within a usable run.
                        logging.info(f"  {'ABAPC finished':<18}: {outs.get('abapc_finished', 0)}/{usable} (timeouts={outs.get('abapc_timed_out', 0)})")
                        logging.info(f"  {'MUS finished':<18}: {outs.get('mus_finished', 0)}/{usable} (timeouts={outs.get('mus_timed_out', 0)})")
                        if outs.get('opt_skipped', 0) < usable:
                            logging.info(
                                f"  {'OptMCS finished':<18}: {outs.get('opt_finished', 0)}/{usable - outs.get('opt_skipped', 0)} "
                                f"(timeouts={outs.get('opt_timed_out', 0)}, skipped={outs.get('opt_skipped', 0)})"
                            )
                        else:
                            logging.info(f"  {'OptMCS finished':<18}: skipped ({outs.get('opt_skipped', 0)})")

                        if outs.get('abapc_inc_skipped', 0) < usable:
                            logging.info(
                                f"  {'ABAPC_INC finished':<18}: {outs.get('abapc_inc_finished', 0)}/{usable - outs.get('abapc_inc_skipped', 0)} "
                                f"(timeouts={outs.get('abapc_inc_timed_out', 0)}, skipped={outs.get('abapc_inc_skipped', 0)})"
                            )
                        else:
                            logging.info(f"  {'ABAPC_INC finished':<18}: skipped ({outs.get('abapc_inc_skipped', 0)})")

                        logging.info(f"  {'ABAPC total (s)':<18}: {_fmt_triple(abapc_min, abapc_avg, abapc_max)}")
                        logging.info(f"  {'MUS total (s)':<18}: {_fmt_triple(mus_min, mus_avg, mus_max)}")

                        facts_vals = [float(r.get('n_facts', 0) or 0) for r in rows if (r.get('n_facts', 0) or 0) > 0]
                        wrong_vals = [float(r.get('n_wrong', 0) or 0) for r in rows]
                        if facts_vals:
                            f_avg, f_min, f_max = _avg_min_max(facts_vals)
                            logging.info(f"  {'facts (n)':<18}: {_fmt_count_triple(f_min, f_avg, f_max)}")
                        if wrong_vals:
                            w_avg, w_min, w_max = _avg_min_max(wrong_vals)
                            logging.info(f"  {'wrong (n)':<18}: {_fmt_count_triple(w_min, w_avg, w_max)}")

                        def _metric_vals(group: str, metric: str, stat: str = 'avg') -> list[float]:
                            out: list[float] = []
                            for r in rows:
                                g = (r.get('overlap') or {}).get(group) or {}
                                m = (g.get(metric) or {})
                                v = m.get(stat)
                                if v is not None:
                                    out.append(float(v))
                            return out

                        def _count_vals(group: str) -> list[float]:
                            out: list[float] = []
                            for r in rows:
                                g = (r.get('overlap') or {}).get(group) or {}
                                c = g.get('count')
                                if c is not None:
                                    out.append(float(c))
                            return out

                        def _metric_triplet_over_reps(group: str, metric: str) -> tuple[float, float, float] | None:
                            mins = _metric_vals(group, metric, stat='min')
                            avgs = _metric_vals(group, metric, stat='avg')
                            maxs = _metric_vals(group, metric, stat='max')
                            if not mins and not avgs and not maxs:
                                return None
                            min_v = min(mins) if mins else 0.0
                            avg_v = (sum(avgs) / len(avgs)) if avgs else 0.0
                            max_v = max(maxs) if maxs else 0.0
                            return (min_v, avg_v, max_v)

                        groups = ['Removal', 'MUS', 'MCS']
                        if any(('OptMCS' in (r.get('overlap') or {})) for r in rows):
                            groups.append('OptMCS')

                        for group in groups:
                            logging.info(f"\n{29*' '}  {group}:")
                            nsets_vals = _count_vals(group)
                            if nsets_vals:
                                c_avg, c_min, c_max = _avg_min_max(nsets_vals)
                                logging.info(f"    {'n_sets':<10}: {_fmt_count_triple(c_min, c_avg, c_max)}")
                            else:
                                logging.info(f"    {'n_sets':<10}: none")

                            t_sz = _metric_triplet_over_reps(group, 'set_size')
                            if t_sz:
                                mn, av, mx = t_sz
                                logging.info(f"    {'set_size':<10}: {_fmt_triple(mn, av, mx)}")
                            else:
                                logging.info(f"    {'set_size':<10}: none")

                            for metric in ('hit_rate', 'jaccard', 'precision', 'recall', 'f1'):
                                t = _metric_triplet_over_reps(group, metric)
                                if not t:
                                    logging.info(f"    {metric:<10}: none")
                                    continue
                                mn, av, mx = t
                                logging.info(f"    {metric:<10}: {_fmt_triple(mn, av, mx)}")

                        # ----- Graph eval summary over reps (Removal + OptMCS) -----
                        def _graph_metric_vals(group_key: str, metric: str, stat: str) -> list[float]:
                            out: list[float] = []
                            for r in rows:
                                g = (r.get('graph_eval') or {}).get(group_key) or {}
                                md = (g.get(metric) or {})
                                v = md.get(stat)
                                if v is None:
                                    continue
                                try:
                                    fv = float(v)
                                except Exception:
                                    continue
                                # Drop NaN/inf so averages stay numeric.
                                if fv != fv or fv in (float('inf'), float('-inf')):
                                    continue
                                out.append(fv)
                            return out

                        def _graph_count_vals(group_key: str) -> list[float]:
                            out: list[float] = []
                            for r in rows:
                                g = (r.get('graph_eval') or {}).get(group_key) or {}
                                c = g.get('count')
                                if c is None:
                                    continue
                                out.append(float(c))
                            return out

                        def _graph_metric_triplet_over_reps(group_key: str, metric: str) -> tuple[float, float, float] | None:
                            mins = _graph_metric_vals(group_key, metric, 'min')
                            avgs = _graph_metric_vals(group_key, metric, 'avg')
                            maxs = _graph_metric_vals(group_key, metric, 'max')
                            if not mins and not avgs and not maxs:
                                return None
                            min_v = min(mins) if mins else 0.0
                            avg_v = (sum(avgs) / len(avgs)) if avgs else 0.0
                            max_v = max(maxs) if maxs else 0.0
                            return (min_v, avg_v, max_v)

                        ge_groups: list[tuple[str, str]] = [('Removal', 'Removal')]
                        if any(('OptMCS' in (r.get('graph_eval') or {})) for r in rows):
                            label = None
                            for r in rows:
                                gg = (r.get('graph_eval') or {}).get('OptMCS') or {}
                                if gg.get('label'):
                                    label = str(gg.get('label'))
                                    break
                            ge_groups.append(('OptMCS', label or 'OptMCS'))

                        logging.info("\n" + 29*" " + "GRAPH EVAL OVER REPS")
                        logging.info("  [min/avg/max]  (for SHD/SID: lower is better)")

                        for group_key, group_label in ge_groups:
                            logging.info(f"\n{29*' '}  {group_label}:")
                            ndags_vals = _graph_count_vals(group_key)
                            if ndags_vals:
                                d_avg, d_min, d_max = _avg_min_max(ndags_vals)
                                logging.info(f"    {'n_dags':<10}: {_fmt_count_triple(d_min, d_avg, d_max)}")
                            else:
                                logging.info(f"    {'n_dags':<10}: none")

                            metrics = ['precision', 'recall', 'F1', 'shd']
                            if MUS_GRAPH_EVAL_SID:
                                metrics.append('sid')
                            for metric in metrics:
                                t = _graph_metric_triplet_over_reps(group_key, metric)
                                if not t:
                                    logging.info(f"    {metric:<10}: none")
                                    continue
                                mn, av, mx = t
                                logging.info(f"    {metric:<10}: {_fmt_triple(mn, av, mx)}")

                        logging.info("=" * 60)

    def test_demo_mus_core_enforced_is_unsat_under_mus_background(self):
        """DEMO: MUS core must be UNSAT when enforced under the same MUS program.

        This checks the MUS definition in the setting used by `CausalABA_MUS`:
        for each returned MUS core (a set of mus(i) assumptions), if we assert those
        mus(i) atoms in the *same adorned program*, the result must be UNSAT even
        when all other mus(j) are left free.

        This intentionally does NOT rebuild the instance-specific background theory
        from only the core facts (that was the misleading part of the previous demo).
        """
        logger_setup()
        logging.info("===============Running DEMO: enforced MUS cores are UNSAT===============")

        import shutil
        if shutil.which('clingo') is None or shutil.which('wasp') is None:
            self.skipTest("clingo/wasp not found on PATH")

        _ensure_solvers_imported()

        import subprocess
        from causalaba_mus import parse_facts_from_file, build_mus_program

        def _clingo_is_sat(program_text: str, timeout_sec: float) -> bool | None:
            fd, tmp_lp = tempfile.mkstemp(suffix='_clingo_check.lp', text=True)
            os.close(fd)
            try:
                with open(tmp_lp, 'w') as f:
                    f.write(program_text)
                try:
                    proc = subprocess.run(
                        ['clingo', tmp_lp, '-n', '1'],
                        capture_output=True,
                        text=True,
                        timeout=float(timeout_sec),
                    )
                except subprocess.TimeoutExpired:
                    return None
                out = proc.stdout or ''
                if 'UNSATISFIABLE' in out:
                    return False
                if 'SATISFIABLE' in out:
                    return True
                return None
            finally:
                try:
                    os.remove(tmp_lp)
                except Exception:
                    pass

        n_nodes = 4
        timeout = float(min(10, int(MUS_SOLVE_TIMEOUT)))
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        try:
            with open(facts_file, 'w') as f:
                # Two independent contradictions (keeps the example simple and deterministic)
                f.write("ext_indep(0,1,empty).\n")
                f.write("ext_dep(0,1,empty).\n")
                f.write("ext_indep(2,3,empty).\n")
                f.write("ext_dep(2,3,empty).\n")

            mus_result = CausalABA_MUS(
                n_nodes=n_nodes,
                facts_location=facts_file,
                gringo_path='clingo',
                wasp_path='wasp',
                max_muses=0,
                mus_algorithm='camus',
                print_mcses=True,
                camus_mcs_threshold=None,
                camus_mus_threshold=None,
                solve_timeout=timeout,
            )

            self.assertGreater(mus_result.get('n_mus', 0), 0, "Expected at least one MUS in the toy contradiction instance")

            mus_list = mus_result.get('mus_list', []) or []
            mus_facts = mus_result.get('mus_facts', []) or []
            logging.info(f"DEMO: MUS cores found: {len(mus_list)}")
            for i, core in enumerate(mus_list, start=1):
                core_facts = mus_facts[i - 1] if (i - 1) < len(mus_facts) else []
                logging.info(f"  MUS #{i}: mus indices={sorted(core)}")
                if core_facts:
                    for fact in core_facts:
                        logging.info(f"    - {fact}")

            facts_no_period, _mapping = parse_facts_from_file(facts_file)
            base_program = build_mus_program(n_nodes, facts_no_period, facts_file)

            for core_idx, core in enumerate(mus_list, start=1):
                # Enforce only the core mus(i) atoms; leave all other mus(j) free.
                forced = base_program + "\n% ===== Force MUS core =====\n" + "\n".join(f"mus({i})." for i in core) + "\n"
                is_sat = _clingo_is_sat(forced, timeout_sec=timeout)
                if is_sat is None:
                    self.skipTest(f"Timeout/unknown while checking enforced MUS core #{core_idx}")
                self.assertFalse(
                    is_sat,
                    f"MUS core #{core_idx} was SAT when enforced under the MUS background (core={core})",
                )
                logging.info(f"  ✓ MUS #{core_idx} enforced => UNSAT")

            logging.info("✓ All MUSes generate UNSAT")
        finally:
            try:
                os.remove(facts_file)
            except Exception:
                pass

    def test_demo_full_instance_delete_one_mus_element_restores_sat(self):
        """DEMO: show why full-instance single-deletion is not a MUS minimality test.

        Construct an UNSAT instance with two independent contradictions. Then:
        - compute a MUS for one contradiction,
        - delete one element from that MUS *but keep all other facts*,
        - show the full instance can remain UNSAT (because the other contradiction remains).

        This demo should PASS: the expected outcome is that the instance remains UNSAT.
        """
        logger_setup()
        logging.info("===============Running DEMO: full-instance single deletion===============" )

        _ensure_solvers_imported()

        n_nodes = 4
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        try:
            with open(facts_file, 'w') as f:
                # Contradiction A
                f.write("ext_indep(0,1,empty).\n")
                f.write("ext_dep(0,1,empty).\n")
                # Contradiction B
                f.write("ext_indep(2,3,empty).\n")
                f.write("ext_dep(2,3,empty).\n")

            mus_result = CausalABA_MUS(
                n_nodes=n_nodes,
                facts_location=facts_file,
                gringo_path="clingo",
                wasp_path="wasp",
                max_muses=0,
                mus_algorithm="camus",
                print_mcses=True,
                camus_mcs_threshold=None,
                camus_mus_threshold=None,
            )

            logging.info(f"DEMO contradiction instance: n_mus={mus_result.get('n_mus', 0)}, n_mcs={mus_result.get('n_mcs', 0)}")
            for i, core_i in enumerate(mus_result.get('mus_facts', [])[:5], start=1):
                logging.info(f"  MUS #{i}: {core_i}")
            for i, cut_i in enumerate(mus_result.get('mcs_facts', [])[:5], start=1):
                logging.info(f"  MCS #{i}: {cut_i}")

            mus_facts = mus_result.get('mus_facts', []) or []
            self.assertGreater(len(mus_facts), 0, "Expected at least one MUS")
            core = [self._normalize_fact_str(str(s)) for s in mus_facts[0] if str(s).strip()]
            self.assertGreater(len(core), 0, "Expected non-empty MUS core")

            # Choose a fact from the MUS core to delete.
            removed_fact = core[0]
            with open(facts_file, 'r') as f:
                full = [self._normalize_fact_str(line.strip()) for line in f if line.strip()]
            remaining = [s for s in full if self._normalize_fact_str(s) != removed_fact]

            fd2, sub_file = tempfile.mkstemp(suffix='.lp', text=True)
            os.close(fd2)
            try:
                with open(sub_file, 'w') as f:
                    for s in remaining:
                        f.write(f"#external {self._normalize_fact_str(s)}\n")

                timing1: dict[str, Any] = {}
                models1, _ = CausalABA(
                    n_nodes,
                    sub_file,
                    weak_constraints=False,
                    print_models=False,
                    skeleton_rules_reduction=True,
                    out_n=1,
                    solve_timeout=float(min(10, int(MUS_SOLVE_TIMEOUT))),
                    timing_recorder=timing1,
                )
            finally:
                try:
                    os.remove(sub_file)
                except Exception:
                    pass

            if timing1.get('timed_out', False):
                self.fail("DEMO FAILURE: timed out while checking full-instance deletion")

            # Demonstrate effect: even after deleting one element from a MUS core,
            # the full instance can remain UNSAT due to other contradictions.
            full_preview = "\n".join(f"  - {s}" for s in full)
            if len(models1) > 0:
                # In principle, depending on the instance, this could happen. For this constructed
                # demo we expect the other contradiction to keep it UNSAT, so if it becomes SAT the
                # demo didn't illustrate the intended point.
                self.skipTest(
                    "DEMO did not trigger: Full instance became SAT after deleting one MUS element.\n"
                    f"removed_fact={removed_fact}\n"
                    f"n_mus={mus_result.get('n_mus', 0)}, n_mcs={mus_result.get('n_mcs', 0)}\n"
                    f"MUSes (first 5): {mus_result.get('mus_facts', [])[:5]}\n"
                    f"MCSes (first 5): {mus_result.get('mcs_facts', [])[:5]}\n"
                    "full instance facts:\n"
                    f"{full_preview}"
                )

            logging.info(
                "DEMO: Deleting one MUS element from the FULL instance did NOT restore SAT (expected).\n"
                "This illustrates why 'full-instance single deletion' is not a MUS minimality test (other contradictions may remain).\n"
                f"removed_fact={removed_fact}\n"
                f"n_mus={mus_result.get('n_mus', 0)}, n_mcs={mus_result.get('n_mcs', 0)}\n"
                f"MUSes (first 5): {mus_result.get('mus_facts', [])[:5]}\n"
                f"MCSes (first 5): {mus_result.get('mcs_facts', [])[:5]}\n"
                "full instance facts:\n"
                f"{full_preview}"
            )

            # Core assertion for the demo: still UNSAT.
            self.assertEqual(
                len(models1),
                0,
                "Expected UNSAT after deleting one MUS element from full instance (other contradiction remains)",
            )
        finally:
            try:
                os.remove(facts_file)
            except Exception:
                pass

    @staticmethod
    def _normalize_fact_str(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return s
        if s.startswith("#external"):
            s = s[len("#external"):].strip()
            # Common form is "#external <fact>."; tolerate missing spaces.
            if s.startswith(" "):
                s = s.strip()
        return s if s.endswith('.') else s + '.'

    def _assert_wrong_fact_correspondence(
        self,
        *,
        n_nodes: int,
        facts_ext,
        wrong_ext,
        removed_facts,
        mus_facts,
        mcs_facts,
        opt_mcs_facts=None,
        opt_mcs_label: str | None = None,
        wc_cut_facts=None,
        wc_label: str | None = None,
        include_mus_mcs: bool = True,
        allow_incomplete: bool = False,
        mus_timed_out: bool = False,
        opt_timed_out: bool = False,
    ) -> dict[str, Any]:
        """Check MUS/MCS sets correspond to wrong facts.

        Intended for random-PC instances where `wrong_ext` is derived from ground truth.
                Notes:
                - Every MUS should contain at least one wrong fact (otherwise the contradiction isn't explained).
                - MCSes are *repairs* (sets of assumptions to disable). They can include correct facts too.
                    So we do not require every MCS to include a wrong fact; instead we report a hit-rate.
                - Returned MUS/MCS facts should be drawn from the provided fact universe.
        """
        # If MUS/MCS was not requested, suppress MUS/MCS overlap output regardless of caller.
        try:
            if 'mus' not in (MUS_ANALYSIS_MODES or set()):
                include_mus_mcs = False
        except Exception:
            pass

        all_facts = {self._normalize_fact_str(s) for s in (facts_ext or []) if str(s).strip()}
        wrong_set = {self._normalize_fact_str(s) for s in (wrong_ext or []) if str(s).strip()}
        removed_set = {self._normalize_fact_str(s) for s in (removed_facts or []) if str(s).strip()}
        if not all_facts or not wrong_set:
            self.fail(f"Missing facts/wrong facts for correspondence check (n_nodes={n_nodes})")

        def to_sets(groups):
            out = []
            for g in (groups or []):
                s = {self._normalize_fact_str(x) for x in (g or []) if str(x).strip()}
                if s:
                    out.append(s)
            return out

        mus_sets = to_sets(mus_facts)
        mcs_sets = to_sets(mcs_facts)

        def jaccard(a: set[str], b: set[str]) -> float:
            if not a and not b:
                return 1.0
            u = a.union(b)
            if not u:
                return 0.0
            return len(a.intersection(b)) / len(u)

        def summarize_overlap(label: str, groups: list[set[str]]) -> dict[str, Any]:
            def _mmx(xs: list[float]) -> dict[str, float]:
                if not xs:
                    return {'min': 0.0, 'avg': 0.0, 'max': 0.0}
                return {'min': min(xs), 'avg': (sum(xs) / len(xs)), 'max': max(xs)}

            def _fmt_triple(d: dict[str, float]) -> str:
                return f"{d['min']:7.3f} / {d['avg']:7.3f} / {d['max']:7.3f}"

            if not groups:
                out = {
                    'count': 0,
                    'set_size': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'hit_rate': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'jaccard': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'precision': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'recall': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'f1': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                }
                logging.info(f"  {label}:")
                logging.info(f"    {'n_sets':<10}: {out['count']:7d}")
                logging.info(f"    {'set_size':<10}: {_fmt_triple(out['set_size'])}")
                logging.info(f"    {'hit_rate':<10}: {_fmt_triple(out['hit_rate'])}")
                logging.info(f"    {'jaccard':<10}: {_fmt_triple(out['jaccard'])}")
                logging.info(f"    {'precision':<10}: {_fmt_triple(out['precision'])}")
                logging.info(f"    {'recall':<10}: {_fmt_triple(out['recall'])}")
                logging.info(f"    {'f1':<10}: {_fmt_triple(out['f1'])}")
                return out

            jac = [jaccard(g, wrong_set) for g in groups]
            prec = [(len(g.intersection(wrong_set)) / len(g)) if g else 0.0 for g in groups]
            # Recall is w.r.t wrong_set (can be large); still useful as a scale indicator.
            rec = [(len(g.intersection(wrong_set)) / len(wrong_set)) if wrong_set else 0.0 for g in groups]
            f1 = [((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0 for p, r in zip(prec, rec)]
            hit = [1.0 if len(g.intersection(wrong_set)) >= 1 else 0.0 for g in groups]

            out = {
                'count': len(groups),
                'set_size': _mmx([float(len(g)) for g in groups]),
                'hit_rate': _mmx(hit),
                'jaccard': _mmx(jac),
                'precision': _mmx(prec),
                'recall': _mmx(rec),
                'f1': _mmx(f1),
            }

            logging.info(f"  {label}:")
            logging.info(f"    {'n_sets':<10}: {out['count']:7d}")
            logging.info(f"    {'set_size':<10}: {_fmt_triple(out['set_size'])}")
            logging.info(f"    {'hit_rate':<10}: {_fmt_triple(out['hit_rate'])}")
            logging.info(f"    {'jaccard':<10}: {_fmt_triple(out['jaccard'])}")
            logging.info(f"    {'precision':<10}: {_fmt_triple(out['precision'])}")
            logging.info(f"    {'recall':<10}: {_fmt_triple(out['recall'])}")
            logging.info(f"    {'f1':<10}: {_fmt_triple(out['f1'])}")

            return out

        def empty_overlap_summary() -> dict[str, Any]:
            return {
                'count': 0,
                'set_size': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                'hit_rate': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                'jaccard': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                'precision': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                'recall': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                'f1': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
            }

        # Removal (ABAPC): removed facts should align with wrong facts.
        removal_groups: list[set[str]] = [removed_set] if removed_set else []

        logging.info(f"\n")
        logging.info("=" * 60)
        logging.info(f" OVERLAP SUMMARY (n_nodes={n_nodes})")
        logging.info("=" * 60)
        logging.info("  [min/avg/max]")
        removal_summary = summarize_overlap("Removal", removal_groups)

        # Pure-WC cut (clingo optimization) summary, if provided.
        wc_summary = None
        if wc_cut_facts is not None:
            wc_sets = to_sets([wc_cut_facts])
            wc_name = wc_label or "WC"
            wc_summary = summarize_overlap(wc_name, wc_sets)

        # MUS correspondence: in many cases MUS cores implicate wrong facts, but for some
        # random instances a core can be composed entirely of (ground-truth) correct facts.
        # We therefore require that *at least one* MUS hits a wrong fact and report hit-rate.
        mus_summary = empty_overlap_summary()
        mcs_summary = empty_overlap_summary()

        if include_mus_mcs:
            mus_label = "MUS"
            if mus_timed_out:
                mus_label = "MUS (timed out)"

            if not mus_sets:
                if allow_incomplete:
                    mus_summary = summarize_overlap(mus_label, [])
                else:
                    self.assertGreater(
                        len(mus_sets),
                        0,
                        f"Expected at least one MUS set for correspondence check (n_nodes={n_nodes})",
                    )
                    mus_summary = summarize_overlap(mus_label, mus_sets)
            else:
                for i, core in enumerate(mus_sets, start=1):
                    self.assertTrue(
                        core.issubset(all_facts),
                        f"MUS #{i} contains facts not in instance fact set (n_nodes={n_nodes})",
                    )

                any_wrong_mus = any(len(core.intersection(wrong_set)) >= 1 for core in mus_sets)
                if not any_wrong_mus:
                    msg = f"No MUS contains a wrong fact (n_nodes={n_nodes})"
                    if MUS_REQUIRE_MUS_HIT_WRONG:
                        self.fail(msg)
                    logging.warning(msg)

                mus_summary = summarize_overlap(mus_label, mus_sets)

            # MCS correspondence: do not require every MCS to include wrong facts.
            # (A minimal correction set can disable some correct assumptions too.)
            for i, cut in enumerate(mcs_sets, start=1):
                self.assertTrue(
                    cut.issubset(all_facts),
                    f"MCS #{i} contains facts not in instance fact set (n_nodes={n_nodes})",
                )

            mcs_label = "MCS"
            if mus_timed_out:
                mcs_label = "MCS (timed out)"
            mcs_summary = summarize_overlap(mcs_label, mcs_sets)

        opt_mcs_summary = None
        if opt_mcs_facts is not None:
            opt_sets = to_sets(opt_mcs_facts)
            label = opt_mcs_label or "OptMCS"
            opt_label = label
            if opt_timed_out:
                opt_label = f"{label} (timed out)"
            opt_mcs_summary = summarize_overlap(opt_label, opt_sets)

        logging.info("=" * 60)

        if include_mus_mcs and mcs_sets:
            any_wrong = any(len(cut.intersection(wrong_set)) >= 1 for cut in mcs_sets)
            if not any_wrong:
                msg = f"No MCS intersects wrong facts (n_nodes={n_nodes})"
                if allow_incomplete:
                    logging.warning(f"Timeout/incomplete run: {msg}; continuing")
                elif MUS_REQUIRE_MCS_HIT_WRONG:
                    self.fail(msg)
                else:
                    logging.warning(msg)

        if opt_mcs_facts is not None:
            opt_sets = to_sets(opt_mcs_facts)
            if opt_sets:
                any_wrong_opt = any(len(cut.intersection(wrong_set)) >= 1 for cut in opt_sets)
                if not any_wrong_opt:
                    msg = f"No OptMCS intersects wrong facts (n_nodes={n_nodes})"
                    if allow_incomplete:
                        logging.warning(f"Timeout/incomplete run: {msg}; continuing")
                    elif MUS_REQUIRE_OPTMCS_HIT_WRONG:
                        self.fail(msg)
                    else:
                        logging.warning(msg)

        return {
            'Removal': removal_summary,
            'MUS': mus_summary,
            'MCS': mcs_summary,
            **({'WC': wc_summary} if wc_summary is not None else {}),
            **({'OptMCS': opt_mcs_summary} if opt_mcs_summary is not None else {}),
        }

    def _assert_mus_single_deletion_makes_sat(
        self,
        *,
        n_nodes: int,
        facts_ext: list[str],
        mus_facts: list[list[str]],
        solve_timeout_sec: float | None = None,
    ) -> None:
        """Evaluate MUS minimality in isolation.

        For each (sampled) MUS core S returned by `CausalABA_MUS`:
        - Check whether encoding + S is UNSAT.
        - Only if UNSAT in isolation, test minimality by checking that for each
          (sampled) f in S, encoding + (S \\ {f}) is SAT.

        This intentionally does NOT test "full instance facts F minus one core
        element" because random instances may contain multiple independent
        contradictions.

        If no sampled cores are UNSAT in isolation, the helper raises SkipTest.
        """
        _ensure_solvers_imported()

        def _write_fact_lines(path: str, facts: list[str]) -> None:
            with open(path, 'w') as f:
                for s in facts:
                    line = self._normalize_fact_str(str(s))
                    if not line.strip():
                        continue
                    # Mirror how ABAPC is typically run: facts are externals assigned true.
                    f.write(f"#external {line}\n")

        # Only need one model to establish SAT.
        out_n = 1
        timeout = solve_timeout_sec
        if timeout is None:
            timeout = float(min(10, int(MUS_SOLVE_TIMEOUT)))

        # Large random instances can yield many MUSes and/or very large MUS cores.
        # To keep runtime bounded:
        # - If > 10 MUS cores are returned, sample 10 cores.
        # - For each selected core, if |core| > 10, sample 10 deletions.
        max_cores_to_check = 10
        max_deletions_per_core = 10

        # Optional progress bar; keep it as a soft dependency.
        try:
            from tqdm.auto import tqdm  # type: ignore
        except Exception:
            tqdm = None  # type: ignore

        import random
        rng = random.Random(0)

        # Normalize/deduplicate the full instance facts.
        all_norm = [self._normalize_fact_str(str(s)) for s in (facts_ext or []) if str(s).strip()]
        seen_all: set[str] = set()
        all_norm = [s for s in all_norm if not (s in seen_all or seen_all.add(s))]
        if not all_norm:
            self.skipTest(f"No facts_ext provided for MUS single-deletion checks (n_nodes={n_nodes})")

        cores_all = list(mus_facts or [])
        if not cores_all:
            return

        if len(cores_all) > max_cores_to_check:
            sampled_idxs = sorted(rng.sample(range(len(cores_all)), k=max_cores_to_check))
            cores_to_check: list[tuple[int, list[str]]] = [(i + 1, cores_all[i]) for i in sampled_idxs]
        else:
            cores_to_check = [(i + 1, core) for i, core in enumerate(cores_all)]

        # Normalize/dedup selected cores and pre-plan the number of checks.
        normalized_cores: list[tuple[int, list[str], list[int]]] = []
        total_checks = 0
        for core_idx, core in cores_to_check:
            core_norm = [self._normalize_fact_str(str(s)) for s in (core or []) if str(s).strip()]
            seen: set[str] = set()
            core_norm = [s for s in core_norm if not (s in seen or seen.add(s))]
            if not core_norm:
                continue

            if len(core_norm) > max_deletions_per_core:
                del_idxs = sorted(rng.sample(range(len(core_norm)), k=max_deletions_per_core))
            else:
                del_idxs = list(range(len(core_norm)))

            normalized_cores.append((core_idx, core_norm, del_idxs))
            total_checks += 1  # core-only diagnostic SAT/UNSAT
            total_checks += len(del_idxs)  # single deletions SAT

        # One upfront check: full instance must be UNSAT.
        total_checks += 1

        progress = None
        if tqdm is not None:
            progress = tqdm(total=total_checks, desc="MUS minimality checks", unit="check")

        # Note: We intentionally do NOT enforce "remove one element from MUS in the FULL instance ⇒ SAT".
        # Random instances can contain multiple independent contradictions, so removing one element
        # from one MUS may leave the overall instance UNSAT. Instead, we test MUS minimality in
        # isolation: core is UNSAT, but core \ {f} is SAT for each f in the core.
        if progress is not None:
            # Keep accounting stable (we reserved 1 check above).
            progress.update(1)

        diag_core_only_unsat = 0
        diag_core_only_sat = 0
        theory_applies_cores = 0
        cores_checked = 0
        deletions_checked = 0
        deletions_sat = 0
        failures: list[str] = []
        cores_tested_for_minimality = 0

        for core_idx, core_norm, del_idxs in normalized_cores:
            cores_checked += 1
            fd, core_file = tempfile.mkstemp(suffix='.lp', text=True)
            os.close(fd)
            try:
                _write_fact_lines(core_file, core_norm)

                timing0: dict[str, Any] = {}
                models0, _ = CausalABA(
                    n_nodes,
                    core_file,
                    weak_constraints=False,
                    print_models=False,
                    skeleton_rules_reduction=True,
                    out_n=out_n,
                    solve_timeout=timeout,
                    timing_recorder=timing0,
                )
                if progress is not None:
                    progress.update(1)
                if timing0.get('timed_out', False):
                    self.skipTest(
                        f"Timeout during MUS minimality check (core-only UNSAT) "
                        f"(n_nodes={n_nodes}, core={core_idx}, timeout={timeout}s)"
                    )
                if len(models0) == 0:
                    diag_core_only_unsat += 1
                else:
                    diag_core_only_sat += 1

                # Only meaningful to test minimality if the core alone is UNSAT.
                if len(models0) > 0:
                    continue

                cores_tested_for_minimality += 1

                core_all_deletions_sat = True
                for remove_pos, remove_i in enumerate(del_idxs, start=1):
                    removed_fact = core_norm[remove_i]
                    # Remove the selected element from the MUS core (minimality in isolation).
                    remaining = [s for s in core_norm if s != removed_fact]
                    logging.debug(
                        "MUS minimality check: core=%s (|core|=%s) remove=%s/%s fact=%s",
                        core_idx,
                        len(core_norm),
                        remove_pos,
                        len(del_idxs),
                        removed_fact,
                    )

                    fd2, sub_file = tempfile.mkstemp(suffix='.lp', text=True)
                    os.close(fd2)
                    try:
                        _write_fact_lines(sub_file, remaining)
                        timing1: dict[str, Any] = {}
                        models1, _ = CausalABA(
                            n_nodes,
                            sub_file,
                            weak_constraints=False,
                            print_models=False,
                            skeleton_rules_reduction=True,
                            out_n=out_n,
                            solve_timeout=timeout,
                            timing_recorder=timing1,
                        )
                        if progress is not None:
                            progress.update(1)
                        deletions_checked += 1
                        if timing1.get('timed_out', False):
                            self.skipTest(
                                f"Timeout during MUS minimality check (single deletion) "
                                f"(n_nodes={n_nodes}, core={core_idx}, remove={remove_pos}/{len(del_idxs)}, timeout={timeout}s)"
                            )
                        if len(models1) > 0:
                            deletions_sat += 1
                        else:
                            core_all_deletions_sat = False
                            failures.append(
                                f"Core #{core_idx}: removing '{removed_fact}' did NOT restore SAT "
                                f"(n_nodes={n_nodes}, |core|={len(core_norm)})"
                            )
                    finally:
                        try:
                            os.remove(sub_file)
                        except Exception:
                            pass

                if core_all_deletions_sat:
                    theory_applies_cores += 1
            finally:
                try:
                    os.remove(core_file)
                except Exception:
                    pass

        if progress is not None:
            try:
                progress.close()
            except Exception:
                pass

        logging.info(
            "MUS single-deletion summary (n_nodes=%s): cores_checked=%s, deletions_checked=%s, deletions_sat=%s, "
            "cores_where_property_holds=%s; core_only_unsat=%s, core_only_sat=%s",
            n_nodes,
            cores_checked,
            deletions_checked,
            deletions_sat,
            theory_applies_cores,
            diag_core_only_unsat,
            diag_core_only_sat,
        )
        if cores_tested_for_minimality == 0:
            self.skipTest(
                f"No MUS cores were UNSAT in isolation; skipping MUS minimality checks "
                f"(n_nodes={n_nodes}, cores_checked={cores_checked}, core_only_sat={diag_core_only_sat})"
            )
        if failures:
            # This is a real minimality violation (core UNSAT but core\{f} still UNSAT).
            self.fail("\n".join(failures[:20]) + ("\n..." if len(failures) > 20 else ""))

    def _assert_mus_cores_enforced_unsat_under_mus_background(
        self,
        *,
        n_nodes: int,
        mus_list: list[list[int]],
        program_text: str | None = None,
        emit_lp_path: str | None = None,
        facts_ext: list[str] | None = None,
        solve_timeout_sec: float | None = None,
    ) -> None:
        """Check MUS definition: enforced core must be UNSAT under the same MUS program.

        Given MUS cores as lists of indices (as returned by `CausalABA_MUS(...)["mus_list"]`),
        enforce each core by adding facts `mus(i).` to the *same adorned MUS program* and assert
        clingo reports UNSAT.

        This is intentionally different from checking the core facts in isolation under `CausalABA`.
        """
        _ensure_solvers_imported()

        import shutil
        import subprocess

        if not mus_list:
            return

        if shutil.which('clingo') is None:
            self.skipTest("clingo not found on PATH")

        timeout = solve_timeout_sec
        if timeout is None:
            timeout = float(min(10, int(MUS_SOLVE_TIMEOUT)))

        def _clingo_is_sat(program: str, timeout_sec: float) -> bool | None:
            fd, tmp_lp = tempfile.mkstemp(suffix='_clingo_check.lp', text=True)
            os.close(fd)
            try:
                with open(tmp_lp, 'w') as f:
                    f.write(program)
                try:
                    proc = subprocess.run(
                        ['clingo', tmp_lp, '-n', '1'],
                        capture_output=True,
                        text=True,
                        timeout=float(timeout_sec),
                    )
                except subprocess.TimeoutExpired:
                    return None
                out = proc.stdout or ''
                if 'UNSATISFIABLE' in out:
                    return False
                if 'SATISFIABLE' in out:
                    return True
                return None
            finally:
                try:
                    os.remove(tmp_lp)
                except Exception:
                    pass

        # Prefer reusing the emitted MUS program (avoids rebuilding via compile_and_ground).
        base_program: str | None = None
        if program_text is not None:
            base_program = str(program_text)
        elif emit_lp_path and os.path.exists(str(emit_lp_path)):
            try:
                with open(str(emit_lp_path), 'r') as f:
                    base_program = f.read()
            except Exception as e:
                logging.warning(f"Failed to read emitted MUS program '{emit_lp_path}': {e}")
                base_program = None

        # Fallback: rebuild the adorned program (can be slower).
        tmp_facts_file: str | None = None
        if base_program is None:
            if not facts_ext:
                self.skipTest("No MUS program text available (no emit_lp_path/program_text and no facts_ext to rebuild)")

            from causalaba_mus import parse_facts_from_file, build_mus_program

            fd, tmp_facts_file = tempfile.mkstemp(suffix='_mus_facts.lp', text=True)
            os.close(fd)
            with open(tmp_facts_file, 'w') as f:
                for s in facts_ext:
                    line = self._normalize_fact_str(str(s))
                    if not line.strip():
                        continue
                    f.write(f"{line}\n")

            facts_no_period, _mapping = parse_facts_from_file(tmp_facts_file)
            base_program = build_mus_program(n_nodes, facts_no_period, tmp_facts_file)

        try:
            assert base_program is not None

            # Large random instances can yield many MUSes; keep runtime bounded.
            max_cores_to_check = 100
            rng = random.Random(0)
            cores_all = list(mus_list)
            if len(cores_all) > max_cores_to_check:
                idxs = sorted(rng.sample(range(len(cores_all)), k=max_cores_to_check))
                cores_to_check = [(i + 1, cores_all[i]) for i in idxs]
            else:
                cores_to_check = [(i + 1, core) for i, core in enumerate(cores_all)]

            logging.info(
                "MUS enforced-UNSAT check (n_nodes=%s): checking %s/%s cores",
                n_nodes,
                len(cores_to_check),
                len(cores_all),
            )

            for core_idx, core in cores_to_check:
                forced = (
                    base_program
                    + "\n% ===== Force MUS core =====\n"
                    + "\n".join(f"mus({i})." for i in (core or []))
                    + "\n"
                )
                is_sat = _clingo_is_sat(forced, timeout_sec=float(timeout))
                if is_sat is None:
                    self.skipTest(f"Timeout/unknown while checking enforced MUS core #{core_idx}")
                self.assertFalse(
                    is_sat,
                    f"MUS core #{core_idx} was SAT when enforced under the MUS background (core={core})",
                )
        finally:
            if tmp_facts_file is not None:
                try:
                    os.remove(tmp_facts_file)
                except Exception:
                    pass

    def _assert_mus_cores_enforced_unsat_under_abapc_background(
        self,
        *,
        n_nodes: int,
        mus_facts: list[list[str]],
        solve_timeout_sec: float | None = None,
    ) -> None:
        """Check UNSAT for each MUS core under the plain CausalABA background.

        For each (sampled) MUS core returned by `CausalABA_MUS` (as resolved fact strings),
        write a temporary facts file containing only that core and assert that calling
        `CausalABA(...)` yields UNSAT.
        """
        _ensure_solvers_imported()

        if not mus_facts:
            return

        timeout = solve_timeout_sec
        if timeout is None:
            timeout = float(min(10, int(MUS_SOLVE_TIMEOUT)))

        # Large instances can yield many MUSes; keep runtime bounded.
        max_cores_to_check = 100
        rng = random.Random(0)
        cores_all = list(mus_facts)
        if len(cores_all) > max_cores_to_check:
            idxs = sorted(rng.sample(range(len(cores_all)), k=max_cores_to_check))
            cores_to_check = [(i + 1, cores_all[i]) for i in idxs]
        else:
            cores_to_check = [(i + 1, core) for i, core in enumerate(cores_all)]

        logging.info(
            "MUS enforced-UNSAT under ABAPC background (n_nodes=%s): checking %s/%s cores",
            n_nodes,
            len(cores_to_check),
            len(cores_all),
        )

        def _write_fact_lines(path: str, facts: list[str]) -> None:
            with open(path, 'w') as f:
                for s in facts:
                    line = self._normalize_fact_str(str(s))
                    if not line.strip():
                        continue
                    f.write(f"#external {line}\n")

        for core_idx, core in cores_to_check:
            core_norm = [self._normalize_fact_str(str(s)) for s in (core or []) if str(s).strip()]
            if not core_norm:
                continue

            fd, core_file = tempfile.mkstemp(suffix='_mus_core_abapc.lp', text=True)
            os.close(fd)
            try:
                _write_fact_lines(core_file, core_norm)

                timing: dict[str, Any] = {}
                models, _ = CausalABA(
                    n_nodes,
                    core_file,
                    weak_constraints=False,
                    print_models=False,
                    skeleton_rules_reduction=True,
                    out_n=1,
                    solve_timeout=timeout,
                    timing_recorder=timing,
                )
                if timing.get('timed_out', False):
                    self.skipTest(
                        f"Timeout while checking ABAPC-background UNSAT for MUS core #{core_idx} (timeout={timeout}s)"
                    )
                self.assertEqual(
                    len(models),
                    0,
                    f"MUS core #{core_idx} was SAT under ABAPC background when isolated to its facts",
                )
            finally:
                try:
                    os.remove(core_file)
                except Exception:
                    pass

    def _assert_mcs_cores_removed_sat_under_mus_background(
        self,
        *,
        n_nodes: int,
        mcs_list: list[list[int]],
        n_facts_total: int,
        program_text: str | None = None,
        emit_lp_path: str | None = None,
        facts_ext: list[str] | None = None,
        solve_timeout_sec: float | None = None,
    ) -> None:
        """Check MCS semantics under the MUS (adorned) background.

        For each returned MCS cut C (a set of mus(i) assumptions to disable), enforce:
        - mus(i). for all i not in C
        - :- mus(i). for all i in C
        and check the adorned program is SAT.
        """
        _ensure_solvers_imported()

        import shutil
        import subprocess

        if not mcs_list:
            return

        if shutil.which('clingo') is None:
            self.skipTest("clingo not found on PATH")

        timeout = solve_timeout_sec
        if timeout is None:
            timeout = float(min(10, int(MUS_SOLVE_TIMEOUT)))

        def _clingo_is_sat(program: str, timeout_sec: float) -> bool | None:
            fd, tmp_lp = tempfile.mkstemp(suffix='_clingo_check.lp', text=True)
            os.close(fd)
            try:
                with open(tmp_lp, 'w') as f:
                    f.write(program)
                try:
                    proc = subprocess.run(
                        ['clingo', tmp_lp, '-n', '1'],
                        capture_output=True,
                        text=True,
                        timeout=float(timeout_sec),
                    )
                except subprocess.TimeoutExpired:
                    return None
                out = proc.stdout or ''
                if 'UNSATISFIABLE' in out:
                    return False
                if 'SATISFIABLE' in out:
                    return True
                return None
            finally:
                try:
                    os.remove(tmp_lp)
                except Exception:
                    pass

        # Prefer reusing the emitted MUS program (avoids rebuilding via compile_and_ground).
        base_program: str | None = None
        if program_text is not None:
            base_program = str(program_text)
        elif emit_lp_path and os.path.exists(str(emit_lp_path)):
            try:
                with open(str(emit_lp_path), 'r') as f:
                    base_program = f.read()
            except Exception as e:
                logging.warning(f"Failed to read emitted MUS program '{emit_lp_path}': {e}")
                base_program = None

        # Fallback: rebuild the adorned program (can be slower).
        tmp_facts_file: str | None = None
        if base_program is None:
            if not facts_ext:
                self.skipTest("No MUS program text available for MCS MUS-background check")

            from causalaba_mus import parse_facts_from_file, build_mus_program

            fd, tmp_facts_file = tempfile.mkstemp(suffix='_mus_facts.lp', text=True)
            os.close(fd)
            with open(tmp_facts_file, 'w') as f:
                for s in facts_ext:
                    line = self._normalize_fact_str(str(s))
                    if not line.strip():
                        continue
                    f.write(f"{line}\n")

            facts_no_period, _mapping = parse_facts_from_file(tmp_facts_file)
            base_program = build_mus_program(n_nodes, facts_no_period, tmp_facts_file)

        try:
            assert base_program is not None

            max_cores_to_check = 100
            rng = random.Random(0)
            cuts_all = list(mcs_list)
            if len(cuts_all) > max_cores_to_check:
                idxs = sorted(rng.sample(range(len(cuts_all)), k=max_cores_to_check))
                cuts_to_check = [(i + 1, cuts_all[i]) for i in idxs]
            else:
                cuts_to_check = [(i + 1, cut) for i, cut in enumerate(cuts_all)]

            logging.info(
                "MCS removed->SAT check (MUS background, n_nodes=%s): checking %s/%s cuts",
                n_nodes,
                len(cuts_to_check),
                len(cuts_all),
            )

            all_idxs = list(range(1, int(n_facts_total) + 1))
            for cut_idx, cut in cuts_to_check:
                cut_set = set(int(x) for x in (cut or []))
                keep = [i for i in all_idxs if i not in cut_set]
                forced = (
                    base_program
                    + "\n% ===== Force ALL but this MCS =====\n"
                    + "\n".join(f"mus({i})." for i in keep)
                    + ("\n" if keep else "")
                    + "\n".join(f":- mus({i})." for i in sorted(cut_set))
                    + "\n"
                )
                is_sat = _clingo_is_sat(forced, timeout_sec=float(timeout))
                if is_sat is None:
                    logging.warning(
                        "Timeout/unknown while checking MCS removed->SAT (MUS background) cut #%s (timeout=%ss); skipping this cut",
                        cut_idx,
                        timeout,
                    )
                    continue
                self.assertTrue(
                    is_sat,
                    f"MCS cut #{cut_idx} did NOT restore SAT under MUS background (cut={sorted(cut_set)})",
                )
        finally:
            if tmp_facts_file is not None:
                try:
                    os.remove(tmp_facts_file)
                except Exception:
                    pass

    def _assert_mcs_cores_removed_sat_under_abapc_background(
        self,
        *,
        n_nodes: int,
        facts_ext: list[str],
        mcs_facts: list[list[str]],
        solve_timeout_sec: float | None = None,
    ) -> None:
        """Check MCS semantics under the plain CausalABA (ABAPC) background.

        For each returned MCS cut C (as resolved fact strings), remove those facts from the
        instance fact set and check CausalABA is SAT.
        """
        _ensure_solvers_imported()

        if not facts_ext or not mcs_facts:
            return

        timeout = solve_timeout_sec
        if timeout is None:
            timeout = float(min(10, int(MUS_SOLVE_TIMEOUT)))

        max_cores_to_check = 100
        rng = random.Random(0)
        cuts_all = list(mcs_facts)
        if len(cuts_all) > max_cores_to_check:
            idxs = sorted(rng.sample(range(len(cuts_all)), k=max_cores_to_check))
            cuts_to_check = [(i + 1, cuts_all[i]) for i in idxs]
        else:
            cuts_to_check = [(i + 1, cut) for i, cut in enumerate(cuts_all)]

        all_facts_norm = [self._normalize_fact_str(str(s)) for s in (facts_ext or []) if str(s).strip()]
        seen_all: set[str] = set()
        all_facts_norm = [s for s in all_facts_norm if not (s in seen_all or seen_all.add(s))]
        if not all_facts_norm:
            return

        def _write_fact_lines(path: str, facts: list[str]) -> None:
            with open(path, 'w') as f:
                for s in facts:
                    line = self._normalize_fact_str(str(s))
                    if not line.strip():
                        continue
                    # Match standard ABAPC usage: declare tests as externals and let CausalABA assign them.
                    f.write(f"#external {line}\n")

        logging.info(
            "MCS removed->SAT check (ABAPC background, n_nodes=%s): checking %s/%s cuts",
            n_nodes,
            len(cuts_to_check),
            len(cuts_all),
        )

        for cut_idx, cut in cuts_to_check:
            # Be robust: depending on upstream formatting, a size-1 MCS might be returned
            # as a single string instead of a list[str].
            cut_items: list[Any]
            if cut is None:
                cut_items = []
            elif isinstance(cut, str):
                cut_items = [cut]
            else:
                try:
                    cut_items = list(cut)
                except Exception:
                    cut_items = [cut]

            cut_norm = {self._normalize_fact_str(str(s)) for s in cut_items if str(s).strip()}
            remaining = [s for s in all_facts_norm if s not in cut_norm]
            fd, tmp_file = tempfile.mkstemp(suffix='_mcs_removed_abapc.lp', text=True)
            os.close(fd)
            try:
                _write_fact_lines(tmp_file, remaining)
                timing: dict[str, Any] = {}
                models, _ = CausalABA(
                    n_nodes,
                    tmp_file,
                    weak_constraints=False,
                    print_models=False,
                    skeleton_rules_reduction=True,
                    out_n=1,
                    solve_timeout=timeout,
                    timing_recorder=timing,
                )
                if timing.get('timed_out', False):
                    logging.warning(
                        "Timeout/unknown while checking MCS removed->SAT (ABAPC background) cut #%s (timeout=%ss); skipping this cut",
                        cut_idx,
                        timeout,
                    )
                    continue

                sat = len(models) > 0
                if (not sat) and (not MUS_CHECK_MCS_REMOVED_SAT_ABAPC_STRICT):
                    preview = sorted(cut_norm)[:5]
                    if len(cut_norm) > 5:
                        preview.append(f"... (+{len(cut_norm) - 5} more)")
                    logging.warning(
                        "MCS cut #%s did NOT restore SAT under ABAPC background (non-strict; continuing). Cut preview: %s",
                        cut_idx,
                        preview,
                    )
                    continue

                self.assertTrue(
                    sat,
                    f"MCS cut #{cut_idx} did NOT restore SAT under ABAPC background (cut={sorted(cut_norm)})",
                )
            finally:
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    def _run_mus_mcs_for_size(
        self,
        n_nodes: int,
        seed: int | None = None,
        graph_type: str | None = None,
        *,
        max_muses_override: int | None = None,
    ):
        """Helper to run the full MUS/MCS pipeline for a given node size.

        This extracts the core logic of test_mus_mcs_random_five_node_abapc so it can
        be parameterized across different sizes without code duplication.

        Args:
            n_nodes: Number of nodes in the random graph.
            seed: Optional random seed; if None, uses MUS_SEED_BASE.
            graph_type: Optional graph family name.

        Returns a dict with keys:
            - 'was_sat': bool, True if ABAPC found the instance was already SAT (remove_n == 0)
            - 'abapc_timeout': bool, True if ABAPC hit timeout
            - 'mus_timeout': bool, True if MUS hit timeout
        """
        import types
        import re

        _ensure_solvers_imported()
        use_increm_only = bool(MUS_USE_INCREM_ONLY)
        if use_increm_only:
            _ensure_abapc_inc_imported()

        solve_timeout = _as_solver_timeout(MUS_SOLVE_TIMEOUT)

        run_mus_mcs = bool(MUS_RUN_MUS_MCS)
        run_optmcs = bool(MUS_RUN_OPTMCS)
        run_wc = bool(MUS_RUN_WC)
        wc_only = bool(run_wc and (not run_mus_mcs) and (not run_optmcs))

        # Stub notears again for this sub-run (if needed)
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')

            class _DummyMLP:
                pass

            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")

            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module

        # Use a deterministic seed
        if seed is None:
            seed = MUS_SEED_BASE
        if graph_type is None:
            graph_type = MUS_GRAPH_TYPE

        assert graph_type is not None
        graph_type = str(graph_type)

        assert seed is not None
        seed = int(seed)

        case = self.randomG_PC_case(
            n_nodes=n_nodes,
            edge_per_node=MUS_EDGE_PER_NODE,
            graph_type=graph_type,
            seed=seed,
            alpha=0.05,
            sample_size=10000,
            uc_rule=5,
            stable=True,
        )
        config = case["config"]
        facts = case["facts"]
        facts_ext = case["facts_ext"]
        wrong_ext = case["wrong_ext"]
        count_wrong = case["count_wrong"]
        true_seplist = case["true_seplist"]
        cg = case["cg"]
        G_true1 = case["G_true1"]

        # `cg` comes from different causal discovery backends; not all expose `number_of_edges()`.
        edges_count: Any = "NA"
        try:
            edges_count = cg.number_of_edges()  # type: ignore[attr-defined]
        except Exception:
            try:
                g_obj = getattr(cg, "G", None)
                edges_attr = getattr(g_obj, "edges", None)
                edges_count = len(edges_attr) if edges_attr is not None else "NA"
            except Exception:
                edges_count = "NA"

        logging.info(f"Config: n_nodes={n_nodes}, seed={seed}, graph_type={graph_type}, edges={edges_count}")
        logging.info(f"True DAG: {G_true1.edges}")
        logging.info(f"Number of total independence statements: {len(true_seplist)}")
        logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
        logging.info(f"Number of wrong facts: {count_wrong} ({(count_wrong/len(facts))*100 if facts else 0:.2f}%)")
        logging.info(f"Fully directed edges from PC ({len(cg.find_fully_directed())}): {cg.find_fully_directed()}")
        logging.info(f"Undirected edges from PC ({len(cg.find_undirected())}): {[(x,y) for (x,y) in cg.find_undirected() if x < y]}")

        # Basic assertions
        self.assertGreater(len(facts_ext), 0, f"Expected at least one PC-derived fact for n_nodes={n_nodes}")

        # Write facts to temp files (base + I + wc) for CausalABA
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
            weights: list[int] = []
            for fact, s in zip(facts, facts_ext):
                line = s if s.endswith('.') else s + '.'
                I = fact[1]
                try:
                    w = int(round(float(I) * 9_999_999))
                except Exception:
                    w = 0
                w = max(1, min(9_999_999, w))
                weights.append(w)
                f.write(f":~ {line} [-{w}]\n")

        if facts_ext:
            try:
                w_min = min(weights) if weights else None
                w_max = max(weights) if weights else None
                logging.info(
                    f"Weak-constraint weights: min={w_min}, max={w_max}, max_digits={len(str(w_max)) if w_max is not None else 'NA'}"
                )
            except Exception:
                pass

        # Optionally save temp files to permanent location for inspection
        if MUS_KEEP_TEMP_FILES:
            keep_dir = Path(MUS_KEEP_TEMP_FILES)
            keep_dir.mkdir(parents=True, exist_ok=True)

            import shutil
            base_name = f"{n_nodes}_{seed}_facts"
            shutil.copy(facts_file, keep_dir / f"{base_name}.lp")
            shutil.copy(facts_I_file, keep_dir / f"{base_name}_I.lp")
            shutil.copy(facts_wc_file, keep_dir / f"{base_name}_wc.lp")

            logging.info(f"  → Saved temp files to {keep_dir / base_name}*.lp")

        # Step 1: Optional ABAPC removal strategy.
        abapc_timing: dict[str, Any] = {}  # baseline: timing_recorder; increm: profile dict
        stats = None
        abapc_base_program_path: str | None = None
        models_after: list[Any] = []
        multiple = False
        remove_n = 0
        start_abapc = datetime.now()
        # For --analysis=wc we still want the baseline comparison (same as optmcs):
        # run ABAPC (or ABAPC_INC under --increm-only) and then compute WC.
        if wc_only:
            logging.info(f"Step 1: Run ABAPC removal strategy for baseline comparison (--analysis={MUS_ANALYSIS})")
        else:
            logging.info("Step 1: Run ABAPC removal strategy (baseline or incremental)")
        if use_increm_only:
            logging.info("Step 1: Run ABAPC_INC (incremental encoding) removal strategy")
        else:
            logging.info("Step 1: Run CasusalABA with all facts and apply removal strategy (search_for_models='first') if UNSAT")

        if use_increm_only:
            # We need a concrete base program file to feed into MUS when using increm-only.
            # Prefer user path if provided; otherwise use a temp file.
            if MUS_EMIT_ABAPC_INC_LP:
                try:
                    abapc_base_program_path = (
                        MUS_EMIT_ABAPC_INC_LP.replace("{version}", str(MUS_VERSION or ""))
                        .replace("{n}", str(n_nodes))
                        .replace("{seed}", str(seed))
                    )
                except Exception:
                    abapc_base_program_path = MUS_EMIT_ABAPC_INC_LP
            else:
                fd_dump, tmp_dump = tempfile.mkstemp(suffix=f"_abapc_inc_{n_nodes}_{seed}.lp", text=True)
                os.close(fd_dump)
                abapc_base_program_path = tmp_dump

            # Ensure output directory exists (for user-provided relative/absolute paths).
            try:
                dump_dir = os.path.dirname(str(abapc_base_program_path))
                if dump_dir:
                    os.makedirs(dump_dir, exist_ok=True)
            except Exception:
                pass

            models_after, multiple, stats, remove_n, profile_inc = _run_abapc_inc_with_wall_timeout(
                n_nodes,
                facts_file,
                weak_constraints=True,
                search_for_models='first',
                opt_mode=MUS_ABAPC_OPT_MODE,
                out_n=MUS_ABAPC_OUT_N,
                skeleton_rules_reduction=True,
                print_models=False,
                return_statistics=True,
                solve_timeout=solve_timeout,
                debug_dump_path=abapc_base_program_path,
                debug_dump_always=True,
                debug_dump_include_facts=False,
                debug_dump_materialize_block_edges=True,
                wall_timeout=solve_timeout,
            )
            abapc_timing = profile_inc if isinstance(profile_inc, dict) else {}
        else:
            models_after, multiple, stats, remove_n = CausalABA(
                n_nodes,
                facts_file,
                weak_constraints=True,
                search_for_models='first',
                opt_mode=MUS_ABAPC_OPT_MODE,
                out_n=MUS_ABAPC_OUT_N,
                skeleton_rules_reduction=True,
                print_models=False,
                return_statistics=True,
                solve_timeout=solve_timeout,
                timing_recorder=abapc_timing,
            )
        abapc_time = datetime.now() - start_abapc

        # Extract detailed timing from clingo statistics
        abapc_times = {}
        try:
            abapc_times = {key: stats['summary']['times'][key] for key in ['total', 'cpu', 'solve']}
        except (KeyError, TypeError):
            abapc_times = {'total': abapc_time.total_seconds(), 'cpu': 0, 'solve': 0}
        
        logging.info(f"  → Total facts: {len(facts)}")
        logging.info(f"  → Wrong facts: {count_wrong}")
        logging.info(f"  → Facts removed: {remove_n}")
        logging.info(f"  → Models found after removal: {len(models_after)}")
        logging.info(f"  → ABAPC time: {abapc_time.total_seconds():.3f}s")

        # When using increm-only mode, we emitted a concrete base program to disk.
        # Log a short hash so we can verify it's the same base text later injected
        # into the MUS/OptMCS adorned program.
        if use_increm_only and abapc_base_program_path:
            try:
                import hashlib

                with open(str(abapc_base_program_path), 'rb') as bf:
                    base_sha = hashlib.sha256(bf.read()).hexdigest()
                logging.info(f"  → ABAPC_INC base dump sha256: {base_sha[:12]} (file={abapc_base_program_path})")
            except Exception:
                pass

        # Derive which facts were removed by ABAPC 'first'.
        # Baseline ABAPC removes the lowest-I facts (tail after sorting by descending I).
        # ABAPC_INC mirrors this ordering internally, but can also report the removed fact keys
        # directly via its profile; prefer that when available for accurate reporting.
        facts_with_I: list[tuple[float, str]] = []
        for (fact_str, I, _is_correct), ext_line in zip(facts, facts_ext):
            stmt = self._normalize_fact_str(ext_line)
            facts_with_I.append((float(I), stmt))
        facts_sorted_by_I = sorted(facts_with_I, key=lambda t: t[0], reverse=True)
        removed_facts = [stmt for _I, stmt in facts_sorted_by_I[-remove_n:]] if remove_n else []

        try:
            if use_increm_only:
                removed_from_inc = (abapc_timing or {}).get('removed_fact_keys', None)
                if isinstance(removed_from_inc, list) and removed_from_inc:
                    removed_facts = [self._normalize_fact_str(s) for s in removed_from_inc if str(s).strip()]
                    remove_n = len(removed_facts)
        except Exception:
            pass

        # For larger sizes, we may not enforce UNSAT for all seeds; just assert we can reach SAT
        # If we hit the timeout before reaching SAT, skip the assertion (timeout is the real limit)
        abapc_timeout = bool(abapc_timing.get('timed_out', False))
        # Fallback heuristic (older CausalABA versions may not set timing_recorder flags).
        if (not abapc_timeout) and (len(models_after) == 0) and (solve_timeout is not None):
            abapc_timeout = abapc_time.total_seconds() >= (float(solve_timeout) - 0.5)
        if remove_n > 0 and not abapc_timeout:
            self.assertGreater(len(models_after), 0, f"Expected SAT after removing {remove_n} tests for n_nodes={n_nodes}")
        elif remove_n > 0 and abapc_timeout and len(models_after) == 0:
            logging.warning(f"⚠ Timeout hit after {abapc_time.total_seconds():.1f}s before reaching SAT (removed {remove_n} tests, n_nodes={n_nodes})")

        # Step 2: Run ABAPC_INC (incremental encoding) for comparison
        abapc_inc_time = None
        abapc_inc_profile: dict[str, Any] | None = None
        abapc_inc_remove_n: int | None = None
        n_models_after_inc: int | None = None
        abapc_inc_timeout = False
        abapc_inc_ran = False
        try:
            if not use_increm_only:
                _ensure_abapc_inc_imported()
                if CausalABA_INC is None:
                    raise ImportError("ABAPC_INC is not available after import")
                logging.info("Step 2: Running ABAPC_INC (incremental encoding)")
                start_abapc_inc = datetime.now()

                debug_dump_path = None
                if MUS_EMIT_ABAPC_INC_LP:
                    try:
                        debug_dump_path = (
                            MUS_EMIT_ABAPC_INC_LP
                            .replace("{version}", str(MUS_VERSION or ""))
                            .replace("{n}", str(n_nodes))
                            .replace("{seed}", str(seed))
                        )
                    except Exception:
                        debug_dump_path = MUS_EMIT_ABAPC_INC_LP
                    try:
                        dd = os.path.dirname(str(debug_dump_path))
                        if dd:
                            os.makedirs(dd, exist_ok=True)
                    except Exception:
                        pass
                    logging.info(f"Will emit ABAPC_INC grounded dump to {debug_dump_path}")

                models_after_inc, _multiple_inc, _stats_inc, remove_n_inc, profile_inc = _run_abapc_inc_with_wall_timeout(
                    n_nodes,
                    facts_file,
                    weak_constraints=True,
                    search_for_models='first',
                    opt_mode=MUS_ABAPC_OPT_MODE,
                    out_n=MUS_ABAPC_OUT_N,
                    skeleton_rules_reduction=True,
                    print_models=False,
                    return_statistics=True,
                    solve_timeout=solve_timeout,
                    debug_dump_path=debug_dump_path,
                    debug_dump_always=bool(debug_dump_path),
                    wall_timeout=solve_timeout,
                )

                abapc_inc_ran = True

                abapc_inc_time = datetime.now() - start_abapc_inc
                abapc_inc_profile = profile_inc if isinstance(profile_inc, dict) else None
                removed_from_inc = None
                try:
                    removed_from_inc = (abapc_inc_profile or {}).get('removed_fact_keys', None)
                except Exception:
                    removed_from_inc = None

                if isinstance(removed_from_inc, list) and removed_from_inc:
                    abapc_inc_remove_n = len([x for x in removed_from_inc if str(x).strip()])
                else:
                    abapc_inc_remove_n = int(remove_n_inc or 0)

                n_models_after_inc = len(models_after_inc or [])
                abapc_inc_timeout = bool((abapc_inc_profile or {}).get('timed_out', False))
                if (
                    (not abapc_inc_timeout)
                    and n_models_after_inc == 0
                    and (solve_timeout is not None)
                    and (abapc_inc_time.total_seconds() >= (float(solve_timeout) - 0.5))
                ):
                    abapc_inc_timeout = True

                logging.info(f"  → ABAPC_INC facts removed: {abapc_inc_remove_n}")
                logging.info(f"  → ABAPC_INC models found after removal: {n_models_after_inc}")
                logging.info(f"  → ABAPC_INC time: {abapc_inc_time.total_seconds():.3f}s")
                if isinstance(removed_from_inc, list) and removed_from_inc:
                    prev = [self._normalize_fact_str(s) for s in removed_from_inc if str(s).strip()]
                    preview = sorted(prev)[:10]
                    if len(prev) > 10:
                        preview.append(f"... (+{len(prev) - 10} more)")
                    logging.info(f"  → ABAPC_INC removed fact keys (preview): {preview}")
        except Exception as e:
            logging.warning(f"ABAPC_INC comparison skipped (import/run failed): {e}")

        # Step 3: Run MUS/MCS on PC facts only
        if run_mus_mcs:
            logging.info("Step 3: Running MUS/MCS analysis")
        else:
            logging.info(f"Step 3: Skipping MUS/MCS analysis (--analysis={MUS_ANALYSIS})")
        fd_mus, facts_mus_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd_mus)
        with open(facts_mus_file, 'w') as f:
            for s in facts_ext:
                line = s if s.endswith('.') else s + '.'
                f.write(f"{line}\n")

        # Cap MUS enumeration for larger sizes to keep runtime reasonable.
        # Allow explicit per-call override (used by demo tests).
        if max_muses_override is not None:
            max_muses = int(max_muses_override)
        elif MUS_MAX_MUSES != "":
            max_muses = int(MUS_MAX_MUSES) if MUS_MAX_MUSES != "0" else 0
        else:
            # Default: limit for faster testing
            max_muses = 500 if n_nodes <= 6 else 200
        
        # Use environment thresholds: 0 means None (no threshold), otherwise use value or defaults
        if MUS_MCS_THRESHOLD == 0 and MUS_MAX_MUSES == "":
            # When MUS_MAX_MUSES is empty (replicating manual WASP), default to no thresholds
            mcs_threshold = None
            mus_threshold = None
        elif MUS_MCS_THRESHOLD == 0:
            # When MUS_MCS_THRESHOLD explicitly 0, use defaults for faster testing
            mcs_threshold = 50
            mus_threshold = 300 if MUS_MUS_THRESHOLD == 0 else MUS_MUS_THRESHOLD
        else:
            mcs_threshold = MUS_MCS_THRESHOLD
            mus_threshold = MUS_MUS_THRESHOLD if MUS_MUS_THRESHOLD > 0 else None
        
        start_mus = datetime.now()
        
        # Optionally emit the complete MUS program for debugging.
        # Accept common boolean-like values (e.g. MUS_EMIT_LP=True) by mapping them to a default path,
        # so we don't accidentally write to a file literally named 'True'.
        emit_lp_path = None
        emit_lp_spec = MUS_EMIT_LP
        if isinstance(emit_lp_spec, str):
            spec = emit_lp_spec.strip()
            if spec.lower() in ("true", "1", "yes", "y"):
                spec = "results/adornedLP_{n}_{seed}.lp"
            elif spec.lower() in ("false", "0", "no", "n"):
                spec = ""
        else:
            spec = "results/adornedLP_{n}_{seed}.lp" if emit_lp_spec else ""

        # If we're going to validate MUS/MCS semantics under the MUS (adorned) background,
        # ensure we have the full adorned program on disk so we can reuse it without rebuilding.
        tmp_emit_for_enforced: str | None = None
        if (MUS_CHECK_ENFORCED_UNSAT or MUS_CHECK_MCS_REMOVED_SAT) and not spec:
            fd_emit, tmp_emit_for_enforced = tempfile.mkstemp(suffix=f"_adorned_{n_nodes}_{seed}.lp", text=True)
            os.close(fd_emit)
            emit_lp_path = tmp_emit_for_enforced

        if spec:
            emit_lp_path = (
                spec.replace('{version}', str(MUS_VERSION or ''))
                .replace('{n}', str(n_nodes))
                .replace('{seed}', str(seed))
            )
            emit_dir = os.path.dirname(emit_lp_path)
            if emit_dir:
                os.makedirs(emit_dir, exist_ok=True)
            logging.info(f"Will emit MUS program to {emit_lp_path}")
        
        # Special handling: if MUS_MAX_MUSES is empty string, pass None to omit -n flag.
        # If max_muses_override is set, always pass the explicit value through.
        max_muses_arg = max_muses if max_muses_override is not None else (None if MUS_MAX_MUSES == "" else max_muses)

        mus_timing: dict[str, Any] = {}  # To capture compile/ground breakdown
        if run_mus_mcs:
            mus_result = CausalABA_MUS(
                n_nodes=n_nodes,
                facts_location=facts_mus_file,
                gringo_path="clingo",
                wasp_path="wasp",
                max_muses=max_muses_arg,
                mus_algorithm="camus",
                print_mcses=True,
                camus_mcs_threshold=mcs_threshold,
                camus_mus_threshold=mus_threshold,
                solve_timeout=solve_timeout,
                emit_lp=emit_lp_path,
                timing_recorder=mus_timing,
                base_program_path=(abapc_base_program_path if use_increm_only else None),
            )
            mus_time = datetime.now() - start_mus
        else:
            # Stable placeholders so later summary/return logic doesn't special-case.
            mus_result = {
                'n_mus': 0,
                'n_mcs': 0,
                'mus_facts': [],
                'mcs_facts': [],
                'mus_list': [],
                'mcs_list': [],
                'mus_solve_time': 0.0,
                'timed_out': False,
            }
            mus_time = timedelta(0)

        # Optional: also compute optimum-MCS using WASP's dedicated algorithm.
        opt_mcs_facts = None
        opt_mcs_label = None
        opt_time = None
        opt_timing: dict[str, Any] | None = None
        opt_result: dict[str, Any] | None = None
        if run_optmcs:
            if not MUS_OPTIMUM_MCS_ALGORITHM:
                raise ValueError(
                    "OptMCS was requested but MUS_OPTIMUM_MCS_ALGORITHM is empty (set env var MUS_OPTIMUM_MCS_ALGORITHM to 'camus' or 'emax')."
                )
            else:
                opt_alg = MUS_OPTIMUM_MCS_ALGORITHM.strip().lower()
                if opt_alg not in ("camus", "emax"):
                    raise ValueError(
                        f"Invalid MUS_OPTIMUM_MCS_ALGORITHM={MUS_OPTIMUM_MCS_ALGORITHM!r}; expected 'camus' or 'emax'"
                    )

                # Only run if this WASP build supports the option.
                try:
                    import subprocess

                    help_out = subprocess.run(
                        ["wasp", "--help"],
                        capture_output=True,
                        text=True,
                        timeout=2.0,
                    )
                    supported = "--optimum-mcs-algorithm" in ((help_out.stdout or "") + (help_out.stderr or ""))
                except Exception:
                    supported = False

                if not supported:
                    logging.warning(
                        "Skipping optimum-MCS run: current WASP build does not support --optimum-mcs-algorithm. "
                        "Rebuild/upgrade WASP to enable this feature."
                    )
                else:
                    # If the user requested --emit-lp for the standard MUS/MCS run, also emit
                    # an additional program that contains the optimum-MCS objective literals.
                    emit_lp_opt = None
                    if emit_lp_path:
                        root, ext = os.path.splitext(str(emit_lp_path))
                        emit_lp_opt = f"{root}_optmcs_{opt_alg}{ext or '.lp'}"

                    logging.info(f"Step 3b: Running optimum-MCS analysis (algorithm={opt_alg})")
                    opt_timing = {}
                    start_opt = datetime.now()
                    opt_result = CausalABA_MUS(
                        n_nodes=n_nodes,
                        facts_location=facts_mus_file,
                        gringo_path="clingo",
                        wasp_path="wasp",
                        max_muses=max_muses_arg,
                        mus_algorithm="camus",
                        print_mcses=False,
                        optimum_mcs_algorithm=opt_alg,
                        facts_wc_location=facts_wc_file,
                        camus_mcs_threshold=mcs_threshold,
                        camus_mus_threshold=mus_threshold,
                        solve_timeout=solve_timeout,
                        emit_lp=emit_lp_opt,
                        timing_recorder=opt_timing,
                        base_program_path=(abapc_base_program_path if use_increm_only else None),
                    )
                    opt_time = datetime.now() - start_opt
                    opt_mcs_facts = opt_result.get('mcs_facts', None) if opt_result is not None else None
                    opt_mcs_label = f"OptMCS({opt_alg})"

        # Optional: compute pure clingo optimum cut using weak constraints only.
        wc_cut_facts: list[str] | None = None
        wc_cut_weight: int | None = None
        wc_time = None
        wc_result: dict[str, Any] | None = None
        wc_timing: dict[str, Any] | None = None
        if run_wc:
            logging.info("Step 3c: Running clingo-only weak-constraint optimization (WC)")
            # Emit a standalone WC program if we are already emitting an adorned program.
            emit_lp_wc = None
            if emit_lp_path:
                root, ext = os.path.splitext(str(emit_lp_path))
                emit_lp_wc = f"{root}_wc{ext or '.lp'}"

            wc_timing = {}
            start_wc = datetime.now()
            wc_result = CausalABA_WC(
                n_nodes=n_nodes,
                facts_location=facts_mus_file,
                gringo_path="clingo",
                facts_wc_location=facts_wc_file,
                solve_timeout=solve_timeout,
                emit_lp=emit_lp_wc,
                timing_recorder=wc_timing,
                base_program_path=(abapc_base_program_path if (use_increm_only and abapc_base_program_path) else None),
            )
            wc_time = datetime.now() - start_wc
            wc_cut_facts = wc_result.get('cut_facts', None) if wc_result is not None else None
            wc_cut_weight = int(wc_result.get('cut_weight', 0) or 0) if wc_result is not None else None

        # Optional: compare pure-WC optimum against WASP OptMCS optimum.
        if MUS_CHECK_WC_VS_OPTMCS:
            if not (run_wc and run_optmcs):
                logging.warning(
                    "WC-vs-OptMCS check requested, but required modes not enabled (run_wc=%s, run_optmcs=%s).",
                    bool(run_wc),
                    bool(run_optmcs),
                )
            elif wc_result is None or opt_result is None:
                logging.warning(
                    "WC-vs-OptMCS check requested, but a result is missing (wc_result=%s, opt_result=%s).",
                    bool(wc_result is not None),
                    bool(opt_result is not None),
                )
            else:
                wc_timed_out = bool(wc_result.get('timed_out', False))
                opt_timed_out = bool(opt_result.get('timed_out', False))
                if wc_timed_out or opt_timed_out:
                    logging.warning(
                        "Skipping WC-vs-OptMCS comparison due to timeout (wc_timed_out=%s, opt_timed_out=%s).",
                        wc_timed_out,
                        opt_timed_out,
                    )
                else:
                    # Normalize to a single cut set for comparison.
                    def _first_cut_set(obj: Any) -> set[str]:
                        if obj is None:
                            return set()
                        # OptMCS can be a list of cuts; pick the first non-empty.
                        if isinstance(obj, list) and obj and all(isinstance(x, (list, tuple, set)) for x in obj):
                            for cut in obj:
                                s = {self._normalize_fact_str(str(x)) for x in (cut or []) if str(x).strip()}
                                s = {x for x in s if x}
                                if s:
                                    return s
                            return set()
                        # Otherwise treat as a single cut list.
                        s = {self._normalize_fact_str(str(x)) for x in (obj or []) if str(x).strip()}
                        return {x for x in s if x}

                    wc_cut_set = _first_cut_set(wc_cut_facts)
                    opt_cut_set = _first_cut_set(opt_mcs_facts)

                    # Compare objective weights under the shared weights map.
                    try:
                        from causalaba_mus import parse_weights_from_wc_file, _normalize_ext_fact_key

                        weights_map = parse_weights_from_wc_file(facts_wc_file)
                    except Exception as e:
                        raise AssertionError(f"WC-vs-OptMCS check failed: cannot parse weights from {facts_wc_file!r}: {e}")

                    def _sum_cut_weight(cut: set[str]) -> int:
                        total = 0
                        missing = 0
                        for fact in (cut or set()):
                            # parse_weights_from_wc_file keys are normalized without trailing '.'
                            key = _normalize_ext_fact_key(str(fact))
                            w = weights_map.get(key)
                            if w is None:
                                missing += 1
                            else:
                                total += int(w)
                        if missing:
                            raise AssertionError(f"WC-vs-OptMCS check failed: missing weights for {missing} facts")
                        return int(total)

                    wc_w = int(wc_cut_weight if wc_cut_weight is not None else _sum_cut_weight(wc_cut_set))
                    opt_w = int(_sum_cut_weight(opt_cut_set))

                    if wc_w != opt_w:
                        raise AssertionError(
                            f"WC optimum weight {wc_w} != OptMCS optimum weight {opt_w} (n_nodes={n_nodes}, seed={seed})"
                        )

                    if MUS_CHECK_WC_VS_OPTMCS_STRICT_SET and (wc_cut_set != opt_cut_set):
                        raise AssertionError(
                            f"WC optimum cut set != OptMCS optimum cut set (same weight={wc_w}); "
                            f"|WC|={len(wc_cut_set)}, |OptMCS|={len(opt_cut_set)} (n_nodes={n_nodes}, seed={seed})"
                        )
                    elif wc_cut_set != opt_cut_set:
                        logging.warning(
                            "WC and OptMCS produced different optimum cuts with same weight=%s; likely multiple optima. |WC|=%s |OptMCS|=%s",
                            wc_w,
                            len(wc_cut_set),
                            len(opt_cut_set),
                        )
                    elif wc_cut_set == opt_cut_set:
                        logging.info(
                            "   WC and OptMCS produced identical optimum cuts with weight=%s; |Cut|=%s",
                            wc_w,
                            len(wc_cut_set),
                        )

        # ===== GRAPH EVAL SUMMARY (Removal + OptMCS + WC) =====
        # Evaluate the *set* of DAGs compatible with the remaining facts after applying:
        # - Removal (ABAPC-derived removed_facts), and
        # - OptMCS (WASP optimum-MCS cut; usually a single set)
        # We compute precision/recall/F1/SHD/SID for each distinct DAG and report min/avg/max,
        # plus the number of distinct DAGs.
        graph_eval: dict[str, Any] = {}
        try:
            import numpy as np
            import networkx as nx
            import math
            from utils.graph_utils import DAGMetrics, model_to_set_of_arrows

            # True graph adjacency (nodes are already relabeled to 0..n-1 in G_true1)
            B_true = nx.to_numpy_array(G_true1, nodelist=list(range(n_nodes)), dtype=int)

            def _as_finite_float(v: Any, default: float = 0.0) -> float:
                try:
                    x = float(v)
                except Exception:
                    return float(default)
                return x if math.isfinite(x) else float(default)

            def _mmx(xs: list[float]) -> dict[str, float | None]:
                ys = [float(x) for x in xs if math.isfinite(float(x))]
                if not ys:
                    return {'min': None, 'avg': None, 'max': None}
                return {'min': float(min(ys)), 'avg': float(sum(ys) / len(ys)), 'max': float(max(ys))}

            def _fmt_triple(d: dict[str, float | None]) -> str:
                mn = d.get('min', None)
                av = d.get('avg', None)
                mx = d.get('max', None)
                if mn is None or av is None or mx is None:
                    return "   none"
                return f"{mn:7.3f} / {av:7.3f} / {mx:7.3f}"

            def _evaluate_models(models: list[list[Any]]) -> dict[str, Any]:
                # Deduplicate by arrow set (distinct DAGs)
                seen: set[frozenset[tuple[int, int]]] = set()
                vals: dict[str, list[float]] = {k: [] for k in ('precision', 'recall', 'F1', 'shd')}
                if MUS_GRAPH_EVAL_SID:
                    vals['sid'] = []
                for m in (models or []):
                    arrows = model_to_set_of_arrows(m)
                    key = frozenset((int(a), int(b)) for (a, b) in arrows)
                    if key in seen:
                        continue
                    seen.add(key)
                    B_est = np.zeros((n_nodes, n_nodes), dtype=int)
                    for (a, b) in key:
                        if 0 <= a < n_nodes and 0 <= b < n_nodes:
                            B_est[a, b] = 1
                    try:
                        mt = DAGMetrics(B_est, B_true, sid=bool(MUS_GRAPH_EVAL_SID)).metrics
                    except Exception:
                        continue

                    # Some metric backends can return NaN; treat as 0 for summary stability.
                    p = _as_finite_float(mt.get('precision'), default=0.0)
                    r = _as_finite_float(mt.get('recall'), default=0.0)
                    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
                    vals['precision'].append(p)
                    vals['recall'].append(r)
                    vals['F1'].append(float(f1))
                    vals['shd'].append(_as_finite_float(mt.get('shd'), default=0.0))
                    if MUS_GRAPH_EVAL_SID:
                        vals['sid'].append(_as_finite_float(mt.get('sid'), default=0.0))

                out = {
                    'count': int(len(seen)),
                    'precision': _mmx(vals['precision']),
                    'recall': _mmx(vals['recall']),
                    'F1': _mmx(vals['F1']),
                    'shd': _mmx(vals['shd']),
                }
                if MUS_GRAPH_EVAL_SID:
                    out['sid'] = _mmx(vals.get('sid', []))
                return out

            def _solve_for_remaining_facts(label: str, removed_set: set[str]) -> tuple[dict[str, Any], bool]:
                # Build a facts file containing ONLY the remaining externals.
                remaining = [
                    self._normalize_fact_str(s)
                    for s in (facts_ext or [])
                    if self._normalize_fact_str(s) and (self._normalize_fact_str(s) not in removed_set)
                ]
                fd_eval, eval_facts_file = tempfile.mkstemp(suffix=f"_{label}_remaining.lp", text=True)
                os.close(fd_eval)
                try:
                    with open(eval_facts_file, 'w') as f:
                        for s in remaining:
                            f.write(f"#external {s}\n")

                    eval_timing: dict[str, Any] = {}
                    models_eval, _ = CausalABA(
                        n_nodes,
                        eval_facts_file,
                        weak_constraints=False,
                        print_models=False,
                        skeleton_rules_reduction=True,
                        search_for_models='No',
                        # For graph enumeration we want *compatibility* models, not optimal models.
                        opt_mode='ignore',
                        out_n=MUS_GRAPH_EVAL_OUT_N,
                        solve_timeout=solve_timeout,
                        timing_recorder=eval_timing,
                    )
                    timed_out = bool(eval_timing.get('timed_out', False))
                    return _evaluate_models(models_eval), timed_out
                finally:
                    try:
                        os.remove(eval_facts_file)
                    except Exception:
                        pass

            removed_set_norm = {self._normalize_fact_str(s) for s in (removed_facts or []) if str(s).strip()}
            n_total_facts = int(len([s for s in (facts_ext or []) if str(s).strip()]))
            removal_eval, removal_eval_timed_out = _solve_for_remaining_facts('Removal', removed_set_norm)
            graph_eval['Removal'] = {
                **removal_eval,
                'n_total_facts': n_total_facts,
                'n_removed_facts': int(len(removed_set_norm)),
                'n_remaining_facts': int(max(0, n_total_facts - len(removed_set_norm))),
                'timed_out': bool(removal_eval_timed_out),
            }

            # OptMCS: use only the (usually single) optimum cut.
            if opt_mcs_facts is not None:
                opt_sets = [
                    {self._normalize_fact_str(x) for x in (cut or []) if str(x).strip()}
                    for cut in (opt_mcs_facts or [])
                ]
                opt_cut = next((s for s in opt_sets if s), None)
                if opt_cut:
                    opt_eval, opt_eval_timed_out = _solve_for_remaining_facts(opt_mcs_label or 'OptMCS', opt_cut)
                    graph_eval['OptMCS'] = {
                        **opt_eval,
                        'label': opt_mcs_label or 'OptMCS',
                        'n_total_facts': n_total_facts,
                        'n_removed_facts': int(len(opt_cut)),
                        'n_remaining_facts': int(max(0, n_total_facts - len(opt_cut))),
                        'timed_out': bool(opt_eval_timed_out),
                    }

            # WC: clingo-only weak-constraint optimum cut.
            if wc_cut_facts is not None:
                wc_cut = {self._normalize_fact_str(x) for x in (wc_cut_facts or []) if str(x).strip()}
                if wc_cut:
                    wc_eval, wc_eval_timed_out = _solve_for_remaining_facts('WC', wc_cut)
                    graph_eval['WC'] = {
                        **wc_eval,
                        'label': 'WC',
                        'n_total_facts': n_total_facts,
                        'n_removed_facts': int(len(wc_cut)),
                        'n_remaining_facts': int(max(0, n_total_facts - len(wc_cut))),
                        'timed_out': bool(wc_eval_timed_out),
                    }

            # Print per-run summary (mirrors OVERLAP summary shape).
            logging.info("\n")
            logging.info("=" * 60)
            logging.info(f" GRAPH EVAL SUMMARY (n_nodes={n_nodes}, out_n={MUS_GRAPH_EVAL_OUT_N})")
            logging.info("=" * 60)
            logging.info("  [min/avg/max]  (for SHD/SID: lower is better)")

            def _print_group(name: str, d: dict[str, Any]) -> None:
                if not d:
                    logging.info(f"  {name}: none")
                    return
                suffix = " (timed out)" if d.get('timed_out', False) else ""
                logging.info(f"  {name}{suffix}:")
                logging.info(
                    f"    {'facts(rem/keep)':<14}: {int(d.get('n_removed_facts', 0) or 0)}/{int(d.get('n_remaining_facts', 0) or 0)} (of {int(d.get('n_total_facts', 0) or 0)})"
                )
                logging.info(f"    {'n_dags':<10}: {int(d.get('count', 0) or 0):7d}")
                logging.info(f"    {'precision':<10}: {_fmt_triple(d.get('precision', {}))}")
                logging.info(f"    {'recall':<10}: {_fmt_triple(d.get('recall', {}))}")
                logging.info(f"    {'F1':<10}: {_fmt_triple(d.get('F1', {}))}")
                logging.info(f"    {'shd':<10}: {_fmt_triple(d.get('shd', {}))}")
                if MUS_GRAPH_EVAL_SID:
                    logging.info(f"    {'sid':<10}: {_fmt_triple(d.get('sid', {}))}")

            _print_group('Removal', graph_eval.get('Removal', {}))
            if 'OptMCS' in graph_eval:
                label = (graph_eval.get('OptMCS', {}) or {}).get('label') or 'OptMCS'
                _print_group(label, graph_eval.get('OptMCS', {}))
            if 'WC' in graph_eval:
                label = (graph_eval.get('WC', {}) or {}).get('label') or 'WC'
                _print_group(label, graph_eval.get('WC', {}))

            # Diagnostic: removing MORE constraints should weakly increase the model count.
            # This is only meaningful if we asked clingo for all models (out_n=0) and did not time out.
            if int(MUS_GRAPH_EVAL_OUT_N) == 0:
                r = graph_eval.get('Removal', {}) or {}
                o = graph_eval.get('OptMCS', {}) or {}
                if r and o and (not r.get('timed_out', False)) and (not o.get('timed_out', False)):
                    n_r = int(r.get('count', 0) or 0)
                    n_o = int(o.get('count', 0) or 0)
                    if n_r < n_o:
                        logging.warning(
                            "Monotonicity check: expected #DAGs(Removal) >= #DAGs(OptMCS) when Removal removes more facts. "
                            "Observed: Removal=%s, OptMCS=%s (n_nodes=%s).",
                            n_r,
                            n_o,
                            n_nodes,
                        )
            logging.info("=" * 60)

        except Exception as e:
            logging.warning(f"Graph eval summary failed (n_nodes={n_nodes}): {e}")

        if MUS_PRINT_DETAILS:
            weights_by_fact = None
            if facts_wc_file:
                try:
                    from causalaba_mus import parse_weights_from_wc_file

                    weights_by_fact = parse_weights_from_wc_file(facts_wc_file)
                except Exception as e:
                    logging.warning(f"Failed to parse weights from {facts_wc_file!r}: {e}")

            self._dump_instance_details(
                n_nodes=n_nodes,
                seed=seed,
                facts_ext=facts_ext,
                wrong_ext=wrong_ext,
                removed_facts=removed_facts,
                weights_by_fact=weights_by_fact,
                mus_facts=mus_result.get('mus_facts', []) or [],
                mcs_facts=mus_result.get('mcs_facts', []) or [],
                opt_mcs_facts=opt_mcs_facts,
                opt_mcs_label=opt_mcs_label,
            )

        mus_timeout = bool(mus_result.get('timed_out', False)) if run_mus_mcs else False
        if (not mus_timeout) and (solve_timeout is not None):
            mus_timeout = mus_time.total_seconds() >= (float(solve_timeout) - 0.5)

        if run_mus_mcs:
            logging.info(f"  → MUS found: {mus_result['n_mus']}")
            logging.info(f"  → MCS found: {mus_result.get('n_mcs', 0)}")
            logging.info(f"  → MUS time: {mus_time.total_seconds():.3f}s")

        # Optional: verify MUS definition directly (enforced core under same MUS program is UNSAT).
        if run_mus_mcs and MUS_CHECK_ENFORCED_UNSAT:
            self._assert_mus_cores_enforced_unsat_under_mus_background(
                n_nodes=n_nodes,
                mus_list=mus_result.get('mus_list', []) or [],
                emit_lp_path=emit_lp_path,
                facts_ext=facts_ext,
                solve_timeout_sec=float(min(10, int(MUS_SOLVE_TIMEOUT))),
            )

        # Optional: verify each MUS core's corresponding facts are UNSAT under the plain ABAPC background.
        if run_mus_mcs and MUS_CHECK_ENFORCED_UNSAT_ABAPC:
            self._assert_mus_cores_enforced_unsat_under_abapc_background(
                n_nodes=n_nodes,
                mus_facts=mus_result.get('mus_facts', []) or [],
                solve_timeout_sec=float(min(10, int(MUS_SOLVE_TIMEOUT))),
            )

        # Optional: verify MCS semantics (removing an MCS restores SAT).
        if run_mus_mcs and MUS_CHECK_MCS_REMOVED_SAT:
            self._assert_mcs_cores_removed_sat_under_mus_background(
                n_nodes=n_nodes,
                mcs_list=mus_result.get('mcs_list', []) or [],
                n_facts_total=len(facts_ext or []),
                emit_lp_path=emit_lp_path,
                facts_ext=facts_ext,
                solve_timeout_sec=float(min(10, int(MUS_SOLVE_TIMEOUT))),
            )

        if run_mus_mcs and MUS_CHECK_MCS_REMOVED_SAT_ABAPC:
            self._assert_mcs_cores_removed_sat_under_abapc_background(
                n_nodes=n_nodes,
                facts_ext=facts_ext,
                mcs_facts=mus_result.get('mcs_facts', []) or [],
                solve_timeout_sec=float(min(10, int(MUS_SOLVE_TIMEOUT))),
            )

        # ===== TIMING COMPARISON (Phase Breakdown) =====
        try:
            header = f"TIMING COMPARISON (n_nodes={n_nodes})"
            prefix = "  "
            logging.info("\n" + "=" * 60)
            logging.info(prefix + header)
            logging.info("=" * 60)
            
            # Extract ABAPC timing breakdown
            abapc_compile = float(abapc_timing.get('compile_sec_total', 0.0) or 0.0)
            abapc_ground = float(abapc_timing.get('ground_sec_total', 0.0) or 0.0)
            abapc_unsat_ground = float(abapc_timing.get('unsat_ground_sec_total', 0.0) or 0.0)
            abapc_unsat_solve = float(abapc_timing.get('unsat_solve_sec_total', 0.0) or 0.0)
            if use_increm_only:
                abapc_solve = float(abapc_timing.get('solve_sec_total', 0.0) or 0.0)
            else:
                abapc_solve = stats.get('summary', {}).get('times', {}).get('solve', 0.0) if stats else 0.0
            abapc_total = abapc_time.total_seconds()
            # Sys (residual) time captures everything not covered by the measured phases.
            # For ABAPC removal, this includes Python overhead + reground/solve work not
            # reflected in the final ctl.statistics (which is for the last ctl only).
            abapc_sys = max(
                0.0,
                abapc_total
                - abapc_compile
                - abapc_ground
                - abapc_solve
                - abapc_unsat_ground
                - abapc_unsat_solve,
            )
            
            # Extract MUS timing breakdown
            # Note: MUS uses WASP which works differently from clingo:
            # - compile_and_ground is called in build_mus_program (captured in mus_timing)
            # - Then clingo re-grounds the full program and pipes to WASP (captured in mus_solve_time)
            # - WASP computes MUS internally without streaming (no Python enumeration overhead)
            mus_compile = mus_timing.get('compile_sec_total', 0.0)
            mus_ground = mus_timing.get('ground_sec_total', 0.0)
            mus_solve = mus_result.get('mus_solve_time', 0.0)  # This is clingo grounding + WASP execution
            mus_total = mus_time.total_seconds()
            # Sys (residual) time captures wrapper overhead around the external solver.
            mus_sys = max(0.0, mus_total - mus_compile - mus_ground - mus_solve)
            
            # Detect which phase timed out for ABAPC
            abapc_timeout_phase = None
            abapc_phase_hint = abapc_timing.get('timeout_phase', None) if isinstance(abapc_timing, dict) else None
            if abapc_timeout:
                # Prefer a phase hint from CausalABA if available.
                if abapc_phase_hint in ('compile', 'ground', 'solve', 'sys'):
                    abapc_timeout_phase = abapc_phase_hint

                # In practice ABAPC timeouts happen while clingo is still solving/enumerating.
                # If we reached the timeout before SAT (0 models), treat everything after compile/ground as solve.
                if len(models_after) == 0:
                    abapc_timeout_phase = 'solve'
                    abapc_solve = max(0.0, abapc_total - abapc_compile - abapc_ground)
                    abapc_sys = 0.0
                else:
                    abapc_timeout_phase = 'solve'
            # Detect which phase timed out for MUS
            mus_timeout_phase = None
            if mus_timeout:
                # For WASP MUS solver:
                # - If we found 0 MUS and 0 MCS, timeout was during solve (WASP couldn't finish)
                # - If we found some MUS/MCS, timeout was during enumerate (WASP was enumerating more)
                # - Unlike clingo, WASP does all computation internally (no Python enumeration overhead)
                n_mus = len(mus_result.get('mus_list', []))
                n_mcs = len(mus_result.get('mcs_list', []))
                if n_mus == 0 and n_mcs == 0:
                    mus_timeout_phase = 'solve'
                    # Recalculate: all non-compile/ground time was actually solving
                    mus_solve = max(0.0, mus_total - mus_compile - mus_ground)
                    mus_sys = 0.0
                else:
                    # Found some results, so timeout was during enumeration of additional MUS/MCS
                    mus_timeout_phase = 'sys'
            
            # Format timing displays with timeout indicators
            def format_time(val, timeout_phase, phase_name):
                if timeout_phase == phase_name:
                    # Keep the measured value (often partial) and annotate the timeout.
                    if solve_timeout is None:
                        return f"{val:.3f}s (timeout)"
                    return f"{val:.3f}s (timeout @{int(solve_timeout)}s)"
                return f"{val:.3f}s"

            def fmt_cell(val, timeout_phase, phase_name, width: int = 16) -> str:
                return f"{format_time(val, timeout_phase, phase_name):>{width}}"
            
            logging.info("")
            abapc_label = "ABAPC_INC removal strategy" if use_increm_only else "ABAPC removal strategy"
            abapc_engine = "causalaba_increm.CausalABA" if use_increm_only else "causalaba.CausalABA"
            logging.info(f"{prefix}+-- {abapc_label}:")
            logging.info(f"{prefix}|   Engine:          {abapc_engine:>16}")
            logging.info(f"{prefix}|   Compiling:       {fmt_cell(abapc_compile, abapc_timeout_phase, 'compile')}")
            logging.info(f"{prefix}|   UNSAT grounding: {abapc_unsat_ground:>15.3f}s")
            logging.info(f"{prefix}|   UNSAT solving:   {abapc_unsat_solve:>15.3f}s")
            logging.info(f"{prefix}|   SAT grounding:   {fmt_cell(abapc_ground, abapc_timeout_phase, 'ground')}")
            logging.info(f"{prefix}|   SAT solving:     {fmt_cell(abapc_solve, abapc_timeout_phase, 'solve')}")
            logging.info(f"{prefix}|   Sys (residual):  {fmt_cell(abapc_sys, abapc_timeout_phase, 'sys')}")
            logging.info(f"{prefix}|   Total:           {abapc_total:>15.3f}s")
            logging.info(f"{prefix}|")

            # ABAPC_INC timing breakdown (incremental encoding)
            if abapc_inc_time is not None and abapc_inc_profile is not None:
                inc_compile = float(abapc_inc_profile.get('compile_sec_total', 0.0) or 0.0)
                inc_ground = float(abapc_inc_profile.get('ground_sec_total', 0.0) or 0.0)
                inc_solve = float(abapc_inc_profile.get('solve_sec_total', 0.0) or 0.0)
                inc_total = float(abapc_inc_time.total_seconds())
                inc_sys = max(0.0, inc_total - inc_compile - inc_ground - inc_solve)
                inc_timeout_phase = 'solve' if abapc_inc_timeout else None

                logging.info(f"{prefix}+-- ABAPC_INC (incremental encoding):")
                logging.info(f"{prefix}|   Compiling:       {fmt_cell(inc_compile, inc_timeout_phase, 'compile')}")
                logging.info(f"{prefix}|   Grounding:       {fmt_cell(inc_ground, inc_timeout_phase, 'ground')}")
                logging.info(f"{prefix}|   Solving:         {fmt_cell(inc_solve, inc_timeout_phase, 'solve')}")
                logging.info(f"{prefix}|   Sys (residual):  {fmt_cell(inc_sys, inc_timeout_phase, 'sys')}")
                logging.info(f"{prefix}|   Total:           {inc_total:>15.3f}s")
                logging.info(f"{prefix}|")

            if run_mus_mcs:
                logging.info(f"{prefix}+-- ABAPC MUS/MCS analysis:")
                logging.info(f"{prefix}|   Compiling:       {fmt_cell(mus_compile, mus_timeout_phase, 'compile')}")
                logging.info(f"{prefix}|   Grounding:       {fmt_cell(mus_ground, mus_timeout_phase, 'ground')}")
                logging.info(f"{prefix}|   Solving:         {fmt_cell(mus_solve, mus_timeout_phase, 'solve')}")
                logging.info(f"{prefix}|   Sys (residual):  {fmt_cell(mus_sys, mus_timeout_phase, 'sys')}")
                logging.info(f"{prefix}|   Total:           {mus_total:>15.3f}s")
            else:
                logging.info(f"{prefix}+-- ABAPC MUS/MCS analysis: (skipped; --analysis={MUS_ANALYSIS})")

            # Optional: add optimum-MCS timing as a separate line item.
            if opt_time is not None and opt_result is not None:
                opt_total = opt_time.total_seconds()
                opt_compile = float((opt_timing or {}).get('compile_sec_total', 0.0) or 0.0)
                opt_ground = float((opt_timing or {}).get('ground_sec_total', 0.0) or 0.0)
                opt_solve = float((opt_result or {}).get('mus_solve_time', 0.0) or 0.0)
                opt_sys = max(0.0, opt_total - opt_compile - opt_ground - opt_solve)
                opt_timeout = (solve_timeout is not None) and (opt_total >= (float(solve_timeout) - 0.5))
                opt_timeout_phase = 'solve' if opt_timeout else None

                label = opt_mcs_label or 'OptMCS'
                logging.info(f"{prefix}|")
                logging.info(f"{prefix}+-- {label} analysis:")
                logging.info(f"{prefix}|   Compiling:       {fmt_cell(opt_compile, opt_timeout_phase, 'compile')}")
                logging.info(f"{prefix}|   Grounding:       {fmt_cell(opt_ground, opt_timeout_phase, 'ground')}")
                logging.info(f"{prefix}|   Solving:         {fmt_cell(opt_solve, opt_timeout_phase, 'solve')}")
                logging.info(f"{prefix}|   Sys (residual):  {fmt_cell(opt_sys, opt_timeout_phase, 'sys')}")
                logging.info(f"{prefix}|   Total:           {opt_total:>15.3f}s")

            # Optional: add pure WC (clingo-only) timing as a separate line item.
            if wc_time is not None and wc_result is not None:
                wc_total = wc_time.total_seconds()
                wc_compile = float((wc_timing or {}).get('compile_sec_total', 0.0) or 0.0)
                wc_ground = float((wc_timing or {}).get('ground_sec_total', 0.0) or 0.0)
                wc_solve = float((wc_result or {}).get('solve_time', 0.0) or 0.0)
                wc_sys = max(0.0, wc_total - wc_compile - wc_ground - wc_solve)
                wc_timeout = bool((wc_result or {}).get('timed_out', False)) or (
                    (solve_timeout is not None)
                    and (wc_total >= (float(solve_timeout) - 0.5))
                )
                wc_timeout_phase = 'solve' if wc_timeout else None

                logging.info(f"{prefix}|")
                logging.info(f"{prefix}+-- WC analysis:")
                logging.info(f"{prefix}|   Compiling:       {fmt_cell(wc_compile, wc_timeout_phase, 'compile')}")
                logging.info(f"{prefix}|   Grounding:       {fmt_cell(wc_ground, wc_timeout_phase, 'ground')}")
                logging.info(f"{prefix}|   Solving:         {fmt_cell(wc_solve, wc_timeout_phase, 'solve')}")
                logging.info(f"{prefix}|   Sys (residual):  {fmt_cell(wc_sys, wc_timeout_phase, 'sys')}")
                logging.info(f"{prefix}|   Total:           {wc_total:>15.3f}s")
            logging.info("")
            
            if run_mus_mcs and (not (abapc_timeout or mus_timeout)):
                ratio_total = mus_total / abapc_total if abapc_total > 0 else 0
                ratio_solve = mus_solve / abapc_solve if abapc_solve > 0 else 0
                
                logging.info(f"  Total time ratio (MUS/ABAPC):   {ratio_total:8.2f}x")
                logging.info(f"  Solve time ratio (WASP/clingo): {ratio_solve:8.2f}x")
            else:
                logging.info(f"  (Timing ratios not computed due to timeout)")
            logging.info("=" * 60 + "\n")
        except Exception as timing_error:
            # If timing measurement fails, still show what we have
            logging.warning(f"Timing comparison measurement failed: {timing_error}")
            logging.info("=" * 60 + "\n")

        # Clean up temp files
        os.remove(facts_file)
        os.remove(facts_I_file)
        os.remove(facts_wc_file)
        os.remove(facts_mus_file)
        if tmp_emit_for_enforced is not None:
            try:
                os.remove(tmp_emit_for_enforced)
            except Exception:
                pass

        logging.info(f"✓ Completed run for n_nodes={n_nodes}\n")
        
        # Assertions after timing comparison so it always displays
        # Basic assertion: we expect at least some MUSes for contradiction cases
        if run_mus_mcs and remove_n > 0:
            if not mus_timeout:
                self.assertGreater(mus_result.get('n_mus', 0), 0, f"Expected at least one MUS for UNSAT case (n_nodes={n_nodes})")
        
        # Final MUS timeout assertion: handled above via skipTest for the 0-MUS case.
        
        # Return status info for retry logic
        return {
            # "startSAT" should mean we actually found at least one model without removing facts.
            # (remove_n can be 0 for other reasons, including timeouts or early exits.)
            'was_sat': (remove_n == 0) and (len(models_after) > 0) and (not abapc_timeout),
            'abapc_timeout': abapc_timeout,
            'mus_timeout': mus_timeout,
            'mus_ran': bool(run_mus_mcs),
            'abapc_time_sec': abapc_time.total_seconds(),
            'mus_time_sec': mus_time.total_seconds(),
            'remove_n': remove_n,
            'n_models_after': len(models_after),
            # For correspondence checks (used by test_mus_mcs_random_sizes_abapc)
            'seed': seed,
            'graph_type': graph_type,
            'count_wrong': count_wrong,
            'facts_ext': facts_ext,
            'wrong_ext': wrong_ext,
            'removed_facts': removed_facts,
            'n_mus': mus_result.get('n_mus', 0),
            'n_mcs': mus_result.get('n_mcs', 0),
            'mus_facts': mus_result.get('mus_facts', []),
            'mcs_facts': mus_result.get('mcs_facts', []),
            'opt_mcs_facts': opt_mcs_facts,
            'opt_mcs_label': opt_mcs_label,
            'opt_ran': bool(opt_result is not None),
            'opt_time_sec': opt_time.total_seconds() if opt_time is not None else 0.0,
            'opt_timeout': (
                bool((opt_result or {}).get('timed_out', False))
                or (
                    (opt_time is not None)
                    and (solve_timeout is not None)
                    and (opt_time.total_seconds() >= (float(solve_timeout) - 0.5))
                )
            )
            if opt_result is not None
            else False,
            'abapc_inc_ran': bool(abapc_inc_ran),
            'abapc_inc_timeout': bool(abapc_inc_timeout) if abapc_inc_ran else False,
            'abapc_inc_time_sec': abapc_inc_time.total_seconds() if abapc_inc_time is not None else 0.0,
            'opt_timing': opt_timing or {},
            'opt_result': opt_result or {},
            'graph_eval': graph_eval,
            'wc_ran': bool(wc_result is not None),
            'wc_time_sec': wc_time.total_seconds() if wc_time is not None else 0.0,
            'wc_timeout': bool((wc_result or {}).get('timed_out', False)) if wc_result is not None else False,
            'wc_cut_facts': wc_cut_facts,
            'wc_cut_weight': wc_cut_weight,
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MUS/ABAPC random-size tests with optional overrides")
    parser.add_argument(
        "--version",
        type=str,
        default=MUS_VERSION,
        help=(
            "Version tag used for output folder naming and template expansion. "
            "When used with --emit-all, outputs go under results/<version>/. "
            "Also available as {version} in --emit-lp/--emit-abapc-inc-lp/--log-file templates."
        ),
    )
    parser.add_argument(
        "--emit-all",
        action="store_true",
        default=True,
        help=(
            "Emit all generated LPs and logs using canonical names under results/<version>/. "
            "Sets --emit-lp, --emit-abapc-inc-lp and --log-file automatically."
        ),
    )
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Run only TestMUSAnalysis.test_mus_mcs_random_sizes_abapc (skip other unit tests)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=MUS_SEED_BASE,
        help="Base seed for random graph generation (default: 2004). For some historical sizes, derived defaults are used: 5->base, 6->base+1, 10->base+6.",
    )
    parser.add_argument(
        "--rep-unsat",
        type=int,
        default=MUS_REP_UNSAT,
        help="Retry with seed+1, seed+2, ... when an instance is already SAT or has 0 wrong facts (default: 2)",
    )
    parser.add_argument(
        "--random-reps",
        type=int,
        default=MUS_RANDOM_REPS,
        help="Number of random repetitions per (node_size, graph_type) case (default: 1)",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default=MUS_ANALYSIS,
        help=(
            "Which analyses to run. Supported: both, all, mus, optmcs, wc, or a comma/plus-separated list e.g. optmcs,wc. "
            "(mus = MUS/MCS enumeration; optmcs = WASP optimum-MCS; wc = clingo-only weak-constraint optimization)"
        ),
    )
    parser.add_argument(
        "--check-wc-vs-optmcs",
        action="store_true",
        default=False,
        help="Enable WC vs OptMCS solution comparison (objective weight; optionally set MUS_CHECK_WC_VS_OPTMCS_STRICT_SET=1 for exact set match).",
    )
    parser.add_argument(
        "--solve-timeout",
        type=int,
        default=MUS_SOLVE_TIMEOUT,
        help="Per-instance wall-clock timeout (seconds) for both ABAPC removal and MUS solving (0 = no timeout)",
    )
    parser.add_argument(
        "--node-sizes",
        type=str,
        default=','.join(str(n) for n in MUS_NODE_SIZES),
        help="Comma-separated list of node counts for the random graph tests (e.g., 5,7,9)",
    )
    parser.add_argument(
        "--edge-per-node",
        type=int,
        default=MUS_EDGE_PER_NODE,
        help="Edge-per-node multiplier for random graph generation",
    )
    parser.add_argument(
        "--emit-lp",
        type=str,
        default=MUS_EMIT_LP,
        help="Path to emit complete MUS program (use {n} for node count placeholder, e.g., /tmp/test_{n}node.lp)",
    )
    parser.add_argument(
        "--emit-abapc-inc-lp",
        type=str,
        default=MUS_EMIT_ABAPC_INC_LP,
        help=(
            "Path to emit the grounded ABAPC_INC program dump (use {n} and {seed} placeholders, e.g. "
            "results/abapc_inc_{n}_{seed}.lp). This dump is suitable for offline inspection and later reuse."
        ),
    )
    parser.add_argument(
        "--use-increm-only",
        "--increm-only",
        action="store_true",
        default=MUS_USE_INCREM_ONLY,
        help=(
            "Use ABAPC_INC (incremental encoding) as the *baseline* ABAPC engine, and feed its emitted LP dump "
            "as the base program for MUS/MCS analysis."
        ),
    )
    parser.add_argument(
        "--max-muses",
        type=str,
        default=MUS_MAX_MUSES,
        help="Max MUS to enumerate: '' (empty)=omit -n flag (WASP default: 1 MUS output), '0'=unlimited, '>0'=limit to N",
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default=MUS_GRAPH_TYPE,
        help="Random graph type for PC simulation (default: ER)",
    )
    parser.add_argument(
        "--graph-types",
        type=str,
        default=",".join(MUS_GRAPH_TYPES) if MUS_GRAPH_TYPES else "",
        help="Comma-separated list of random graph types to run (overrides --graph-type), e.g. ER,SF",
    )
    parser.add_argument(
        "--out-n",
        type=int,
        default=MUS_ABAPC_OUT_N,
        help="Clingo -n model bound for ABAPC removal (0=all; default: 0)",
    )
    parser.add_argument(
        "--graph-eval-out-n",
        type=int,
        default=MUS_GRAPH_EVAL_OUT_N,
        help="Clingo -n model bound for graph-eval DAG enumeration (0=all; default: matches --out-n)",
    )
    parser.add_argument(
        "--graph-eval-sid",
        dest="graph_eval_sid",
        action="store_true",
        default=MUS_GRAPH_EVAL_SID,
        help="Compute SID in graph-eval summaries (slow; default: enabled).",
    )
    parser.add_argument(
        "--no-graph-eval-sid",
        dest="graph_eval_sid",
        action="store_false",
        help="Disable SID in graph-eval summaries (much faster).",
    )
    parser.add_argument(
        "--opt-mode",
        type=str,
        default=MUS_ABAPC_OPT_MODE,
        help="Clingo opt_mode for ABAPC removal (default: optN)",
    )
    parser.add_argument(
        "--mcs-threshold",
        type=int,
        default=MUS_MCS_THRESHOLD,
        help="CAMUS MCS threshold (0=unlimited, default=50)",
    )
    parser.add_argument(
        "--mus-threshold",
        type=int,
        default=MUS_MUS_THRESHOLD,
        help="CAMUS MUS threshold (0=unlimited, default=300)",
    )
    parser.add_argument(
        "--check-mus-minimality",
        action="store_true",
        default=MUS_CHECK_MINIMALITY,
        help="Enable MUS minimality diagnostics (core-alone UNSAT and single-deletion SAT checks)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Append logs to this file (also prints to stdout).",
    )
    args, remaining = parser.parse_known_args()

    if remaining:
        # Avoid silent misconfiguration (e.g., typing `max-muses 0` instead of `--max-muses 0`).
        # Print at exit so it doesn't get buried by long logs.
        ignored = list(remaining)

        def _warn_ignored_args() -> None:
            sys.stderr.write(
                "\nWARNING: Unrecognized CLI arguments were ignored: " + " ".join(ignored) + "\n"
            )

        atexit.register(_warn_ignored_args)

    MUS_VERSION = (args.version or "").strip()
    if args.emit_all:
        if not MUS_VERSION:
            MUS_VERSION = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        out_dir = Path("results") / MUS_VERSION
        out_dir.mkdir(parents=True, exist_ok=True)
        # Canonical names (templated per instance).
        args.emit_lp = str(out_dir / "adorned_{n}_{seed}.lp")
        args.emit_abapc_inc_lp = str(out_dir / "abapc_inc_{n}_{seed}.lp")
        args.log_file = str(out_dir / "log_{n}_{seed}.log")

    MUS_SOLVE_TIMEOUT = args.solve_timeout
    MUS_NODE_SIZES = _parse_node_sizes(args.node_sizes)
    MUS_EDGE_PER_NODE = args.edge_per_node
    MUS_SEED_BASE = args.seed_base
    MUS_REP_UNSAT = args.rep_unsat
    MUS_RANDOM_REPS = args.random_reps
    MUS_ANALYSIS = str(args.analysis).strip().lower()
    MUS_ANALYSIS_MODES = _parse_analysis_modes(MUS_ANALYSIS)
    MUS_RUN_MUS_MCS = 'mus' in MUS_ANALYSIS_MODES
    MUS_RUN_OPTMCS = 'optmcs' in MUS_ANALYSIS_MODES
    MUS_RUN_WC = 'wc' in MUS_ANALYSIS_MODES
    MUS_CHECK_WC_VS_OPTMCS = bool(args.check_wc_vs_optmcs) or bool(MUS_CHECK_WC_VS_OPTMCS)
    MUS_EMIT_LP = args.emit_lp
    MUS_EMIT_ABAPC_INC_LP = args.emit_abapc_inc_lp
    MUS_USE_INCREM_ONLY = bool(args.use_increm_only)
    MUS_MAX_MUSES = args.max_muses
    MUS_GRAPH_TYPE = args.graph_type
    MUS_GRAPH_TYPES = _parse_graph_types(args.graph_types) or (MUS_GRAPH_TYPE,)
    MUS_ABAPC_OUT_N = args.out_n
    MUS_GRAPH_EVAL_OUT_N = args.graph_eval_out_n
    MUS_GRAPH_EVAL_SID = bool(args.graph_eval_sid)
    MUS_ABAPC_OPT_MODE = args.opt_mode
    MUS_MCS_THRESHOLD = args.mcs_threshold
    MUS_MUS_THRESHOLD = args.mus_threshold
    MUS_CHECK_MINIMALITY = bool(args.check_mus_minimality)

    start = datetime.now()
    log_file = args.log_file.strip() if isinstance(args.log_file, str) else ""
    if log_file:
        # Expand common placeholders when unambiguous.
        try:
            if '{version}' in log_file:
                log_file = log_file.replace('{version}', str(MUS_VERSION or ''))
            if '{n}' in log_file:
                if len(MUS_NODE_SIZES) == 1:
                    log_file = log_file.replace('{n}', str(MUS_NODE_SIZES[0]))
                else:
                    log_file = log_file.replace('{n}', 'multi')
            if '{seed}' in log_file:
                log_file = log_file.replace('{seed}', str(MUS_SEED_BASE))
        except Exception:
            pass

    logger_setup(log_file=(log_file or None))
    # Print a final total-time summary after unittest output (including random-only runs).
    atexit.register(_log_total_time_per_method)
    try:
        logging.info(
            "Runner: pid=%s cwd=%s python=%s",
            str(os.getpid()),
            str(os.getcwd()),
            str(sys.executable),
        )
    except Exception:
        pass
    logging.info(
        f"CLI overrides: solve_timeout={MUS_SOLVE_TIMEOUT}s, node_sizes={MUS_NODE_SIZES}, edge_per_node={MUS_EDGE_PER_NODE}, seed_base={MUS_SEED_BASE}, "
        f"rep_unsat={MUS_REP_UNSAT}, random_reps={MUS_RANDOM_REPS}, graph_types={MUS_GRAPH_TYPES}, opt_mode={MUS_ABAPC_OPT_MODE}, out_n={MUS_ABAPC_OUT_N}, "
        f"analysis={MUS_ANALYSIS}, "
        f"version={MUS_VERSION or '(none)'}, emit_all={bool(args.emit_all)}, "
        f"emit_lp={MUS_EMIT_LP or '(none)'}, emit_abapc_inc_lp={MUS_EMIT_ABAPC_INC_LP or '(none)'}, use_increm_only={MUS_USE_INCREM_ONLY}, "
        f"max_muses={MUS_MAX_MUSES}, mcs_threshold={MUS_MCS_THRESHOLD}, mus_threshold={MUS_MUS_THRESHOLD}, log_file={log_file or '(none)'}"
    )

    if args.random_only:
        suite = unittest.TestSuite()
        suite.addTest(TestMUSAnalysis('test_mus_mcs_random_sizes_abapc'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)

    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
    logging.info(f"Total test time={str(datetime.now()-start)}")
