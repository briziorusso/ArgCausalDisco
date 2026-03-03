#!/usr/bin/env python3
"""WC opt-strategy/objective sweep across ABAPC encodings.

This script builds a single PC instance and compares:

- Objective: [C@1] (`sum`) vs [C@1,X] (`lex`)
- clingo `--opt-strategy`: {bb, usc}
- Encoding: incremental base dump vs baseline compile-and-ground

For context it also times the standard ABAPC removal runs
(`causalaba.CausalABA` and `causalaba_increm.CausalABA`).

Example (small default case):
  python scripts/wc_opt_strategy_sweep.py --n-nodes 7 --seed 2004 --timeout-sec 900 --out-dir results/wc_sweep_7_2004
"""

from __future__ import annotations

import argparse
import datetime as _dt
import signal
import json
import logging
import multiprocessing
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _build_pc_case_from_data(
    *,
    data: Any,
    B_true: Any,
    alpha: float,
    indep_test: str,
    uc_rule: int,
    uc_priority: int,
    stable: bool,
    strength_S_weight: bool,
) -> dict[str, Any]:
    """Build a sweep case from observed data + ground-truth adjacency.

    Produces the same fields as `tests_mus.build_random_pc_case`, but avoids
    enumerating all d-separation tests (which can explode for larger graphs).

    Facts are generated only for CI tests actually performed by PC (from
    `cg.sepset`), and are marked correct/incorrect using d-separation in the
    ground-truth DAG.
    """

    import numpy as np
    import pandas as pd
    import networkx as nx

    from utils.graph_utils import initial_strength

    try:
        from utils.data_utils import pc  # type: ignore
    except Exception:
        from cd_algorithms.PC import pc  # type: ignore

    B_true_arr = np.asarray(B_true)
    n_nodes = int(B_true_arr.shape[0])

    # True graph with X1..Xn labels (needed by `nx.algorithms.d_separated`).
    labels = [f"X{i+1}" for i in range(n_nodes)]
    B_df = pd.DataFrame(B_true_arr, columns=labels, index=labels)
    G_true = nx.DiGraph(B_df)
    # Integer-labeled true graph for graph-eval.
    G_true1 = nx.from_numpy_array(B_true_arr.astype(int), create_using=nx.DiGraph)

    # Run PC on the observed data.
    cg = pc(
        data=np.asarray(data),
        alpha=float(alpha),
        indep_test=str(indep_test),
        ikb=True,
        uc_rule=int(uc_rule),
        uc_priority=int(uc_priority),
        stable=bool(stable),
        verbose=False,
        show_progress=False,
    )

    fact_map: dict[tuple[str, int, int, tuple[int, ...]], tuple[float, bool]] = {}

    # Iterate only over CI tests PC actually considered.
    for x in range(n_nodes):
        for y in range(x + 1, n_nodes):
            try:
                tests_xy = cg.sepset[x, y]
            except Exception:
                tests_xy = None
            if not tests_xy:
                continue

            # Some causallearn versions store object arrays; normalize to a list.
            try:
                tests_list = list(tests_xy)
            except Exception:
                tests_list = []
            if not tests_list:
                continue

            for t in tests_list:
                if not t or not isinstance(t, (list, tuple)) or len(t) < 2:
                    continue
                S_raw, p_raw = t[0], t[1]
                try:
                    p = float(p_raw)
                except Exception:
                    continue

                try:
                    S_set = set(int(s) for s in S_raw)
                except Exception:
                    try:
                        S_set = set(int(s) for s in (S_raw or ()))
                    except Exception:
                        S_set = set()

                S_sorted = tuple(sorted(S_set))
                s_str = "empty" if len(S_sorted) == 0 else ("s" + "y".join(str(i) for i in S_sorted))

                # PC outcome.
                dep_type_pc = "indep" if p > float(alpha) else "dep"

                # Truth via d-separation in the true DAG.
                try:
                    Z = {f"X{s+1}" for s in S_sorted}
                    is_indep_true = bool(
                        nx.algorithms.d_separated(G_true, {f"X{x+1}"}, {f"X{y+1}"}, Z)
                    )
                except Exception:
                    # If d-separation fails for any reason, skip this test.
                    continue
                dep_type_true = "indep" if is_indep_true else "dep"
                is_correct = dep_type_pc == dep_type_true

                I = float(
                    initial_strength(
                        p,
                        len(S_sorted),
                        float(alpha),
                        0.5,
                        n_nodes,
                        S_weight=bool(strength_S_weight),
                    )
                )

                key = (dep_type_pc, int(x), int(y), tuple(int(s) for s in S_sorted))
                prev = fact_map.get(key)
                if prev is None:
                    fact_map[key] = (float(I), bool(is_correct))
                else:
                    prev_I, prev_correct = prev
                    # Correctness should be consistent; keep max strength.
                    fact_map[key] = (float(max(prev_I, I)), bool(prev_correct))

    facts: list[tuple[str, float, bool]] = []
    facts_ext: list[str] = []
    wrong_ext: list[str] = []
    count_wrong = 0

    for (dep_type_pc, x, y, S_sorted), (I, is_correct) in fact_map.items():
        s_str = "empty" if len(S_sorted) == 0 else ("s" + "y".join(str(i) for i in S_sorted))
        fact_str = f"{dep_type_pc}({x},{y},{s_str})."
        facts.append((fact_str, float(I), bool(is_correct)))
        ext_line = f"ext_{fact_str}"
        facts_ext.append(ext_line)
        if not is_correct:
            wrong_ext.append(ext_line)
            count_wrong += 1

    return {
        "B_true": B_true_arr,
        "G_true": G_true,
        "G_true1": G_true1,
        "data": np.asarray(data),
        "cg": cg,
        "facts": facts,
        "facts_ext": facts_ext,
        "wrong_ext": wrong_ext,
        "count_wrong": int(count_wrong),
    }


# Ensure local repo imports win when running as `python scripts/...`.
# When executed this way, Python sets `sys.path[0]` to the `scripts/` directory,
# which can cause imports like `from causalaba import CausalABA` to resolve to a
# different module (e.g., an installed package) instead of the repo's
# `causalaba.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_repo_symbol(module_filename: str, symbol: str) -> Any:
    """Import a symbol from a module file under this repo.

    When running `python scripts/...`, Python sets `sys.path[0]` to the scripts
    directory; importing by name can then accidentally resolve to an installed
    module with the same name instead of the repo-local one.
    """

    module_path = PROJECT_ROOT / module_filename
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module file: {module_path}")

    spec = importlib.util.spec_from_file_location(f"_wc_sweep_{module_path.stem}", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, symbol):
        raise AttributeError(f"{module_filename} has no attribute {symbol}")
    return getattr(mod, symbol)


def _import_repo_module(module_filename: str, module_name: str) -> Any:
    """Import a module from this repo under a specific module name."""
    module_path = PROJECT_ROOT / module_filename
    if not module_path.exists():
        raise FileNotFoundError(f"Missing module file: {module_path}")

    # IMPORTANT: this script often runs under `multiprocessing` with `fork`.
    # In that case the child inherits `sys.modules` from the parent, which may
    # already contain an installed/stale module with the same name. Only reuse
    # the cached module if it actually points at our repo-local file.
    try:
        cached = sys.modules.get(module_name)
        cached_file = Path(getattr(cached, "__file__", "") or "").resolve() if cached is not None else None
        if cached is not None and cached_file == module_path.resolve():
            return cached
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_SIGTERM_SNAPSHOT_HOOK: Any = None
_MAIN_PID: int = int(os.getpid())


def _install_sigterm_handler() -> None:
    """Best-effort SIGTERM handler.

    If this process is terminated externally (e.g. scheduler, container stop),
    we try to flush a final summary.partial.json snapshot so `--resume` can
    continue cleanly.

    NOTE: This does not prevent termination; it only improves observability.
    """

    def _handler(signum: int, _frame: Any) -> None:
        # Child workers created via fork inherit this handler. For worker
        # processes, exit immediately on SIGTERM (no traceback/snapshot) so
        # parent-triggered terminate() remains cheap and predictable.
        if int(os.getpid()) != int(_MAIN_PID):
            try:
                os._exit(128 + int(signum))
            except Exception:
                raise SystemExit(128 + int(signum))

        try:
            logging.error("Received SIGTERM (will exit 143). Writing snapshot if possible...")
        except Exception:
            pass

        # Dump a traceback for debugging (best-effort).
        try:
            import faulthandler

            faulthandler.dump_traceback(all_threads=True)
        except Exception:
            pass

        # Flush a snapshot if main() registered a hook.
        try:
            if callable(_SIGTERM_SNAPSHOT_HOOK):
                _SIGTERM_SNAPSHOT_HOOK()
        except Exception:
            pass

        raise SystemExit(128 + int(signum))

    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _causalaba_worker(
    result_queue: multiprocessing.Queue[tuple[str, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Run CausalABA in a child process and return only picklable results."""
    try:
        # Always import from the repo root (avoid resolving a stale installed module).
        CausalABA = getattr(_import_repo_module("causalaba.py", "causalaba"), "CausalABA")

        timing: dict[str, Any] = {}
        kwargs = dict(kwargs)
        # Always capture timing from the baseline implementation.
        # This is required for consistent build/solve/eval decomposition.
        kwargs["timing_recorder"] = timing

        result = CausalABA(*args, **kwargs)
        # CausalABA returns a list like [models, multiple] or [models, multiple, stats, remove_n].
        # IMPORTANT: clingo.Symbol objects are not reliably picklable across processes.
        # Serialize models to plain strings so the parent process always receives a result.
        models_after_raw: Any = []
        multiple: bool = False
        remove_n: int = 0
        timing_out: dict[str, Any] = timing
        if isinstance(result, (list, tuple)) and len(result) >= 2:
            models_after_raw = result[0]
            multiple = bool(result[1])
            if len(result) >= 4:
                try:
                    remove_n = int(result[3] or 0)
                except Exception:
                    remove_n = 0
            if len(result) >= 5 and isinstance(result[4], dict):
                # causalaba_binsearch returns a profile dict in position 4.
                timing_out = dict(result[4])

        models_after: list[Any] = []
        try:
            if isinstance(models_after_raw, list):
                for m in models_after_raw:
                    # Expected shape: list[list[clingo.Symbol]]
                    if isinstance(m, list):
                        models_after.append([str(s) for s in m])
                    else:
                        models_after.append(str(m))
        except Exception:
            models_after = []

        safe_payload = {
            "models_after": models_after,
            "multiple": bool(multiple),
            # clingo statistics are often not picklable; omit.
            "stats": None,
            "remove_n": int(remove_n),
            "timing": timing_out if isinstance(timing_out, dict) else {},
        }
        result_queue.put(("ok", safe_payload))
    except BaseException as e:
        result_queue.put(("err", (repr(e), traceback.format_exc())))


def _abapc_inc_worker(
    result_queue: multiprocessing.Queue[tuple[str, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Run ABAPC incremental solver in a child process (picklable payload)."""
    try:
        # Ensure intra-module imports inside causalaba_increm resolve to this repo.
        _import_repo_module("causalaba.py", "causalaba")
        try:
            _import_repo_module("causalaba_binsearch.py", "causalaba_binsearch")
        except Exception:
            pass
        inc_mod = _import_repo_module("causalaba_increm.py", "causalaba_increm")
        CausalABA_INC = getattr(inc_mod, "CausalABA")

        raw = CausalABA_INC(*args, **kwargs)
        # Expected shape when return_statistics=True: [models_out, multiple, stats, remove_n, profile]
        models_out, multiple, stats, remove_n, profile = raw

        def _stringify_models(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, list):
                return [_stringify_models(x) for x in obj]
            try:
                return str(obj)
            except Exception:
                return repr(obj)

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
    """Run ABAPC incremental solver with a hard wall-clock timeout (ground+solve)."""

    # In-process path.
    if wall_timeout is None or wall_timeout <= 0:
        _import_repo_module("causalaba.py", "causalaba")
        try:
            _import_repo_module("causalaba_binsearch.py", "causalaba_binsearch")
        except Exception:
            pass
        inc_mod = _import_repo_module("causalaba_increm.py", "causalaba_increm")
        CausalABA_INC = getattr(inc_mod, "CausalABA")
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
        proc.join(timeout=0.5)
        if proc.is_alive():
            try:
                proc.kill()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                proc.join(timeout=0.5)
            except Exception:
                pass

        timed_profile = {
            "timed_out": True,
            "timeout_phase": "wall",
            "timeout_s": float(wall_timeout),
        }
        return [], False, None, 0, timed_profile

    try:
        status, payload = result_queue.get(timeout=2.0)
    except Exception:
        logging.warning("⚠ ABAPC_INC subprocess exited without returning a result.")
        timed_profile = {
            "timed_out": True,
            "timeout_phase": "aborted",
            "timeout_s": float(wall_timeout),
        }
        return [], False, None, 0, timed_profile

    if status == "ok" and isinstance(payload, dict):
        models_after = payload.get("models_out", [])
        multiple = bool(payload.get("multiple", False))
        stats = payload.get("stats", None)
        remove_n = int(payload.get("remove_n", 0) or 0)
        profile = payload.get("profile", {})
        if not isinstance(profile, dict):
            profile = {}
        return models_after, multiple, stats, remove_n, profile

    err_repr, err_tb = payload
    raise RuntimeError(f"ABAPC_INC subprocess failed: {err_repr}\n{err_tb}")


def _run_causalaba_with_wall_timeout(
    *args: Any,
    wall_timeout: float | None,
    **kwargs: Any,
) -> tuple[list[Any], bool, Any, int, dict[str, Any]]:
    """Run CausalABA with a hard wall-clock timeout (ground+solve).

    This guards against cases where grounding/path-enumeration in native code
    does not yield control frequently enough for in-process deadline checks.
    """
    if wall_timeout is None or wall_timeout <= 0:
        # Use baseline-grounding + binary-search removal so baseline and
        # incremental agree on the minimal suffix-removal count.
        try:
            from causalaba_binsearch import CausalABA  # type: ignore
            use_timing_recorder = False
        except Exception:
            from causalaba import CausalABA  # type: ignore
            use_timing_recorder = True

        timing: dict[str, Any] = {}
        kwargs = dict(kwargs)
        if use_timing_recorder:
            kwargs["timing_recorder"] = timing
        else:
            kwargs.pop("timing_recorder", None)

        result = CausalABA(*args, **kwargs)
        models_after = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else []
        multiple = bool(result[1]) if isinstance(result, (list, tuple)) and len(result) > 1 else False
        remove_n = 0
        if isinstance(result, (list, tuple)) and len(result) >= 4:
            try:
                remove_n = int(result[3] or 0)
            except Exception:
                remove_n = 0
        timing_out: dict[str, Any] = timing
        if isinstance(result, (list, tuple)) and len(result) >= 5 and isinstance(result[4], dict):
            timing_out = dict(result[4])
        return (models_after if isinstance(models_after, list) else []), multiple, None, remove_n, timing_out

    try:
        ctx = multiprocessing.get_context("fork")
    except Exception:
        ctx = multiprocessing.get_context()

    result_queue: multiprocessing.Queue[tuple[str, Any]] = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_causalaba_worker, args=(result_queue, args, kwargs))
    proc.daemon = True
    proc.start()
    proc.join(timeout=float(wall_timeout))

    if proc.is_alive():
        logging.warning(f"⚠ CausalABA exceeded wall timeout ({wall_timeout}s); terminating.")
        proc.terminate()
        # Keep teardown bounded; the goal is to stop work, not to wait.
        proc.join(timeout=0.5)
        if proc.is_alive():
            try:
                proc.kill()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                proc.join(timeout=0.5)
            except Exception:
                pass
        # Best-effort: the worker may have already produced a result but the
        # parent hit the wall-time budget before `join()` observed exit.
        try:
            status, payload = result_queue.get_nowait()
            if status == "ok" and isinstance(payload, dict):
                models_after = payload.get("models_after", [])
                multiple = bool(payload.get("multiple", False))
                remove_n = int(payload.get("remove_n", 0) or 0)
                timing = payload.get("timing", {})
                if not isinstance(models_after, list):
                    models_after = []
                if not isinstance(timing, dict):
                    timing = {}
                timing = dict(timing)
                timing["timed_out"] = True
                timing.setdefault("timeout_phase", "wall")
                timing.setdefault("timeout_s", float(wall_timeout))
                return models_after, multiple, None, remove_n, timing
        except Exception:
            pass
        timing = {
            "timed_out": True,
            "timeout_phase": "wall",
            "timeout_s": float(wall_timeout),
        }
        return [], False, None, 0, timing

    try:
        # NOTE: `multiprocessing.Queue.get_nowait()` can race with the queue's
        # feeder thread/pipe flush, especially when the child exits quickly.
        # Use a short blocking read to avoid spurious "no result" timeouts.
        status, payload = result_queue.get(timeout=2.0)
    except Exception:
        logging.warning("⚠ CausalABA subprocess exited without returning a result.")
        timing = {
            # This is an abnormal termination / IPC failure, not a solver timeout.
            "timed_out": False,
            "timeout_phase": "aborted",
            "timeout_s": float(wall_timeout),
        }
        return [], False, None, 0, timing

    if status == "ok" and isinstance(payload, dict):
        models_after = payload.get("models_after", [])
        multiple = bool(payload.get("multiple", False))
        remove_n = int(payload.get("remove_n", 0) or 0)
        timing = payload.get("timing", {})
        if not isinstance(models_after, list):
            models_after = []
        if not isinstance(timing, dict):
            timing = {}
        return models_after, multiple, None, remove_n, timing

    err_repr, err_tb = payload
    raise RuntimeError(f"CausalABA subprocess failed: {err_repr}\n{err_tb}")


def _causalaba_wc_worker(
    result_queue: multiprocessing.Queue[tuple[str, Any]],
    kwargs: dict[str, Any],
) -> None:
    """Run CausalABA_WC in a child process and return only picklable results."""
    try:
        from causalaba_mus import CausalABA_WC

        res = CausalABA_WC(**kwargs)
        result_queue.put(("ok", res if isinstance(res, dict) else {"result": res}))
    except BaseException as e:
        result_queue.put(("err", (repr(e), traceback.format_exc())))


def _direct_wc_worker(
    result_queue: multiprocessing.Queue[tuple[str, Any]],
    kwargs: dict[str, Any],
) -> None:
    """Run the *direct* WC optimization (clingo Python API) in a child process.

    Motivation: clingo's Python bindings are native code; on large instances we
    can see SIGSEGV/abort/OOM-related crashes. Isolating keeps the sweep alive.
    """
    try:
        import clingo

        n_nodes = int(kwargs["n_nodes"])
        base_program_path = str(kwargs["base_program_path"])
        facts_path = str(kwargs["facts_path"])
        keys_in_file_order = list(kwargs["keys_in_file_order"])
        key_to_weight = dict(kwargs["key_to_weight"])
        objective = str(kwargs["objective"])
        opt_mode = str(kwargs["opt_mode"])
        opt_strategy = str(kwargs["opt_strategy"])
        threads = int(kwargs.get("threads", 1) or 1)
        solve_timeout_sec = float(kwargs.get("solve_timeout", 0.0) or 0.0)

        # Build weak constraints text from keys and weights.
        wc_text = _build_direct_wc_text(
            keys_in_file_order=keys_in_file_order,
            key_to_weight=key_to_weight,
            objective=objective,
        )

        build_t0 = time.perf_counter()
        base_program_text = Path(base_program_path).read_text()
        normalized_base_program = _normalize_base_program_for_analysis(base_program_text, n_nodes=n_nodes)

        # Write normalized base program to a temp file for clingo.
        normalized_base_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".lp", delete=False) as tmp:
                tmp.write(normalized_base_program)
                normalized_base_path = tmp.name
        except Exception:
            normalized_base_path = None

        build_sec = time.perf_counter() - build_t0

        solve_t0 = time.perf_counter()
        status_direct = "UNKNOWN"
        costs_direct: list[int] = []
        timed_out_direct = False
        accepted_keys_direct: set[str] = set()

        try:
            ctl = clingo.Control(
                [
                    f"--opt-mode={opt_mode}",
                    f"--opt-strategy={opt_strategy}",
                    "-n",
                    "1",
                    "-t",
                    str(threads),
                    "--warn=none",
                ]
            )
            if normalized_base_path is None:
                # Fallback: add the program via ctl.add.
                ctl.add("base", [], normalized_base_program)
            else:
                ctl.load(str(normalized_base_path))
            ctl.load(str(facts_path))
            if wc_text.strip():
                ctl.add("base", [], wc_text)
            ctl.ground([("base", [])])

            key_syms: list[tuple[str, clingo.Symbol]] = []
            for k in keys_in_file_order:
                try:
                    key_syms.append((k, clingo.parse_term(k)))
                except Exception:
                    continue

            for _k, sym in key_syms:
                try:
                    ctl.assign_external(sym, None)
                except Exception:
                    pass

            last_cost: list[int] = []
            last_keys: set[str] = set()
            last_optimality_proven: bool = False

            def _on_model(m: clingo.Model) -> None:
                nonlocal last_cost, last_keys, last_optimality_proven
                try:
                    last_cost = [int(x) for x in (m.cost or [])]
                except Exception:
                    last_cost = []
                try:
                    last_optimality_proven = bool(getattr(m, "optimality_proven", False))
                except Exception:
                    last_optimality_proven = False
                chosen: set[str] = set()
                for kk, ss in key_syms:
                    try:
                        if m.contains(ss):
                            chosen.add(kk)
                    except Exception:
                        continue
                last_keys = chosen

            if solve_timeout_sec <= 0.0:
                timed_out_direct = True
                finished = False
                handle = None
                res = None
            else:
                handle = ctl.solve(async_=True, on_model=_on_model)
                finished = bool(handle.wait(timeout=float(solve_timeout_sec)))
                if not finished:
                    timed_out_direct = True
                    try:
                        handle.cancel()
                    except Exception:
                        pass
                    try:
                        handle.wait(timeout=0.5)
                    except Exception:
                        pass
                res = None
                if finished and not timed_out_direct:
                    try:
                        res = handle.get()
                    except Exception:
                        res = None

            if timed_out_direct:
                status_direct = "TIMEOUT"
            elif res is not None and bool(getattr(res, "unsatisfiable", False)):
                status_direct = "UNSAT"
            elif res is not None and bool(getattr(res, "satisfiable", False)):
                status_direct = "OPT" if last_optimality_proven else "SAT"
            elif res is not None and bool(getattr(res, "unknown", False)):
                status_direct = "UNKNOWN"
            elif res is not None and bool(getattr(res, "interrupted", False)):
                status_direct = "UNKNOWN"
            else:
                status_direct = "UNKNOWN"

            costs_direct = list(last_cost or [])
            accepted_keys_direct = set(last_keys or set())

        except Exception:
            status_direct = "ERROR"
            costs_direct = []
            accepted_keys_direct = set()
            timed_out_direct = False

        solve_sec = time.perf_counter() - solve_t0

        try:
            if normalized_base_path:
                os.remove(normalized_base_path)
        except Exception:
            pass

        payload = {
            "status": status_direct,
            "costs": [int(x) for x in (costs_direct or [])],
            "timed_out": bool(timed_out_direct),
            "accepted_keys": sorted(str(k) for k in accepted_keys_direct),
            "build_time_sec": float(build_sec),
            "solve_time_sec": float(solve_sec),
        }
        result_queue.put(("ok", payload))
    except BaseException as e:
        result_queue.put(("err", (repr(e), traceback.format_exc())))


def _run_direct_wc_isolated(
    *,
    wall_timeout: float | None,
    kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, int | None, str | None]:
    """Run the direct clingo-Python-API solve in a subprocess.

    Returns (status, payload, exitcode, err_text) where status in:
      - ok: payload is dict
      - timeout: exceeded wall_timeout
      - killed: process exited by signal (e.g., SIGSEGV)
      - err: worker raised exception
    """

    if wall_timeout is None:
        # In-process run (no isolation)
        q: multiprocessing.Queue[tuple[str, Any]] = multiprocessing.Queue(maxsize=1)
        _direct_wc_worker(q, kwargs)
        st, payload = q.get(timeout=1.0)
        if st == "ok" and isinstance(payload, dict):
            return "ok", payload, 0, None
        return "err", None, 1, str(payload)

    if wall_timeout <= 0:
        return "timeout", None, None, "No time budget left"

    try:
        ctx = multiprocessing.get_context("fork")
    except Exception:
        ctx = multiprocessing.get_context()

    result_queue: multiprocessing.Queue[tuple[str, Any]] = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_direct_wc_worker, args=(result_queue, kwargs))
    proc.daemon = True
    proc.start()
    proc.join(timeout=float(wall_timeout))

    if proc.is_alive():
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=0.5)
        except Exception:
            pass
        try:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)
        except Exception:
            pass
        return "timeout", None, None, None

    exitcode = proc.exitcode
    if isinstance(exitcode, int) and exitcode < 0:
        return "killed", None, exitcode, None

    try:
        st, payload = result_queue.get(timeout=1.0)
    except Exception:
        return "err", None, exitcode, "No payload from direct worker"

    if st == "ok" and isinstance(payload, dict):
        return "ok", payload, exitcode, None

    err_repr, err_tb = payload
    return "err", None, exitcode, f"{err_repr}\n{err_tb}"


def _run_causalaba_wc_isolated(
    *,
    wall_timeout: float | None,
    kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any] | None, int | None, str | None]:
    """Run CausalABA_WC in a subprocess.

    Returns (status, payload, exitcode, err_text) where:
      - status in {"ok","timeout","killed","err"}
      - payload is the returned dict when ok
      - exitcode is the subprocess exit code (negative => signal)
    """

    if wall_timeout is None:
        # In-process run (no hard wall timeout requested).
        from causalaba_mus import CausalABA_WC

        res = CausalABA_WC(**kwargs)
        return "ok", (res if isinstance(res, dict) else {"result": res}), 0, None

    if wall_timeout <= 0:
        # No time budget left: do not run anything (and crucially, do not fall back
        # to an in-process run, which defeats isolation and can OOM the parent).
        return "timeout", None, None, None

    try:
        ctx = multiprocessing.get_context("fork")
    except Exception:
        ctx = multiprocessing.get_context()

    result_queue: multiprocessing.Queue[tuple[str, Any]] = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_causalaba_wc_worker, args=(result_queue, kwargs))
    proc.daemon = True
    proc.start()
    proc.join(timeout=float(wall_timeout))

    if proc.is_alive():
        # Hard timeout: terminate the worker and report as timeout.
        try:
            proc.terminate()
        except Exception:
            pass
        proc.join(timeout=5.0)
        if proc.is_alive():
            try:
                proc.kill()  # type: ignore[attr-defined]
            except Exception:
                pass
        return "timeout", None, proc.exitcode, None

    exitcode = proc.exitcode
    try:
        status, payload = result_queue.get_nowait()
    except Exception:
        # No payload returned: if killed by signal, surface that.
        if isinstance(exitcode, int) and exitcode < 0:
            return "killed", None, exitcode, None
        return "err", None, exitcode, "CausalABA_WC subprocess exited without returning a result."

    if status == "ok" and isinstance(payload, dict):
        return "ok", payload, exitcode, None
    if status == "err" and isinstance(payload, (list, tuple)) and len(payload) == 2:
        err_repr, err_tb = payload
        return "err", None, exitcode, f"{err_repr}\n{err_tb}"

    return "err", None, exitcode, "CausalABA_WC subprocess returned an unexpected payload."


# When run under VS Code's python runner (runpy.run_path), the workspace root
# is not guaranteed to be on sys.path. Ensure imports like `tests_mus` work.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class BaselineResult:
    rep: int
    solver: str  # "causalaba" | "causalaba_increm"
    opt_mode: str  # "opt" | "optN"
    wall_time_sec: float
    timed_out: bool
    removed: int
    models_after: int
    method_time_sec: float | None = None
    build_time_sec: float | None = None
    solve_time_sec: float | None = None
    eval_time_sec: float | None = None
    base_program_path: str | None = None
    accepted_weight: float | None = None
    n_tests_total: int = 0
    n_tests_true: int = 0
    n_tests_accepted: int = 0
    n_tests_accepted_true: int = 0
    weight_total: int = 0
    weight_true: int = 0
    weight_accepted_true: int = 0
    accepted_fact_f1: float | None = None
    graph_eval_timed_out: bool | None = None
    graph_eval_cache_hit: bool | None = None
    n_dags_compat: int | None = None
    true_dag_in_compat: int | None = None
    n_cpdags_compat: int | None = None
    true_cpdag_in_compat: int | None = None
    shd_avg: float | None = None
    f1_avg: float | None = None
    adjacency_f1_avg: float | None = None
    arrowhead_f1_avg: float | None = None
    shd_best: float | None = None
    shd_worst: float | None = None
    f1_best: float | None = None
    f1_worst: float | None = None
    adjacency_f1_best: float | None = None
    adjacency_f1_worst: float | None = None
    arrowhead_f1_best: float | None = None
    arrowhead_f1_worst: float | None = None
    cpdag_shd_avg: float | None = None
    cpdag_f1_avg: float | None = None
    cpdag_adjacency_f1_avg: float | None = None
    cpdag_arrowhead_f1_avg: float | None = None
    cpdag_shd_best: float | None = None
    cpdag_shd_worst: float | None = None
    cpdag_f1_best: float | None = None
    cpdag_f1_worst: float | None = None
    cpdag_adjacency_f1_best: float | None = None
    cpdag_adjacency_f1_worst: float | None = None
    cpdag_arrowhead_f1_best: float | None = None
    cpdag_arrowhead_f1_worst: float | None = None


@dataclass(frozen=True)
class StrategyResult:
    rep: int
    encoding: str  # "inc" | "base"
    objective: str  # "sum" | "lex"
    opt_strategy: str
    opt_mode: str  # "opt" | "optN"
    reification: str  # "mus" (assumption layer) | "direct" (ext_* choices)
    timed_out: bool
    wall_time_sec: float
    status: str
    objective_costs: list[int]
    method_time_sec: Optional[float] = None
    cut_weight: Optional[int] = None
    selected_count: Optional[int] = None
    build_time_sec: Optional[float] = None
    solve_time_sec: Optional[float] = None
    eval_time_sec: Optional[float] = None
    accepted_weight: Optional[float] = None
    n_tests_total: int = 0
    n_tests_true: int = 0
    n_tests_accepted: int = 0
    n_tests_accepted_true: int = 0
    weight_total: int = 0
    weight_true: int = 0
    weight_accepted_true: int = 0
    accepted_fact_f1: float | None = None
    graph_eval_timed_out: bool | None = None
    graph_eval_cache_hit: bool | None = None
    n_dags_compat: int | None = None
    true_dag_in_compat: int | None = None
    n_cpdags_compat: int | None = None
    true_cpdag_in_compat: int | None = None
    shd_avg: float | None = None
    f1_avg: float | None = None
    adjacency_f1_avg: float | None = None
    arrowhead_f1_avg: float | None = None
    shd_best: float | None = None
    shd_worst: float | None = None
    f1_best: float | None = None
    f1_worst: float | None = None
    adjacency_f1_best: float | None = None
    adjacency_f1_worst: float | None = None
    arrowhead_f1_best: float | None = None
    arrowhead_f1_worst: float | None = None
    cpdag_shd_avg: float | None = None
    cpdag_f1_avg: float | None = None
    cpdag_adjacency_f1_avg: float | None = None
    cpdag_arrowhead_f1_avg: float | None = None
    cpdag_shd_best: float | None = None
    cpdag_shd_worst: float | None = None
    cpdag_f1_best: float | None = None
    cpdag_f1_worst: float | None = None
    cpdag_adjacency_f1_best: float | None = None
    cpdag_adjacency_f1_worst: float | None = None
    cpdag_arrowhead_f1_best: float | None = None
    cpdag_arrowhead_f1_worst: float | None = None


def _f1_from_counts(*, tp: int, fp: int, fn: int) -> float:
    """Standard F1 from count triples (safe for zeros)."""
    tp_f = float(max(0, int(tp)))
    fp_f = float(max(0, int(fp)))
    fn_f = float(max(0, int(fn)))
    denom = (2.0 * tp_f + fp_f + fn_f)
    return float((2.0 * tp_f / denom) if denom > 0 else 0.0)


def _fact_f1(*, n_true: int, n_acc: int, n_acc_true: int) -> float:
    """F1 of accepted facts against true facts."""
    tp = int(n_acc_true)
    fp = int(n_acc - n_acc_true)
    fn = int(n_true - n_acc_true)
    return _f1_from_counts(tp=tp, fp=fp, fn=fn)


def _compute_acceptance_metrics(
    *,
    all_keys: list[str],
    accepted_keys: set[str],
    key_to_weight: dict[str, int],
    key_to_is_true: dict[str, bool],
) -> tuple[int, int, int, int, int, int, int, int]:
    """Return (n_total, n_true, n_acc, n_acc_true, w_total, w_true, w_acc, w_acc_true)."""
    keys_unique = list(dict.fromkeys(k for k in all_keys if k))
    n_total = len(keys_unique)
    n_true = 0
    n_acc = 0
    n_acc_true = 0
    w_total = 0
    w_true = 0
    w_acc = 0
    w_acc_true = 0

    for k in keys_unique:
        w = int(key_to_weight.get(k, 0) or 0)
        is_true = bool(key_to_is_true.get(k, False))
        w_total += w
        if is_true:
            n_true += 1
            w_true += w
        if k in accepted_keys:
            n_acc += 1
            w_acc += w
            if is_true:
                n_acc_true += 1
                w_acc_true += w

    return n_total, n_true, n_acc, n_acc_true, w_total, w_true, w_acc, w_acc_true


_EXT_RE = re.compile(r"^(?:#external\s+)?(ext_(?:dep|indep)\([^)]*\))\.?$")


def _normalize_fact_line(line: str) -> str:
    s = (line or "").strip()
    if not s:
        return ""
    if s.startswith("%") or s.startswith("#") and not s.startswith("#external"):
        return ""
    if s.startswith("#external"):
        s = s[len("#external") :].strip()
    if s.endswith("."):
        s = s[:-1]
    s = s.strip()
    m = _EXT_RE.match(s)
    if not m:
        return ""
    return m.group(1)


def _parse_ext_fact_line(line: str) -> tuple[str, int, int, tuple[int, ...]] | None:
    """Parse an ext fact like `ext_indep(0,1,s1y2)` (optionally with trailing '.')"""
    raw = (line or "").strip()
    if not raw:
        return None
    if raw.startswith("#external"):
        raw = raw[len("#external") :].strip()
    if raw.startswith("#") or raw.startswith("%"):
        return None
    if raw.endswith("."):
        raw = raw[:-1]

    m = re.fullmatch(r"(ext_indep|ext_dep)\((\d+)\s*,\s*(\d+)\s*,\s*([^\)\s]+)\)", raw)
    if not m:
        return None
    fact_type, x_str, y_str, s_sym = m.groups()
    x, y = int(x_str), int(y_str)

    if s_sym == "empty":
        s_tuple: tuple[int, ...] = ()
    elif s_sym.startswith("s"):
        tail = s_sym[1:]
        if not tail:
            s_tuple = ()
        else:
            parts = [p for p in tail.split("y") if p]
            try:
                s_tuple = tuple(int(p) for p in parts)
            except Exception:
                return None
    else:
        return None

    return fact_type, x, y, s_tuple


def _flip_fact_str(fact_str: str) -> str:
    if "indep" in fact_str:
        return fact_str.replace("indep", "dep", 1)
    if "dep" in fact_str:
        return fact_str.replace("dep", "indep", 1)
    return fact_str


def _apply_pct_wrong_facts(
    facts: list[tuple[str, float, bool]],
    pct_wrong: float,
    *,
    rng: random.Random,
) -> tuple[list[tuple[str, float, bool]], int]:
    if not facts:
        return facts, 0
    pct = max(0.0, min(1.0, float(pct_wrong)))
    total = len(facts)
    target_wrong = int(round(pct * total))

    wrong_idx = [i for i, (_, _, is_correct) in enumerate(facts) if not is_correct]
    correct_idx = [i for i, (_, _, is_correct) in enumerate(facts) if is_correct]

    if target_wrong > len(wrong_idx):
        need = min(target_wrong - len(wrong_idx), len(correct_idx))
        flip_idx = rng.sample(correct_idx, need) if need > 0 else []
    elif target_wrong < len(wrong_idx):
        need = min(len(wrong_idx) - target_wrong, len(wrong_idx))
        flip_idx = rng.sample(wrong_idx, need) if need > 0 else []
    else:
        flip_idx = []

    facts_out = list(facts)
    for i in flip_idx:
        fact_str, I, is_correct = facts_out[i]
        facts_out[i] = (_flip_fact_str(fact_str), I, not is_correct)

    wrong_count = sum(1 for _, _, is_correct in facts_out if not is_correct)
    return facts_out, wrong_count


def _write_instance_files(
    out_dir: Path,
    *,
    facts_ext: list[str],
    weights: list[int],
    I_values: list[float],
    truths: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    facts_path = out_dir / "facts.lp"
    facts_I_path = out_dir / "facts_I.lp"
    wc_path = out_dir / "facts_wc.lp"

    with facts_path.open("w") as f:
        for atom in facts_ext:
            f.write(f"#external {atom}.\n")

    if truths is None:
        truths = ["unknown"] * len(facts_ext)

    with facts_I_path.open("w") as f:
        for atom, I, truth in zip(facts_ext, I_values, truths):
            # ABAPC_INC convention: when weak_constraints=True, it reads facts_location.replace('.lp','_I.lp')
            # and expects each line to include an " I=<float>,<label>" suffix.
            f.write(f"#external {atom}. I={float(I)},{truth}\n")

    with wc_path.open("w") as f:
        for atom, w in zip(facts_ext, weights):
            f.write(f":~ {atom}. [-{int(w)}]\n")

    return facts_path, wc_path, facts_I_path


def _build_direct_wc_text(
    *,
    keys_in_file_order: list[str],
    key_to_weight: dict[str, int],
    objective: str,
) -> str:
    """Return a weak-constraint program for direct reification.

        - objective='sum': minimizes excluded weight (mirrors MUS's `:~ not mus(X) ... [C@1]`).
    - objective='lex': minimizes excluded weight with a lex tie-break by file order
      (mirrors MUS's `:~ not mus(X) ... [C@1,X]` behaviour).
    """
    obj = (objective or "sum").strip().lower()
    lines: list[str] = []
    if obj == "lex":
        # Lex: minimize excluded weights with a per-index cost component.
        # The tuple term creates lexicographic tie-breaking by index.
        for i, k in enumerate(keys_in_file_order, 1):
            w = int(key_to_weight.get(k, 0) or 0)
            if w <= 0:
                continue
            lines.append(f":~ not {k}. [{w}@1,{i}]")
    else:
        # Sum: minimize excluded weights.
        for k in keys_in_file_order:
            w = int(key_to_weight.get(k, 0) or 0)
            if w <= 0:
                continue
            lines.append(f":~ not {k}. [{w}@1]")
    return "\n".join(lines) + ("\n" if lines else "")


def _parse_weights_from_wc_file(wc_path: Path) -> dict[str, int]:
    weights: dict[str, int] = {}
    with wc_path.open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            # :~ ext_dep(1,2,s0). [-123]
            m = re.match(r"^:~\s+(.+?)\.\s*\[\s*-?(\d+)\s*\]", s)
            if not m:
                continue
            atom = _normalize_fact_line(m.group(1))
            if not atom:
                atom = _normalize_fact_line(m.group(1) + ".")
            if not atom:
                continue
            weights[atom] = int(m.group(2))
    return weights


def _run_clingo_file(program_path: Path, *, timeout_sec: int, opt_strategy: str | None) -> tuple[bool, str, list[int], int | None, float]:
    cmd = [
        "clingo",
        str(program_path),
        "--opt-mode=optN",
        "--outf=2",
        "-n",
        "1",
    ]
    if opt_strategy:
        cmd.append(f"--opt-strategy={opt_strategy}")

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_sec))
    except subprocess.TimeoutExpired:
        return True, "UNKNOWN", [], None, time.perf_counter() - start

    wall = time.perf_counter() - start
    stdout = proc.stdout or ""

    try:
        data = json.loads(stdout)
    except Exception:
        # clingo printed non-JSON (e.g. warnings). Treat as unknown.
        return False, "UNKNOWN", [], None, wall

    status = str(data.get("Result", "UNKNOWN"))
    calls = data.get("Call", []) or []
    costs: list[int] = []
    selected_count: int | None = None
    if calls:
        witnesses = calls[-1].get("Witnesses", []) or []
        if witnesses:
            w = witnesses[-1]
            costs = [int(x) for x in (w.get("Costs", []) or [])]
            vals = w.get("Value", []) or []
            selected_count = len(vals)

    return False, status, costs, selected_count, wall




def _normalize_base_program_for_analysis(base_text: str, *, n_nodes: int) -> str:
    # Match the normalization used by causalaba_mus.build_mus_program when a base dump is provided.
    # - Drop '#program main(...)' lines (base dumps can include program parts)
    # - Replace 'n_vars' with n_nodes-1
    # - Strip '#show' directives (keeps symbol table stable)
    # - Strip skeleton-reduction artifacts involving block_edge/2 (for comparability)
    out_lines: list[str] = []
    for raw in (base_text or "").splitlines():
        s = raw.strip()
        if s.startswith("#program main"):
            continue
        if raw.lstrip().startswith("#show"):
            continue
        # Drop materialized block_edge facts and any rule/constraint mentioning block_edge.
        if s.startswith("block_edge("):
            continue
        if "block_edge(" in raw:
            continue
        out_lines.append(raw.replace("n_vars", str(int(n_nodes)-1)))
    return "\n".join(out_lines)


def _emit_full_base_program_for_direct(
    *,
    n_nodes: int,
    facts_path: Path,
    out_path: Path,
) -> None:
    """Emit a full base program (encoding + specific rules) without skeleton reduction."""
    if out_path.exists():
        return
    specific_rules_file = facts_path.parent / f"{facts_path.stem}_specific.lp"
    if not specific_rules_file.exists():
        # Build indep/dep fact maps from the facts file
        indep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        dep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for line in facts_path.read_text().splitlines():
            parsed = _parse_ext_fact_line(line)
            if not parsed:
                continue
            fact_type, x, y, s_tuple = parsed
            target = indep_facts if fact_type == "ext_indep" else dep_facts
            target.setdefault((x, y), set()).add(s_tuple)

        from causalaba import compile_and_ground

        compile_and_ground(
            n_nodes,
            facts_location="",
            skeleton_rules_reduction=False,
            weak_constraints=False,
            indep_facts=indep_facts,
            dep_facts=dep_facts,
            opt_mode="optN",
            out_n=1,
            show=["arrow"],
            pre_grounding=False,
            ext_flag=False,
            prior_knowledge=None,
            max_path_length=None,
            max_conditioning_size=None,
            collider_tree_depth=None,
            cycle_length=None,
            dump_specific=str(specific_rules_file),
            threads=None,
            deadline=None,
            timing_recorder=None,
        )

    base_encoding_path = _REPO_ROOT / "encodings" / "causalaba.lp"
    base_text = base_encoding_path.read_text()
    specific_text = specific_rules_file.read_text()
    full_text = _normalize_base_program_for_analysis(f"{base_text}\n{specific_text}", n_nodes=n_nodes)
    out_path.write_text(full_text)
def main() -> None:
    _install_sigterm_handler()
    parser = argparse.ArgumentParser(description="Sweep WC opt-strategies/objectives on ABAPC encodings")
    parser.add_argument("--n-nodes", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2004)
    parser.add_argument("--edge-per-node", type=int, default=2)
    parser.add_argument("--graph-type", type=str, default="ER")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=900,
        help="Solve timeout in seconds (default: 900 = 15m). Applies to optimization only (not graph-eval).",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["synthetic", "bnlearn"],
        default="synthetic",
        help="Dataset source for the sweep.",
    )
    parser.add_argument(
        "--bnlearn-dataset",
        type=str,
        nargs="+",
        default=["asia"],
        help=(
            "BNLearn dataset name (e.g. asia, alarm, insurance). "
            "You can also pass a comma/space-separated list to run multiple datasets sequentially, "
            "e.g. 'cancer, earthquake, survey'."
        ),
    )
    parser.add_argument(
        "--bn-data-path",
        type=str,
        default="datasets",
        help="Path containing BNLearn .bif/.csv assets (as expected by load_bnlearn_data_dag).",
    )
    parser.add_argument(
        "--bn-standardise",
        action="store_true",
        help="Standardise BNLearn data (passed through to the BNLearn loader).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for PC CI tests.",
    )
    parser.add_argument(
        "--pc-indep-test",
        type=str,
        default="gsq",
        help="Independence test identifier for PC (passed through to causallearn).",
    )
    parser.add_argument(
        "--no-condset-weight",
        action="store_true",
        help="Disable conditioning-set-size weighting in initial_strength (i.e., ignore |S| when computing fact weights).",
    )
    parser.add_argument(
        "--graph-eval-timeout",
        type=int,
        default=300,
        help=(
            "DAG/CPDAG evaluation timeout in seconds (default: 300). This budget is separate from --timeout-sec. "
            "Set to 0 to skip graph evaluation."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help=(
            "Max clingo threads to use inside each run (default: 8). "
            "Use small values when running multiple sweeps concurrently to avoid OS OOM kills."
        ),
    )

    parser.add_argument(
        "--isolate-wc",
        action="store_true",
        help=(
            "Run ALL WC optimization (both reifications: mus + direct) in isolated subprocesses. "
            "This contains native clingo/gringo crashes (e.g., SIGSEGV) to a single run so the sweep can continue."
        ),
    )
    parser.add_argument("--out-dir", type=str, default="", help="Directory to write artifacts/results")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run by reusing an existing results directory and skipping already-recorded runs. "
            "If --out-dir is provided, it must point to an existing wc_sweep_*_*/ directory. "
            "If --out-dir is omitted, the latest matching results/wc_sweep_{n_nodes}_{seed}_*/ directory is used."
        ),
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="bb",
        help=(
            "Opt strategies to test (default: bb). Use ';' to separate multiple strategies "
            "to keep commas available for clingo tactics, e.g. 'bb,lin;usc;usc,1'. "
            "Alternatively, pass multiple --strategy flags to avoid shell issues."
        ),
    )
    parser.add_argument(
        "--strategy",
        dest="strategy_list",
        action="append",
        help="Add a single opt-strategy entry (can be given multiple times).",
    )
    parser.add_argument(
        "--opt-modes",
        type=str,
        default="optN",
        help=(
            "Comma-separated clingo --opt-mode values to run (default: optN). "
            "Use 'opt' for standard optimization; include both (e.g. 'opt,optN') to compare. "
            "Alternatively, pass multiple --opt-mode flags."
        ),
    )
    parser.add_argument(
        "--opt-mode",
        dest="opt_mode_list",
        action="append",
        help="Add a single opt-mode entry (can be given multiple times).",
    )
    parser.add_argument(
        "--encodings",
        type=str,
        default="inc,base",
        help="Comma-separated encodings: 'inc' (incremental base dump) and/or 'base' (compile-and-ground).",
    )
    parser.add_argument(
        "--reifications",
        type=str,
        default="mus",
        help="Comma-separated reifications: 'mus' (assumption layer with MUS) and/or 'direct' (direct WC on externals).",
    )
    parser.add_argument(
        "--objectives",
        type=str,
        default="lex",
        help="Comma-separated WC objectives: sum (C@1) and/or lex (C@1,X).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline removal runs (causalaba + causalaba_increm) and only run the WC optimizations.",
    )
    parser.add_argument(
        "--skip-wc",
        action="store_true",
        help="Skip WC optimization sweeps and only run baselines (causalaba + causalaba_increm).",
    )
    parser.add_argument(
        "--only-baseline",
        action="store_true",
        help="Alias for --skip-wc (run baselines only).",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        dest="only_baseline",
        help="Alias for --skip-wc (run baselines only).",
    )
    parser.add_argument(
        "--only-wc",
        action="store_true",
        help="Alias for --skip-baseline (run WC optimizations only).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="Number of repetitions to run for each configuration (default: 1).",
    )
    parser.add_argument(
        "--pct-wrong-facts",
        type=float,
        default=None,
        help="Target fraction of incorrect tests (0..1). If set, flips facts per rep to match the rate.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-form note stored at the top of 0.Config (use quotes if it contains spaces).",
    )
    parser.add_argument(
        "--assert-direct-mus-opt-eq",
        action="store_true",
        help=(
            "Exit non-zero if any (direct,mus) pair differs (status/incumbent/objective proxy). "
            "Prints the per-run wc_... .lp filenames to inspect in the results directory."
        ),
    )
    args = parser.parse_args()

    # Convenience aliases.
    if bool(getattr(args, "only_baseline", False)):
        args.skip_wc = True
        args.skip_baseline = False
    if bool(getattr(args, "only_wc", False)):
        args.skip_baseline = True

    if bool(getattr(args, "skip_wc", False)) and bool(getattr(args, "skip_baseline", False)):
        raise SystemExit("Nothing to run: both --skip-wc and --skip-baseline are set.")

    # Add timestamps to all logs (including those emitted from imported modules
    # like causalaba.py that log via the root logger).
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
    except TypeError:
        # Python < 3.8: no force= support.
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _parse_bnlearn_datasets(raw: str) -> list[str]:
        s = str(raw or "").strip()
        if not s:
            return []
        # Split on commas and/or whitespace.
        parts = [p.strip() for p in re.split(r"[\s,]+", s) if p.strip()]
        return parts

    # Allow `--bnlearn-dataset "a,b,c"` to run multiple datasets.
    if str(getattr(args, "source", "synthetic")) == "bnlearn":
        raw_bn = getattr(args, "bnlearn_dataset", "")
        raw_bn_s = " ".join(str(x) for x in raw_bn) if isinstance(raw_bn, list) else str(raw_bn or "")
        datasets = _parse_bnlearn_datasets(raw_bn_s)
        if not datasets:
            raise SystemExit("--bnlearn-dataset must be non-empty when --source bnlearn")
        if len(datasets) > 1:
            if bool(getattr(args, "resume", False)):
                raise SystemExit("Multiple BNLearn datasets are not supported with --resume; run datasets separately")
            if str(getattr(args, "out_dir", "") or "").strip():
                raise SystemExit(
                    "Multiple BNLearn datasets are not supported with --out-dir; omit --out-dir and let the script "
                    "create per-dataset results folders automatically"
                )

            base_argv = list(sys.argv[1:])

            def _replace_flag_value(argv: list[str], flag: str, value: str) -> list[str]:
                out: list[str] = []
                i = 0
                replaced = False
                while i < len(argv):
                    tok = argv[i]
                    if tok == flag:
                        out.append(flag)
                        out.append(str(value))
                        replaced = True
                        i += 1
                        # Drop ALL existing values for this flag until the next option.
                        while i < len(argv) and not str(argv[i]).startswith("--"):
                            i += 1
                        continue
                    out.append(tok)
                    i += 1
                if not replaced:
                    out.extend([flag, str(value)])
                return out

            for ds in datasets:
                child_argv = _replace_flag_value(base_argv, "--bnlearn-dataset", ds)
                logging.info(f"[bnlearn] running dataset='{ds}'")
                subprocess.run([sys.executable, str(Path(__file__).resolve())] + child_argv, check=True)
            return

        # Sanitize single dataset (e.g., trailing commas/spaces).
        args.bnlearn_dataset = datasets[0]

    n_nodes = int(args.n_nodes)
    if str(getattr(args, "source", "synthetic")) == "bnlearn":
        from utils.data_utils import load_bnlearn_data_dag

        _, B_true0 = load_bnlearn_data_dag(
            dataset_name=str(args.bnlearn_dataset),
            data_path=str(args.bn_data_path),
            sample_size=1,
            seed=int(args.seed),
            standardise=bool(args.bn_standardise),
            print_info=False,
        )
        try:
            n_nodes = int(B_true0.shape[0])
        except Exception:
            n_nodes = int(args.n_nodes)
    seed = int(args.seed)
    strength_S_weight = not bool(getattr(args, "no_condset_weight", False))
    timeout_sec = int(args.timeout_sec)
    if timeout_sec <= 0:
        raise SystemExit("--timeout-sec must be positive")
    try:
        graph_eval_timeout_sec = int(args.graph_eval_timeout)
    except Exception:
        graph_eval_timeout_sec = 30
    graph_eval_timeout_sec = max(0, int(graph_eval_timeout_sec))
    try:
        threads = int(args.threads)
    except Exception:
        threads = 1
    threads = max(1, int(threads))
    pct_wrong_facts = args.pct_wrong_facts
    if pct_wrong_facts is not None:
        if pct_wrong_facts < 0.0 or pct_wrong_facts > 1.0:
            raise SystemExit("--pct-wrong-facts must be between 0 and 1")
    def _parse_strategies(raw: str, extra: list[str] | None) -> list[str]:
        out: list[str] = []
        if extra:
            out.extend([s.strip() for s in extra if str(s).strip()])
        s = str(raw or "").strip()
        if s:
            parts = s.split(";") if ";" in s else s.split(",")
            out.extend([p.strip() for p in parts if p.strip()])
        return out

    strategies = _parse_strategies(args.strategies, args.strategy_list)
    if not strategies:
        raise SystemExit("No opt strategies provided")

    def _progress(tag: str, **fields: Any) -> None:
        """Structured progress logging for long sweeps."""
        try:
            bits = [f"{k}={v}" for k, v in fields.items()]
            msg = f"[progress] {tag}" + (" " + " ".join(bits) if bits else "")
            logging.info(msg)
        except Exception:
            # Never let logging break the sweep.
            try:
                logging.info(f"[progress] {tag}")
            except Exception:
                pass

    _progress(
        "run_start",
        n_nodes=n_nodes,
        seed=seed,
        reps=int(getattr(args, "reps", 1) or 1),
        timeout_sec=timeout_sec,
        graph_eval_timeout_sec=graph_eval_timeout_sec,
        threads=threads,
        strategies=";".join(str(s) for s in strategies),
        source=str(getattr(args, "source", "synthetic")),
        dataset=(str(args.bnlearn_dataset) if str(getattr(args, "source", "synthetic")) == "bnlearn" else None),
        strength_S_weight=bool(strength_S_weight),
    )

    def _parse_opt_modes(raw: str, extra: list[str] | None) -> list[str]:
        out: list[str] = []
        if extra:
            out.extend([s.strip() for s in extra if str(s).strip()])
        s = str(raw or "").strip()
        if s:
            out.extend([p.strip() for p in s.split(",") if p.strip()])

        normalized: list[str] = []
        for m in out:
            mm = (m or "").strip()
            if not mm:
                continue
            if mm.lower() == "optn":
                normalized.append("optN")
            else:
                normalized.append(mm.lower())

        allowed = {"opt", "optN"}
        bad = [m for m in normalized if m not in allowed]
        if bad:
            raise SystemExit(f"Unknown opt-mode(s) {bad}. Use 'opt' or 'optN'.")

        # De-dup while preserving order
        seen: set[str] = set()
        uniq: list[str] = []
        for m in normalized:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return uniq

    opt_modes = _parse_opt_modes(args.opt_modes, args.opt_mode_list)
    if not opt_modes:
        raise SystemExit("No opt-modes provided")

    encodings_raw = [s.strip().lower() for s in str(args.encodings).split(",") if s.strip()]
    encodings: list[str] = []
    for enc in encodings_raw:
        if enc in {"inc", "incremental", "increm"}:
            encodings.append("inc")
        elif enc in {"base", "baseline", "compile"}:
            encodings.append("base")
        else:
            raise SystemExit(f"Unknown encoding '{enc}'. Use 'inc' (incremental dump) or 'base' (compile-and-ground).")
    if not encodings:
        encodings = ["inc"]

    objectives_raw = [s.strip().lower() for s in str(args.objectives).split(",") if s.strip()]
    objectives: list[str] = []
    for obj in objectives_raw:
        if obj in {"sum", "c@1", "c1"}:
            objectives.append("sum")
        elif obj in {"lex", "c@1,x", "c@1x"}:
            objectives.append("lex")
        else:
            raise SystemExit(f"Unknown objective '{obj}'. Use 'sum' (C@1) or 'lex' (C@1,X).")
    if not objectives:
        objectives = ["sum"]

    reifications_raw = [s.strip().lower() for s in str(args.reifications).split(",") if s.strip()]
    reifications: list[str] = []
    for reif in reifications_raw:
        if reif in {"mus", "assumption", "assumptions"}:
            reifications.append("mus")
        elif reif in {"direct", "wc", "weakconstraints"}:
            reifications.append("direct")
        else:
            raise SystemExit(f"Unknown reification '{reif}'. Use 'mus' (assumption layer) or 'direct' (WC on externals).")
    if not reifications:
        reifications = ["mus"]

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        base_out_dir = Path(args.out_dir)
    else:
        if str(getattr(args, "source", "synthetic")) == "bnlearn":
            base_out_dir = Path("results") / (
                f"wc_sweep_bnlearn_{args.bnlearn_dataset}_n{n_nodes}_ss{int(args.sample_size)}_a{float(args.alpha)}_{seed}"
            )
        else:
            base_out_dir = Path("results") / f"wc_sweep_{n_nodes}_{seed}"

    if bool(getattr(args, "resume", False)):
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            # Auto-pick the latest timestamped folder.
            pattern = f"{base_out_dir.name}_*"
            candidates = [p for p in base_out_dir.parent.glob(pattern) if p.is_dir()]
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if not candidates:
                raise SystemExit(f"--resume requested but no matching directory found under {base_out_dir.parent} with pattern '{pattern}'")
            out_dir = candidates[0]
        if not out_dir.exists() or not out_dir.is_dir():
            raise SystemExit(f"--resume requested but out_dir does not exist or is not a directory: {out_dir}")
    else:
        out_dir = base_out_dir.parent / f"{base_out_dir.name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
    reps = max(1, int(args.reps))

    # Import locally so this script doesn't break `--help` if optional deps missing.
    from tests_mus import RandomPCSimConfig, build_random_pc_case
    from causalaba import CausalABA
    from causalaba_mus import CausalABA_WC

    from utils.graph_eval import graph_eval_from_accepted as _graph_eval_from_accepted

    baseline_results: list[BaselineResult] = []
    wc_results: list[StrategyResult] = []
    wrong_fact_counts: list[int] = []
    total_weight: Optional[int] = None

    def _load_resume_state() -> None:
        nonlocal baseline_results, wc_results, wrong_fact_counts, total_weight

        if not bool(getattr(args, "resume", False)):
            return

        resume_path_partial = out_dir / "summary.partial.json"
        resume_path_full = out_dir / "summary.json"
        resume_path = resume_path_partial if resume_path_partial.exists() else resume_path_full
        if not resume_path.exists():
            print(f"[resume] no summary file found in {out_dir}; starting fresh in-place")
            return

        try:
            d = json.loads(resume_path.read_text())
        except Exception as e:
            raise SystemExit(f"Failed to read resume state from {resume_path}: {e}")

        # Restore total_weight if present so summary writing works even when
        # the sweep has nothing new to do on resume.
        try:
            if d.get("total_weight") is not None:
                total_weight = int(d.get("total_weight"))
        except Exception:
            total_weight = None

        # Basic config sanity checks to avoid mixing incompatible runs.
        try:
            if int(d.get("n_nodes")) != int(n_nodes):
                raise SystemExit(f"Resume mismatch: n_nodes in {resume_path} is {d.get('n_nodes')} but current is {n_nodes}")
        except Exception:
            pass
        try:
            if int(d.get("seed")) != int(seed):
                raise SystemExit(f"Resume mismatch: seed in {resume_path} is {d.get('seed')} but current is {seed}")
        except Exception:
            pass
        # pct_wrong_facts must match if present in resume file.
        if d.get("pct_wrong_facts") is not None and pct_wrong_facts is not None:
            try:
                if float(d.get("pct_wrong_facts")) != float(pct_wrong_facts):
                    raise SystemExit(
                        f"Resume mismatch: pct_wrong_facts in {resume_path} is {d.get('pct_wrong_facts')} but current is {pct_wrong_facts}"
                    )
            except Exception:
                pass

        def _baseline_from_json(r: dict[str, Any]) -> BaselineResult:
            return BaselineResult(
                rep=int(r.get("rep", 0) or 0),
                solver=str(r.get("solver", "")),
                opt_mode=str(r.get("opt_mode", "optN") or "optN"),
                wall_time_sec=float(r.get("wall_time_sec", 0.0) or 0.0),
                method_time_sec=(
                    float(r.get("method_time_sec")) if r.get("method_time_sec") is not None else None
                ),
                build_time_sec=(float(r.get("build_time_sec")) if r.get("build_time_sec") is not None else None),
                solve_time_sec=(float(r.get("solve_time_sec")) if r.get("solve_time_sec") is not None else None),
                eval_time_sec=(float(r.get("eval_time_sec")) if r.get("eval_time_sec") is not None else None),
                timed_out=bool(r.get("timed_out", False)),
                graph_eval_timed_out=(
                    bool(r.get("graph_eval_timed_out")) if r.get("graph_eval_timed_out") is not None else None
                ),
                graph_eval_cache_hit=(
                    bool(r.get("graph_eval_cache_hit")) if r.get("graph_eval_cache_hit") is not None else None
                ),
                removed=int(r.get("removed", 0) or 0),
                models_after=int(r.get("models_after", 0) or 0),
                base_program_path=(str(r.get("base_program_path")) if r.get("base_program_path") is not None else None),
                accepted_weight=(float(r.get("accepted_weight")) if r.get("accepted_weight") is not None else None),
                n_tests_total=int(r.get("n_tests_total", 0) or 0),
                n_tests_true=int(r.get("n_tests_true", 0) or 0),
                n_tests_accepted=int(r.get("n_tests_accepted", 0) or 0),
                n_tests_accepted_true=int(r.get("n_tests_accepted_true", 0) or 0),
                weight_total=int(r.get("weight_total", 0) or 0),
                weight_true=int(r.get("weight_true", 0) or 0),
                weight_accepted_true=int(r.get("weight_accepted_true", 0) or 0),
                accepted_fact_f1=(float(r.get("accepted_fact_f1")) if r.get("accepted_fact_f1") is not None else None),
                n_dags_compat=(int(r.get("n_dags_compat")) if r.get("n_dags_compat") is not None else None),
                true_dag_in_compat=(int(r.get("true_dag_in_compat")) if r.get("true_dag_in_compat") is not None else None),
                n_cpdags_compat=(int(r.get("n_cpdags_compat")) if r.get("n_cpdags_compat") is not None else None),
                true_cpdag_in_compat=(
                    int(r.get("true_cpdag_in_compat")) if r.get("true_cpdag_in_compat") is not None else None
                ),
                shd_avg=(float(r.get("shd_avg")) if r.get("shd_avg") is not None else None),
                f1_avg=(float(r.get("f1_avg")) if r.get("f1_avg") is not None else None),
                adjacency_f1_avg=(float(r.get("adjacency_f1_avg")) if r.get("adjacency_f1_avg") is not None else None),
                arrowhead_f1_avg=(float(r.get("arrowhead_f1_avg")) if r.get("arrowhead_f1_avg") is not None else None),
                shd_best=(float(r.get("shd_best")) if r.get("shd_best") is not None else None),
                shd_worst=(float(r.get("shd_worst")) if r.get("shd_worst") is not None else None),
                f1_best=(float(r.get("f1_best")) if r.get("f1_best") is not None else None),
                f1_worst=(float(r.get("f1_worst")) if r.get("f1_worst") is not None else None),
                adjacency_f1_best=(
                    float(r.get("adjacency_f1_best")) if r.get("adjacency_f1_best") is not None else None
                ),
                adjacency_f1_worst=(
                    float(r.get("adjacency_f1_worst")) if r.get("adjacency_f1_worst") is not None else None
                ),
                arrowhead_f1_best=(
                    float(r.get("arrowhead_f1_best")) if r.get("arrowhead_f1_best") is not None else None
                ),
                arrowhead_f1_worst=(
                    float(r.get("arrowhead_f1_worst")) if r.get("arrowhead_f1_worst") is not None else None
                ),
                cpdag_shd_avg=(float(r.get("cpdag_shd_avg")) if r.get("cpdag_shd_avg") is not None else None),
                cpdag_f1_avg=(float(r.get("cpdag_f1_avg")) if r.get("cpdag_f1_avg") is not None else None),
                cpdag_adjacency_f1_avg=(
                    float(r.get("cpdag_adjacency_f1_avg")) if r.get("cpdag_adjacency_f1_avg") is not None else None
                ),
                cpdag_arrowhead_f1_avg=(
                    float(r.get("cpdag_arrowhead_f1_avg")) if r.get("cpdag_arrowhead_f1_avg") is not None else None
                ),
                cpdag_shd_best=(float(r.get("cpdag_shd_best")) if r.get("cpdag_shd_best") is not None else None),
                cpdag_shd_worst=(float(r.get("cpdag_shd_worst")) if r.get("cpdag_shd_worst") is not None else None),
                cpdag_f1_best=(float(r.get("cpdag_f1_best")) if r.get("cpdag_f1_best") is not None else None),
                cpdag_f1_worst=(float(r.get("cpdag_f1_worst")) if r.get("cpdag_f1_worst") is not None else None),
                cpdag_adjacency_f1_best=(
                    float(r.get("cpdag_adjacency_f1_best")) if r.get("cpdag_adjacency_f1_best") is not None else None
                ),
                cpdag_adjacency_f1_worst=(
                    float(r.get("cpdag_adjacency_f1_worst")) if r.get("cpdag_adjacency_f1_worst") is not None else None
                ),
                cpdag_arrowhead_f1_best=(
                    float(r.get("cpdag_arrowhead_f1_best")) if r.get("cpdag_arrowhead_f1_best") is not None else None
                ),
                cpdag_arrowhead_f1_worst=(
                    float(r.get("cpdag_arrowhead_f1_worst")) if r.get("cpdag_arrowhead_f1_worst") is not None else None
                ),
            )

        def _wc_from_json(r: dict[str, Any]) -> StrategyResult:
            return StrategyResult(
                rep=int(r.get("rep", 0) or 0),
                encoding=str(r.get("encoding", "")),
                objective=str(r.get("objective", "")),
                opt_strategy=str(r.get("opt_strategy", "")),
                opt_mode=str(r.get("opt_mode", "optN") or "optN"),
                reification=str(r.get("reification", "")),
                timed_out=bool(r.get("timed_out", False)),
                graph_eval_timed_out=(
                    bool(r.get("graph_eval_timed_out")) if r.get("graph_eval_timed_out") is not None else None
                ),
                graph_eval_cache_hit=(
                    bool(r.get("graph_eval_cache_hit")) if r.get("graph_eval_cache_hit") is not None else None
                ),
                wall_time_sec=float(r.get("wall_time_sec", 0.0) or 0.0),
                method_time_sec=(
                    float(r.get("method_time_sec")) if r.get("method_time_sec") is not None else None
                ),
                status=str(r.get("status", "")),
                objective_costs=[int(x) for x in (r.get("objective_costs", []) or [])],
                cut_weight=(int(r.get("cut_weight")) if r.get("cut_weight") is not None else None),
                selected_count=(int(r.get("selected_count")) if r.get("selected_count") is not None else None),
                build_time_sec=(float(r.get("build_time_sec")) if r.get("build_time_sec") is not None else None),
                solve_time_sec=(float(r.get("solve_time_sec")) if r.get("solve_time_sec") is not None else None),
                eval_time_sec=(float(r.get("eval_time_sec")) if r.get("eval_time_sec") is not None else None),
                accepted_weight=(float(r.get("accepted_weight")) if r.get("accepted_weight") is not None else None),
                n_tests_total=int(r.get("n_tests_total", 0) or 0),
                n_tests_true=int(r.get("n_tests_true", 0) or 0),
                n_tests_accepted=int(r.get("n_tests_accepted", 0) or 0),
                n_tests_accepted_true=int(r.get("n_tests_accepted_true", 0) or 0),
                weight_total=int(r.get("weight_total", 0) or 0),
                weight_true=int(r.get("weight_true", 0) or 0),
                weight_accepted_true=int(r.get("weight_accepted_true", 0) or 0),
                accepted_fact_f1=(float(r.get("accepted_fact_f1")) if r.get("accepted_fact_f1") is not None else None),
                n_dags_compat=(int(r.get("n_dags_compat")) if r.get("n_dags_compat") is not None else None),
                true_dag_in_compat=(int(r.get("true_dag_in_compat")) if r.get("true_dag_in_compat") is not None else None),
                n_cpdags_compat=(int(r.get("n_cpdags_compat")) if r.get("n_cpdags_compat") is not None else None),
                true_cpdag_in_compat=(
                    int(r.get("true_cpdag_in_compat")) if r.get("true_cpdag_in_compat") is not None else None
                ),
                shd_avg=(float(r.get("shd_avg")) if r.get("shd_avg") is not None else None),
                f1_avg=(float(r.get("f1_avg")) if r.get("f1_avg") is not None else None),
                adjacency_f1_avg=(float(r.get("adjacency_f1_avg")) if r.get("adjacency_f1_avg") is not None else None),
                arrowhead_f1_avg=(float(r.get("arrowhead_f1_avg")) if r.get("arrowhead_f1_avg") is not None else None),
                shd_best=(float(r.get("shd_best")) if r.get("shd_best") is not None else None),
                shd_worst=(float(r.get("shd_worst")) if r.get("shd_worst") is not None else None),
                f1_best=(float(r.get("f1_best")) if r.get("f1_best") is not None else None),
                f1_worst=(float(r.get("f1_worst")) if r.get("f1_worst") is not None else None),
                adjacency_f1_best=(
                    float(r.get("adjacency_f1_best")) if r.get("adjacency_f1_best") is not None else None
                ),
                adjacency_f1_worst=(
                    float(r.get("adjacency_f1_worst")) if r.get("adjacency_f1_worst") is not None else None
                ),
                arrowhead_f1_best=(
                    float(r.get("arrowhead_f1_best")) if r.get("arrowhead_f1_best") is not None else None
                ),
                arrowhead_f1_worst=(
                    float(r.get("arrowhead_f1_worst")) if r.get("arrowhead_f1_worst") is not None else None
                ),
                cpdag_shd_avg=(float(r.get("cpdag_shd_avg")) if r.get("cpdag_shd_avg") is not None else None),
                cpdag_f1_avg=(float(r.get("cpdag_f1_avg")) if r.get("cpdag_f1_avg") is not None else None),
                cpdag_adjacency_f1_avg=(
                    float(r.get("cpdag_adjacency_f1_avg")) if r.get("cpdag_adjacency_f1_avg") is not None else None
                ),
                cpdag_arrowhead_f1_avg=(
                    float(r.get("cpdag_arrowhead_f1_avg")) if r.get("cpdag_arrowhead_f1_avg") is not None else None
                ),
                cpdag_shd_best=(float(r.get("cpdag_shd_best")) if r.get("cpdag_shd_best") is not None else None),
                cpdag_shd_worst=(float(r.get("cpdag_shd_worst")) if r.get("cpdag_shd_worst") is not None else None),
                cpdag_f1_best=(float(r.get("cpdag_f1_best")) if r.get("cpdag_f1_best") is not None else None),
                cpdag_f1_worst=(float(r.get("cpdag_f1_worst")) if r.get("cpdag_f1_worst") is not None else None),
                cpdag_adjacency_f1_best=(
                    float(r.get("cpdag_adjacency_f1_best")) if r.get("cpdag_adjacency_f1_best") is not None else None
                ),
                cpdag_adjacency_f1_worst=(
                    float(r.get("cpdag_adjacency_f1_worst")) if r.get("cpdag_adjacency_f1_worst") is not None else None
                ),
                cpdag_arrowhead_f1_best=(
                    float(r.get("cpdag_arrowhead_f1_best")) if r.get("cpdag_arrowhead_f1_best") is not None else None
                ),
                cpdag_arrowhead_f1_worst=(
                    float(r.get("cpdag_arrowhead_f1_worst")) if r.get("cpdag_arrowhead_f1_worst") is not None else None
                ),
            )

        baseline_results = [_baseline_from_json(r) for r in (d.get("baseline_results", []) or []) if isinstance(r, dict)]
        wc_results = [_wc_from_json(r) for r in (d.get("wc_results", []) or []) if isinstance(r, dict)]

        wfc = d.get("wrong_fact_counts", [])
        if isinstance(wfc, list):
            out_wfc: list[int] = []
            for x in wfc:
                try:
                    out_wfc.append(int(x))
                except Exception:
                    continue
            wrong_fact_counts = out_wfc

        print(
            f"[resume] loaded {len(baseline_results)} baseline rows + {len(wc_results)} wc rows from {resume_path.name}"
        )

    _load_resume_state()

    done_baseline: set[tuple[int, str, str]] = set(
        (
            int(r.rep),
            str(r.solver),
            str(getattr(r, "opt_mode", "optN") or "optN"),
        )
        for r in (baseline_results or [])
    )
    done_wc: set[tuple[int, str, str, str, str, str]] = set(
        (
            int(r.rep),
            str(r.encoding),
            str(r.objective),
            str(r.opt_strategy),
            str(getattr(r, "opt_mode", "optN") or "optN"),
            str(r.reification),
        )
        for r in (wc_results or [])
    )

    # Persist a small run manifest at the top of the folder.
    cfg_path = out_dir / "0.Config"
    notes_raw = str(getattr(args, "notes", "") or "").strip()
    # Keep 0.Config line-based for grepping/parsing.
    notes_sanitized = notes_raw.replace("\n", "\\n")
    cfg_lines: list[str] = []
    if notes_sanitized:
        cfg_lines.append(f"notes={notes_sanitized}")
    if not (bool(getattr(args, "resume", False)) and cfg_path.exists()):
        cfg_path.write_text(
            "\n".join(
                cfg_lines
                + [
                    f"timestamp={ts}",
                    f"argv={' '.join(sys.argv)}",
                    f"out_dir={out_dir}",
                    f"source={str(getattr(args, 'source', 'synthetic'))}",
                    f"bnlearn_dataset={str(getattr(args, 'bnlearn_dataset', ''))}",
                    f"bn_data_path={str(getattr(args, 'bn_data_path', ''))}",
                    f"n_nodes={n_nodes}",
                    f"seed={seed}",
                    f"reps={reps}",
                    f"timeout_sec={timeout_sec}",
                    f"graph_eval_timeout_sec={graph_eval_timeout_sec}",
                    f"pct_wrong_facts={pct_wrong_facts}",
                    f"alpha={float(getattr(args, 'alpha', 0.05))}",
                    f"pc_indep_test={str(getattr(args, 'pc_indep_test', ''))}",
                    f"strength_S_weight={bool(strength_S_weight)}",
                    f"graph_type={args.graph_type}",
                    f"edge_per_node={args.edge_per_node}",
                    f"sample_size={args.sample_size}",
                    f"strategies={strategies}",
                    f"opt_modes={opt_modes}",
                    f"encodings={encodings}",
                    f"objectives={objectives}",
                    f"reifications={reifications}",
                    "",
                ]
            )
        )

    # Write a rolling snapshot so `wc_sweep_report.py --results-dir ...` can show
    # real partial summaries while the sweep is still running.
    partial_json = out_dir / "summary.partial.json"

    def _write_summary_snapshot(*, path: Path, is_partial: bool, total_weight_current: Optional[int]) -> None:
        def _baseline_to_json(r: BaselineResult) -> dict[str, Any]:
            return {
                "rep": r.rep,
                "solver": r.solver,
                "opt_mode": r.opt_mode,
                "wall_time_sec": r.wall_time_sec,
                "method_time_sec": r.method_time_sec,
                "build_time_sec": r.build_time_sec,
                "solve_time_sec": r.solve_time_sec,
                "eval_time_sec": r.eval_time_sec,
                "timed_out": r.timed_out,
                "graph_eval_timed_out": r.graph_eval_timed_out,
                "graph_eval_cache_hit": r.graph_eval_cache_hit,
                "removed": r.removed,
                "models_after": r.models_after,
                "base_program_path": r.base_program_path,
                "accepted_weight": r.accepted_weight,
                "n_tests_total": r.n_tests_total,
                "n_tests_true": r.n_tests_true,
                "n_tests_accepted": r.n_tests_accepted,
                "n_tests_accepted_true": r.n_tests_accepted_true,
                "weight_total": r.weight_total,
                "weight_true": r.weight_true,
                "weight_accepted": int(r.accepted_weight) if r.accepted_weight is not None else None,
                "weight_accepted_true": r.weight_accepted_true,
                "accepted_fact_f1": r.accepted_fact_f1,
                "n_dags_compat": r.n_dags_compat,
                "true_dag_in_compat": r.true_dag_in_compat,
                "n_cpdags_compat": r.n_cpdags_compat,
                "true_cpdag_in_compat": r.true_cpdag_in_compat,
                "shd_avg": r.shd_avg,
                "f1_avg": r.f1_avg,
                "adjacency_f1_avg": r.adjacency_f1_avg,
                "arrowhead_f1_avg": r.arrowhead_f1_avg,
                "shd_best": r.shd_best,
                "shd_worst": r.shd_worst,
                "f1_best": r.f1_best,
                "f1_worst": r.f1_worst,
                "adjacency_f1_best": r.adjacency_f1_best,
                "adjacency_f1_worst": r.adjacency_f1_worst,
                "arrowhead_f1_best": r.arrowhead_f1_best,
                "arrowhead_f1_worst": r.arrowhead_f1_worst,
                "cpdag_shd_avg": r.cpdag_shd_avg,
                "cpdag_f1_avg": r.cpdag_f1_avg,
                "cpdag_adjacency_f1_avg": r.cpdag_adjacency_f1_avg,
                "cpdag_arrowhead_f1_avg": r.cpdag_arrowhead_f1_avg,
                "cpdag_shd_best": r.cpdag_shd_best,
                "cpdag_shd_worst": r.cpdag_shd_worst,
                "cpdag_f1_best": r.cpdag_f1_best,
                "cpdag_f1_worst": r.cpdag_f1_worst,
                "cpdag_adjacency_f1_best": r.cpdag_adjacency_f1_best,
                "cpdag_adjacency_f1_worst": r.cpdag_adjacency_f1_worst,
                "cpdag_arrowhead_f1_best": r.cpdag_arrowhead_f1_best,
                "cpdag_arrowhead_f1_worst": r.cpdag_arrowhead_f1_worst,
            }

        def _wc_to_json(r: StrategyResult) -> dict[str, Any]:
            # Intentionally omit objective_costs and cut_weight from JSON.
            return {
                "rep": r.rep,
                "encoding": r.encoding,
                "objective": r.objective,
                "opt_strategy": r.opt_strategy,
                "opt_mode": r.opt_mode,
                "reification": r.reification,
                "timed_out": r.timed_out,
                "graph_eval_timed_out": r.graph_eval_timed_out,
                "graph_eval_cache_hit": r.graph_eval_cache_hit,
                "wall_time_sec": r.wall_time_sec,
                "method_time_sec": r.method_time_sec,
                "status": r.status,
                "selected_count": r.selected_count,
                "build_time_sec": r.build_time_sec,
                "solve_time_sec": r.solve_time_sec,
                "eval_time_sec": r.eval_time_sec,
                "accepted_weight": r.accepted_weight,
                "n_tests_total": r.n_tests_total,
                "n_tests_true": r.n_tests_true,
                "n_tests_accepted": r.n_tests_accepted,
                "n_tests_accepted_true": r.n_tests_accepted_true,
                "weight_total": r.weight_total,
                "weight_true": r.weight_true,
                "weight_accepted": int(r.accepted_weight) if r.accepted_weight is not None else None,
                "weight_accepted_true": r.weight_accepted_true,
                "accepted_fact_f1": r.accepted_fact_f1,
                "n_dags_compat": r.n_dags_compat,
                "true_dag_in_compat": r.true_dag_in_compat,
                "n_cpdags_compat": r.n_cpdags_compat,
                "true_cpdag_in_compat": r.true_cpdag_in_compat,
                "shd_avg": r.shd_avg,
                "f1_avg": r.f1_avg,
                "adjacency_f1_avg": r.adjacency_f1_avg,
                "arrowhead_f1_avg": r.arrowhead_f1_avg,
                "shd_best": r.shd_best,
                "shd_worst": r.shd_worst,
                "f1_best": r.f1_best,
                "f1_worst": r.f1_worst,
                "adjacency_f1_best": r.adjacency_f1_best,
                "adjacency_f1_worst": r.adjacency_f1_worst,
                "arrowhead_f1_best": r.arrowhead_f1_best,
                "arrowhead_f1_worst": r.arrowhead_f1_worst,
                "cpdag_shd_avg": r.cpdag_shd_avg,
                "cpdag_f1_avg": r.cpdag_f1_avg,
                "cpdag_adjacency_f1_avg": r.cpdag_adjacency_f1_avg,
                "cpdag_arrowhead_f1_avg": r.cpdag_arrowhead_f1_avg,
                "cpdag_shd_best": r.cpdag_shd_best,
                "cpdag_shd_worst": r.cpdag_shd_worst,
                "cpdag_f1_best": r.cpdag_f1_best,
                "cpdag_f1_worst": r.cpdag_f1_worst,
                "cpdag_adjacency_f1_best": r.cpdag_adjacency_f1_best,
                "cpdag_adjacency_f1_worst": r.cpdag_adjacency_f1_worst,
                "cpdag_arrowhead_f1_best": r.cpdag_arrowhead_f1_best,
                "cpdag_arrowhead_f1_worst": r.cpdag_arrowhead_f1_worst,
            }

        reps_completed = 0
        try:
            reps_completed = max(
                [int(r.rep) for r in baseline_results] + [int(r.rep) for r in wc_results] + [0]
            )
        except Exception:
            reps_completed = 0

        payload: dict[str, Any] = {
            "n_nodes": n_nodes,
            "seed": seed,
            "timeout_sec": timeout_sec,
            "pct_wrong_facts": pct_wrong_facts,
            "wrong_fact_counts": wrong_fact_counts,
            "strategies": strategies,
            "opt_modes": opt_modes,
            "encodings": encodings,
            "objectives": objectives,
            "reifications": reifications,
            "reps": reps,
            "total_weight": total_weight_current,
            "baseline_results": [_baseline_to_json(r) for r in baseline_results],
            "wc_results": [_wc_to_json(r) for r in wc_results],
            "is_partial": bool(is_partial),
            "reps_planned": int(reps),
            "reps_completed": int(reps_completed),
        }

        try:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except Exception:
            pass

    # If the parent gets SIGTERM (exit 143), try to flush a final snapshot so
    # a subsequent `--resume` can continue without losing progress.
    global _SIGTERM_SNAPSHOT_HOOK  # noqa: PLW0603
    _SIGTERM_SNAPSHOT_HOOK = lambda: _write_summary_snapshot(
        path=partial_json,
        is_partial=True,
        total_weight_current=total_weight,
    )

    for rep in range(1, reps + 1):
        pending_baseline_runs: list[tuple[str, str]] = []  # (solver, opt_mode)
        if not args.skip_baseline:
            for _om in opt_modes:
                if (rep, "causalaba_increm", _om) not in done_baseline:
                    pending_baseline_runs.append(("causalaba_increm", _om))
                if (rep, "causalaba", _om) not in done_baseline:
                    pending_baseline_runs.append(("causalaba", _om))

        pending_wc: list[tuple[str, str, str, str, str]] = []
        if not bool(getattr(args, "skip_wc", False)):
            for enc in encodings:
                for reif in reifications:
                    for obj in objectives:
                        for opt in strategies:
                            for opt_mode in opt_modes:
                                if (rep, enc, obj, opt, opt_mode, reif) not in done_wc:
                                    pending_wc.append((enc, reif, obj, opt, opt_mode))

        baseline_total = int(len(pending_baseline_runs))
        wc_total = int(len(pending_wc))
        baseline_i = 0
        wc_i = 0
        _progress(
            "rep_plan",
            rep=rep,
            reps=reps,
            baseline_pending=baseline_total,
            wc_pending=wc_total,
        )

        # Per-rep diagnostic store for base/inc mismatch checks.
        baseline_diag: dict[tuple[str, str], dict[str, Any]] = {}

        if not pending_baseline_runs and not pending_wc:
            # Fully done: skip all work for this repetition.
            # If we somehow have results but no wrong_fact_counts entry yet, regenerate
            # just enough to fill it deterministically.
            if bool(getattr(args, "resume", False)) and len(wrong_fact_counts) < rep:
                rep_seed = seed + rep - 1
                if str(getattr(args, "source", "synthetic")) == "bnlearn":
                    from utils.data_utils import load_bnlearn_data_dag

                    X, B_true = load_bnlearn_data_dag(
                        dataset_name=str(args.bnlearn_dataset),
                        data_path=str(args.bn_data_path),
                        sample_size=int(args.sample_size),
                        seed=int(rep_seed),
                        standardise=bool(args.bn_standardise),
                        print_info=False,
                    )
                    case = _build_pc_case_from_data(
                        data=X,
                        B_true=B_true,
                        alpha=float(args.alpha),
                        indep_test=str(args.pc_indep_test),
                        uc_rule=5,
                        uc_priority=2,
                        stable=True,
                        strength_S_weight=bool(strength_S_weight),
                    )
                else:
                    cfg = RandomPCSimConfig(
                        n_nodes=n_nodes,
                        alpha=float(args.alpha),
                        graph_type=str(args.graph_type),
                        edge_per_node=int(args.edge_per_node),
                        seed=rep_seed,
                        sample_size=int(args.sample_size),
                        uc_rule=5,
                        uc_priority=2,
                        stable=True,
                        strength_S_weight=bool(strength_S_weight),
                    )
                    case = build_random_pc_case(cfg)
                facts_tmp = list(case["facts"])
                if pct_wrong_facts is not None:
                    rng = random.Random(rep_seed)
                    _facts_tmp2, wrong_count_tmp = _apply_pct_wrong_facts(facts_tmp, pct_wrong_facts, rng=rng)
                else:
                    wrong_count_tmp = int(case.get("count_wrong", 0))
                if len(wrong_fact_counts) < rep:
                    wrong_fact_counts.extend([0] * (rep - len(wrong_fact_counts)))
                wrong_fact_counts[rep - 1] = int(wrong_count_tmp)
            continue

        # Generate a new case for each repetition with incremented seed
        rep_seed = seed + rep - 1
        if str(getattr(args, "source", "synthetic")) == "bnlearn":
            from utils.data_utils import load_bnlearn_data_dag

            X, B_true = load_bnlearn_data_dag(
                dataset_name=str(args.bnlearn_dataset),
                data_path=str(args.bn_data_path),
                sample_size=int(args.sample_size),
                seed=int(rep_seed),
                standardise=bool(args.bn_standardise),
                print_info=False,
            )
            case = _build_pc_case_from_data(
                data=X,
                B_true=B_true,
                alpha=float(args.alpha),
                indep_test=str(args.pc_indep_test),
                uc_rule=5,
                uc_priority=2,
                stable=True,
                strength_S_weight=bool(strength_S_weight),
            )
        else:
            cfg = RandomPCSimConfig(
                n_nodes=n_nodes,
                alpha=float(args.alpha),
                graph_type=str(args.graph_type),
                edge_per_node=int(args.edge_per_node),
                seed=rep_seed,
                sample_size=int(args.sample_size),
                uc_rule=5,
                uc_priority=2,
                stable=True,
                strength_S_weight=bool(strength_S_weight),
            )
            case = build_random_pc_case(cfg)
        G_true1 = case.get("G_true1") or case.get("G_true")

        # Graph-eval is often the bottleneck and is deterministic given the
        # accepted set + budget. Cache results within a rep so identical
        # accepted sets across methods (e.g., baseline vs incremental) don't
        # repeat the same expensive enumeration.
        graph_eval_cache: dict[frozenset[str], tuple[Any, ...]] = {}
        # Store wall time spent computing the cached graph-eval. Cache hits
        # reuse this as attributed eval time so rows stay comparable.
        graph_eval_cache_wall: dict[frozenset[str], float] = {}

        def _graph_eval_from_accepted_wall(
            *,
            accepted_keys: set[str],
            timeout_sec: float,
        ) -> tuple[Any, ...]:
            """Graph-eval with a *hard wall-clock* timeout.

            Rationale: clingo's Python API timeout controls solving, but grounding
            can still exceed budgets because it runs in native code.

                        - Caches results within a rep so identical accepted sets across methods
                            don't repeat the same expensive enumeration.
            """

            cache_key = frozenset(accepted_keys or set())
            if cache_key in graph_eval_cache:
                _progress(
                    "graph_eval_cache_hit",
                    rep=rep,
                    accepted=int(len(accepted_keys or [])),
                )
                cached = graph_eval_cache[cache_key]
                return (*cached, True, float(graph_eval_cache_wall.get(cache_key, 0.0)))

            # Fast-path: disabled or no budget.
            try:
                if float(timeout_sec) <= 0.0:
                    out29 = (
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        True,
                    )
                    return (*out29, False, 0.0)
            except Exception:
                out29 = (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    True,
                )
                return (*out29, False, 0.0)

            # Hard wall-time budget: keep grace small so timeouts are tight.
            wall_timeout = max(0.1, float(timeout_sec) + 0.2)

            # Try to use fork when available (faster, and allows nested targets on Linux).
            try:
                ctx = multiprocessing.get_context("fork")
            except Exception:
                ctx = multiprocessing

            # Debug/diagnostic controls for graph-eval progress streaming.
            # - DAG markers are emitted per compatible DAG completion.
            # - Partial snapshots are emitted every N DAGs to preserve work on wall-timeout.
            ge_dag_markers = str(os.environ.get("WC_SWEEP_GRAPH_EVAL_DAG_MARKERS") or "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
            try:
                ge_snapshot_every = int(os.environ.get("WC_SWEEP_GRAPH_EVAL_SNAPSHOT_EVERY") or 25)
            except Exception:
                ge_snapshot_every = 25
            ge_snapshot_every = max(1, int(ge_snapshot_every))

            def _empty_out29(*, timed_out_flag: bool = True) -> tuple[Any, ...]:
                return (*((None,) * 28), bool(timed_out_flag))

            def _force_timed_out29(out: Any) -> tuple[Any, ...]:
                if isinstance(out, (list, tuple)) and len(out) == 29:
                    out_l = list(out)
                    out_l[-1] = True
                    return tuple(out_l)
                return _empty_out29(timed_out_flag=True)

            q: Any = ctx.Queue()

            def _worker() -> None:
                try:
                    def _emit_progress(event: str, payload: dict[str, Any]) -> None:
                        try:
                            q.put(("progress", (str(event), dict(payload or {}))))
                        except Exception:
                            pass

                    out = _graph_eval_from_accepted(
                        n_nodes=n_nodes,
                        G_true1=G_true1,
                        keys_in_file_order=keys_in_file_order,
                        accepted_keys=accepted_keys,
                        timeout_sec=float(timeout_sec),
                        threads=threads,
                        cache={},
                        include_extrema=True,
                        progress_cb=_emit_progress,
                        progress_snapshot_every=int(ge_snapshot_every),
                    )
                    q.put(("ok", out))
                except BaseException as e:
                    q.put(("err", (repr(e), traceback.format_exc())))

            _progress(
                "graph_eval_start",
                rep=rep,
                accepted=int(len(accepted_keys or [])),
                timeout_sec=round(float(timeout_sec), 3),
                wall_timeout_sec=round(float(wall_timeout), 3),
            )
            t0_ge = time.perf_counter()
            proc = ctx.Process(target=_worker)
            proc.daemon = True
            proc.start()
            latest_partial: tuple[Any, ...] | None = None
            terminal: tuple[str, Any] | None = None

            def _handle_queue_msg(st: Any, payload: Any) -> tuple[str, Any] | None:
                nonlocal latest_partial
                if st == "progress":
                    try:
                        event, info = payload
                    except Exception:
                        return None
                    if str(event) == "graph_eval_dag_done":
                        if ge_dag_markers:
                            try:
                                _progress(
                                    "graph_eval_dag_done",
                                    rep=rep,
                                    accepted=int(len(accepted_keys or [])),
                                    nD=int(info.get("n_dags", 0)),
                                    nCPD=int(info.get("n_cpdags", 0)),
                                )
                            except Exception:
                                pass
                        return None
                    if str(event) == "graph_eval_partial":
                        out_p = info.get("out") if isinstance(info, dict) else None
                        if isinstance(out_p, (list, tuple)) and len(out_p) == 29:
                            latest_partial = tuple(out_p)
                        return None
                    return None
                return (str(st), payload)

            deadline_wall = t0_ge + float(wall_timeout)
            while True:
                if terminal is not None:
                    break
                rem = deadline_wall - time.perf_counter()
                if rem <= 0.0:
                    break
                try:
                    st, payload = q.get(timeout=min(0.25, max(0.01, rem)))
                    maybe_terminal = _handle_queue_msg(st, payload)
                    if maybe_terminal is not None:
                        terminal = maybe_terminal
                        break
                except Exception:
                    if not proc.is_alive():
                        break

            wall_ge = time.perf_counter() - t0_ge

            def _drain_queue() -> None:
                nonlocal terminal
                while True:
                    try:
                        st, payload = q.get_nowait()
                    except Exception:
                        break
                    maybe_terminal = _handle_queue_msg(st, payload)
                    if maybe_terminal is not None and terminal is None:
                        terminal = maybe_terminal

            _drain_queue()

            if proc.is_alive():
                _progress(
                    "graph_eval_wall_timeout",
                    rep=rep,
                    accepted=int(len(accepted_keys or [])),
                    wall_timeout_s=round(float(wall_timeout), 3),
                    wall_elapsed_s=round(float(wall_ge), 3),
                )
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.join(timeout=0.5)
                except Exception:
                    pass
                try:
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=0.5)
                except Exception:
                    pass
                _drain_queue()
                out29 = _force_timed_out29(latest_partial) if latest_partial is not None else _empty_out29(timed_out_flag=True)
                try:
                    _progress(
                        "graph_eval_wall_timeout_partial",
                        rep=rep,
                        accepted=int(len(accepted_keys or [])),
                        nD=(int(out29[0]) if out29[0] is not None else None),
                        nCPD=(int(out29[6]) if out29[6] is not None else None),
                    )
                except Exception:
                    pass
                graph_eval_cache[cache_key] = out29
                graph_eval_cache_wall[cache_key] = float(wall_ge)
                return (*out29, False, float(wall_ge))

            # Worker exited naturally; flush any remaining queue messages.
            _drain_queue()
            if terminal is None:
                terminal = ("err", ("No graph-eval payload", ""))

            st, payload = terminal
            if st != "ok":
                _progress(
                    "graph_eval_error",
                    rep=rep,
                    wall_elapsed_s=round(float(wall_ge), 3),
                    err=str(payload[0]) if isinstance(payload, (list, tuple)) and payload else str(payload),
                )
                out29 = _force_timed_out29(latest_partial) if latest_partial is not None else _empty_out29(timed_out_flag=True)
                graph_eval_cache[cache_key] = out29
                graph_eval_cache_wall[cache_key] = float(wall_ge)
                return (*out29, False, float(wall_ge))

            out = payload
            _progress(
                "graph_eval_done",
                rep=rep,
                accepted=int(len(accepted_keys or [])),
                wall_elapsed_s=round(float(wall_ge), 3),
                timed_out=bool(out[-1]) if isinstance(out, (list, tuple)) and len(out) == 29 else "?",
            )
            if isinstance(out, (list, tuple)) and len(out) == 29:
                graph_eval_cache[cache_key] = out  # type: ignore[assignment]
                graph_eval_cache_wall[cache_key] = float(wall_ge)
                return (*out, False, float(wall_ge))
            # Unexpected payload shape; treat as failure.
            out29 = _force_timed_out29(latest_partial) if latest_partial is not None else _empty_out29(timed_out_flag=True)
            graph_eval_cache[cache_key] = out29
            graph_eval_cache_wall[cache_key] = float(wall_ge)
            return (*out29, False, float(wall_ge))
        facts = list(case["facts"])
        if pct_wrong_facts is not None:
            rng = random.Random(rep_seed)
            facts, wrong_count = _apply_pct_wrong_facts(facts, pct_wrong_facts, rng=rng)
        else:
            wrong_count = int(case.get("count_wrong", 0))
        if len(wrong_fact_counts) < rep:
            wrong_fact_counts.extend([0] * (rep - len(wrong_fact_counts)))
        wrong_fact_counts[rep - 1] = int(wrong_count)

        _progress(
            "rep_case",
            rep=rep,
            rep_seed=rep_seed,
            facts=int(len(facts) if facts is not None else 0),
            wrong=int(wrong_count or 0),
        )

        facts_ext_raw = [f"ext_{fact_str}" for (fact_str, _I, _is_correct) in facts]

        facts_ext: list[str] = []
        weights: list[int] = []
        I_values: list[float] = []
        truths: list[str] = []
        keys_in_file_order: list[str] = []
        is_true_in_file_order: list[bool] = []
        for (fact_str, I, _is_correct), ext in zip(facts, facts_ext_raw):
            atom = _normalize_fact_line(ext)
            if not atom:
                # fallback: ext is usually like "ext_dep(1,2,s0)"
                atom = _normalize_fact_line(str(ext).strip() + ".")
            if not atom:
                continue
            facts_ext.append(atom)
            keys_in_file_order.append(atom)
            is_true_in_file_order.append(bool(_is_correct))
            try:
                I_float = float(I)
            except Exception:
                I_float = 0.0
            I_values.append(I_float)
            truths.append("unknown")
            try:
                w = int(round(float(I) * 9_999_999))
            except Exception:
                w = 1
            w = max(1, min(9_999_999, w))
            weights.append(int(w))

        if not facts_ext:
            raise SystemExit("No ext_* facts generated; cannot run sweep")

        # Sort facts by I (descending) to match solver removal ordering.
        facts_sorted = sorted(
            facts,
            key=lambda x: float(x[1]) if x[1] is not None else 0.0,
            reverse=True,
        )

        # Create per-rep subdirectory for artifacts
        rep_out_dir = out_dir / f"rep{rep}"
        rep_out_dir.mkdir(parents=True, exist_ok=True)
        rep_lps_dir = rep_out_dir / "lps"
        rep_lps_dir.mkdir(parents=True, exist_ok=True)

        facts_path, wc_path, _facts_I_path = _write_instance_files(
            rep_out_dir,
            facts_ext=facts_ext,
            weights=weights,
            I_values=I_values,
            truths=truths,
        )

        weight_map = _parse_weights_from_wc_file(wc_path)
        total_weight = sum(weight_map.values())

        # Per-rep truth/weight maps keyed by normalized ext atom.
        key_to_weight: dict[str, int] = {k: int(weight_map.get(k, 0) or 0) for k in keys_in_file_order}
        key_to_is_true: dict[str, bool] = {k: bool(t) for k, t in zip(keys_in_file_order, is_true_in_file_order)}
        n_total_rep, n_true_rep, _n_acc_rep, _n_acc_true_rep, w_total_rep, w_true_rep, _w_acc_rep, _w_acc_true_rep = _compute_acceptance_metrics(
            all_keys=keys_in_file_order,
            accepted_keys=set(keys_in_file_order),
            key_to_weight=key_to_weight,
            key_to_is_true=key_to_is_true,
        )
        # Total weight of true (correct) facts for this rep.
        total_true_weight = int(w_true_rep)
        # Incremental base dumps:
        # - `base_program_inc_paths`: skeleton-reduced dump used for the *incremental baseline* timing
        # - `base_program_inc_full_paths`: full dump (NO skeleton reduction) used for WC sweeps
        # IMPORTANT: baselines must respect opt_mode because clingo optimization mode can change results.
        base_program_inc_paths: dict[str, Path] = {}
        base_program_inc_full_paths: dict[str, Path] = {}
        for _om in opt_modes:
            need_inc_baseline_row = (not args.skip_baseline) and ((rep, "causalaba_increm", _om) not in done_baseline)
            need_inc_base_program = any(
                (enc == "inc" and optm == _om) for enc, _reif, _obj, _opt, optm in pending_wc
            )
            if not (need_inc_baseline_row or need_inc_base_program):
                continue

            base_program_inc_path = rep_lps_dir / f"abapc_inc_base_rep{rep}_{_om}.lp"
            base_program_inc_paths[str(_om)] = base_program_inc_path

            # Full (no-skeleton-reduction) base dump for WC runs.
            base_program_inc_full_path = rep_lps_dir / f"abapc_inc_full_base_rep{rep}_{_om}.lp"
            base_program_inc_full_paths[str(_om)] = base_program_inc_full_path

            if need_inc_baseline_row or (not base_program_inc_path.exists()):
                baseline_i += 1
                _progress(
                    "baseline_inc_start",
                    rep=rep,
                    opt_mode=str(_om),
                    timeout_sec=timeout_sec,
                    i=f"{baseline_i}/{baseline_total}" if baseline_total else "?",
                )
                t_inc0 = time.perf_counter()
                inc_deadline = t_inc0 + float(timeout_sec)
                inc_wall_budget_solve = max(0.0, inc_deadline - time.perf_counter())
                models_after_inc, _multiple_inc, _stats_inc, remove_n_inc, profile_inc = _run_abapc_inc_with_wall_timeout(
                    n_nodes,
                    str(facts_path),
                    weak_constraints=True,
                    search_for_models="first",
                    opt_mode=str(_om),
                    out_n=1,
                    skeleton_rules_reduction=True,
                    print_models=False,
                    return_statistics=True,
                    solve_timeout=float(timeout_sec),
                    threads=threads,
                    debug_dump_path=str(base_program_inc_path),
                    debug_dump_always=True,
                    debug_dump_include_facts=False,
                    debug_dump_materialize_block_edges=True,
                    wall_timeout=float(inc_wall_budget_solve),
                )
                wall_inc_solve = time.perf_counter() - t_inc0
                build_time_inc = None
                solve_time_inc = None
                post_solve_inc = 0.0
                try:
                    solve_time_inc = float(profile_inc.get("solve_sec_total", 0.0) or 0.0)
                except Exception:
                    solve_time_inc = None
                try:
                    post_solve_inc = float(profile_inc.get("post_solve_sec_total", 0.0) or 0.0)
                except Exception:
                    post_solve_inc = 0.0
                try:
                    if solve_time_inc is not None:
                        build_time_inc = max(
                            0.0,
                            float(wall_inc_solve) - float(solve_time_inc) - float(post_solve_inc),
                        )
                except Exception:
                    build_time_inc = None
                _progress(
                    "baseline_inc_done",
                    rep=rep,
                    opt_mode=str(_om),
                    wall_solve_sec=round(float(wall_inc_solve), 3),
                    remove_n=int(remove_n_inc or 0),
                    timed_out=bool((profile_inc or {}).get("timed_out", False)),
                )
                if not base_program_inc_path.exists():
                    # If the worker was terminated due to wall-time, it may not have
                    # flushed debug dumps. This is not a correctness issue for the
                    # baseline row itself; it only limits downstream diagnostics.
                    try:
                        if isinstance(profile_inc, dict):
                            profile_inc["base_program_dump_missing"] = True
                    except Exception:
                        pass
                    logging.warning(
                        "[warn] ABAPC_INC base program dump missing; skipping diagnostics that require it. "
                        f"rep={rep} opt_mode={_om} path={base_program_inc_path} timed_out={bool((profile_inc or {}).get('timed_out', False))}"
                    )

                if need_inc_baseline_row:
                    # Compute accepted weight independently using remove_n_inc and the sorted facts list.
                    removed_weight_inc = 0
                    removed_keys_inc: set[str] = set()
                    # Prefer explicit removed keys from ABAPC_INC (if available) because
                    # the removal set is not guaranteed to be exactly the last N facts by I.
                    removed_fact_keys_from_inc = None
                    try:
                        removed_keys = (profile_inc or {}).get("removed_fact_keys", None)
                        if isinstance(removed_keys, list) and removed_keys:
                            removed_fact_keys_from_inc = removed_keys
                            for k in removed_keys:
                                kw = _normalize_fact_line(str(k))
                                if kw:
                                    removed_keys_inc.add(kw)
                                    removed_weight_inc += int(weight_map.get(kw, 0))
                    except Exception:
                        removed_fact_keys_from_inc = None

                    if not removed_keys_inc and remove_n_inc and remove_n_inc > 0:
                        for fact_str, _I, _is_correct in facts_sorted[-int(remove_n_inc):]:
                            kw = _normalize_fact_line(f"ext_{fact_str}")
                            removed_weight_inc += int(weight_map.get(kw, 0))
                            if kw:
                                removed_keys_inc.add(kw)
                    accepted_weight = float(max(0, total_weight - removed_weight_inc))

                    accepted_keys_inc = set(keys_in_file_order) - removed_keys_inc
                    ge_budget_inc = float(graph_eval_timeout_sec)
                    (
                        n_dags_inc,
                        shd_avg_inc,
                        f1_avg_inc,
                        adj_f1_avg_inc,
                        ah_f1_avg_inc,
                        true_dag_in_compat_inc,
                        n_cpdags_inc,
                        true_cpdag_in_compat_inc,
                        cpdag_shd_avg_inc,
                        cpdag_f1_avg_inc,
                        cpdag_adj_f1_avg_inc,
                        cpdag_ah_f1_avg_inc,
                        shd_best_inc,
                        shd_worst_inc,
                        f1_best_inc,
                        f1_worst_inc,
                        adj_f1_best_inc,
                        adj_f1_worst_inc,
                        ah_f1_best_inc,
                        ah_f1_worst_inc,
                        cpdag_shd_best_inc,
                        cpdag_shd_worst_inc,
                        cpdag_f1_best_inc,
                        cpdag_f1_worst_inc,
                        cpdag_adj_f1_best_inc,
                        cpdag_adj_f1_worst_inc,
                        cpdag_ah_f1_best_inc,
                        cpdag_ah_f1_worst_inc,
                        _timed_out_ge_inc,
                        _ge_cache_hit_inc,
                        ge_inc_sec,
                    ) = _graph_eval_from_accepted_wall(
                        accepted_keys=accepted_keys_inc,
                        timeout_sec=float(ge_budget_inc),
                    )
                    (
                        n_total,
                        n_true,
                        n_acc,
                        n_acc_true,
                        w_total,
                        w_true,
                        w_acc,
                        w_acc_true,
                    ) = _compute_acceptance_metrics(
                        all_keys=keys_in_file_order,
                        accepted_keys=accepted_keys_inc,
                        key_to_weight=key_to_weight,
                        key_to_is_true=key_to_is_true,
                    )

                    # Optional consistency check vs removed_fact_keys from incremental profile.
                    try:
                        if removed_fact_keys_from_inc is not None:
                            rem_w_keys = 0
                            for k in removed_fact_keys_from_inc:
                                kw = _normalize_fact_line(str(k))
                                rem_w_keys += int(weight_map.get(kw, 0))
                            if rem_w_keys != removed_weight_inc:
                                print(f"[warn] inc removed weight mismatch: by_keys={rem_w_keys} by_used={removed_weight_inc}")
                    except Exception:
                        pass
                    wall_inc_total = time.perf_counter() - t_inc0
                    # `timed_out` is solve-timeout only; graph-eval has its own `graph_eval_timed_out` field.
                    timed_out_inc = bool((profile_inc or {}).get("timed_out", False)) or (wall_inc_solve >= float(timeout_sec) + 1e-6)

                    solve_used_inc = float(solve_time_inc) if solve_time_inc is not None else None
                    eval_time_inc = None
                    method_time_inc = None
                    try:
                        if build_time_inc is not None and solve_used_inc is not None:
                            # Eval wall-time remainder after solve (includes post-solve processing + graph-eval wall-time).
                            eval_wall = float(wall_inc_total) - float(build_time_inc) - float(solve_used_inc)
                            if eval_wall < 0.0:
                                build_time_inc = max(0.0, float(build_time_inc) + float(eval_wall))
                                eval_wall = 0.0

                            # Cache attribution: if graph-eval was a cache hit, use the cached miss cost
                            # so later methods don't look artificially cheap.
                            eval_attr = float(eval_wall)
                            try:
                                if bool(_ge_cache_hit_inc):
                                    eval_attr = max(eval_attr, float(ge_inc_sec or 0.0))
                            except Exception:
                                pass

                            eval_time_inc = float(eval_attr)
                            method_time_inc = float(build_time_inc) + float(solve_used_inc) + float(eval_time_inc)
                    except Exception:
                        eval_time_inc = None
                        method_time_inc = None
                    baseline_results.append(
                        BaselineResult(
                            rep=rep,
                            solver="causalaba_increm",
                            opt_mode=str(_om),
                            wall_time_sec=float(wall_inc_total),
                            method_time_sec=method_time_inc,
                            build_time_sec=build_time_inc,
                            solve_time_sec=solve_used_inc,
                            eval_time_sec=eval_time_inc,
                            timed_out=bool(timed_out_inc),
                            removed=int(remove_n_inc or 0),
                            models_after=len(models_after_inc or []),
                            base_program_path=str(base_program_inc_path),
                            accepted_weight=accepted_weight,
                            n_tests_total=int(n_total),
                            n_tests_true=int(n_true),
                            n_tests_accepted=int(n_acc),
                            n_tests_accepted_true=int(n_acc_true),
                            weight_total=int(w_total),
                            weight_true=int(w_true),
                            weight_accepted_true=int(w_acc_true),
                            accepted_fact_f1=float(
                                _fact_f1(n_true=int(n_true), n_acc=int(n_acc), n_acc_true=int(n_acc_true))
                            ),
                            graph_eval_timed_out=bool(_timed_out_ge_inc),
                            graph_eval_cache_hit=bool(_ge_cache_hit_inc),
                            n_dags_compat=n_dags_inc,
                            true_dag_in_compat=true_dag_in_compat_inc,
                            n_cpdags_compat=n_cpdags_inc,
                            true_cpdag_in_compat=true_cpdag_in_compat_inc,
                            shd_avg=shd_avg_inc,
                            f1_avg=f1_avg_inc,
                            adjacency_f1_avg=adj_f1_avg_inc,
                            arrowhead_f1_avg=ah_f1_avg_inc,
                            shd_best=shd_best_inc,
                            shd_worst=shd_worst_inc,
                            f1_best=f1_best_inc,
                            f1_worst=f1_worst_inc,
                            adjacency_f1_best=adj_f1_best_inc,
                            adjacency_f1_worst=adj_f1_worst_inc,
                            arrowhead_f1_best=ah_f1_best_inc,
                            arrowhead_f1_worst=ah_f1_worst_inc,
                            cpdag_shd_avg=cpdag_shd_avg_inc,
                            cpdag_f1_avg=cpdag_f1_avg_inc,
                            cpdag_adjacency_f1_avg=cpdag_adj_f1_avg_inc,
                            cpdag_arrowhead_f1_avg=cpdag_ah_f1_avg_inc,
                            cpdag_shd_best=cpdag_shd_best_inc,
                            cpdag_shd_worst=cpdag_shd_worst_inc,
                            cpdag_f1_best=cpdag_f1_best_inc,
                            cpdag_f1_worst=cpdag_f1_worst_inc,
                            cpdag_adjacency_f1_best=cpdag_adj_f1_best_inc,
                            cpdag_adjacency_f1_worst=cpdag_adj_f1_worst_inc,
                            cpdag_arrowhead_f1_best=cpdag_ah_f1_best_inc,
                            cpdag_arrowhead_f1_worst=cpdag_ah_f1_worst_inc,
                        )
                    )
                    baseline_diag[("inc", str(_om))] = {
                        "removed": int(remove_n_inc or 0),
                        "removed_keys": sorted(removed_keys_inc),
                        "accepted_keys": sorted(accepted_keys_inc),
                        "removed_fact_keys_from_inc": removed_fact_keys_from_inc,
                        "models_after": len(models_after_inc or []),
                        "timed_out": bool(timed_out_inc),
                    }
                    done_baseline.add((rep, "causalaba_increm", str(_om)))

                    _write_summary_snapshot(
                        path=partial_json,
                        is_partial=True,
                        total_weight_current=total_weight,
                    )

            # Ensure we have a *full* (no skeleton reduction) inc base-program dump for WC runs.
            # This should not affect baseline timing/semantics; it's used only as an emitted artifact
            # that MUS/WC can wrap with {mus(i)} choices + ext_* :- mus(i) links + weights.
            if need_inc_base_program and (not base_program_inc_full_path.exists()):
                _progress(
                    "inc_full_dump_start",
                    rep=rep,
                    opt_mode=str(_om),
                    path=str(base_program_inc_full_path),
                )
                # The source dump is written *before* grounding/solving in causalaba_increm.
                # Use a tight wall budget: enough to build and dump, not necessarily to finish solving.
                dump_wall_budget = min(10.0, max(1.0, float(timeout_sec) * 0.25))
                try:
                    _run_abapc_inc_with_wall_timeout(
                        n_nodes,
                        str(facts_path),
                        weak_constraints=True,
                        search_for_models="first",
                        opt_mode=str(_om),
                        out_n=1,
                        # FULL encoding: no skeleton reductions.
                        skeleton_rules_reduction=False,
                        print_models=False,
                        return_statistics=True,
                        # Minimize time spent after the dump is emitted.
                        solve_timeout=min(1.0, float(timeout_sec)),
                        threads=threads,
                        debug_dump_path=str(base_program_inc_full_path),
                        debug_dump_always=True,
                        debug_dump_include_facts=False,
                        # No skeleton reduction => no block_edge externals to materialize.
                        debug_dump_materialize_block_edges=False,
                        wall_timeout=float(dump_wall_budget),
                    )
                except Exception as e:
                    logging.warning(
                        "[warn] Failed to emit full inc base dump for WC runs: rep=%s opt_mode=%s path=%s err=%s",
                        rep,
                        str(_om),
                        str(base_program_inc_full_path),
                        repr(e),
                    )
                if not base_program_inc_full_path.exists():
                    logging.warning(
                        "[warn] Full inc base dump missing; WC inc runs may be skipped. rep=%s opt_mode=%s path=%s",
                        rep,
                        str(_om),
                        str(base_program_inc_full_path),
                    )
                _progress(
                    "inc_full_dump_done",
                    rep=rep,
                    opt_mode=str(_om),
                    exists=bool(base_program_inc_full_path.exists()),
                )

        # Baseline non-incremental ABAPC removal (per opt_mode)
        if not args.skip_baseline:
            for _om in opt_modes:
                if (rep, "causalaba", _om) in done_baseline:
                    continue

                timing_base: dict[str, Any] = {}
                baseline_i += 1
                _progress(
                    "baseline_base_start",
                    rep=rep,
                    opt_mode=str(_om),
                    timeout_sec=timeout_sec,
                    i=f"{baseline_i}/{baseline_total}" if baseline_total else "?",
                )
                t_base0 = time.perf_counter()
                base_deadline = t_base0 + float(timeout_sec)
                base_wall_budget = max(0.0, base_deadline - time.perf_counter())
                models_after_base, _multiple_base, _stats_base, remove_n_base, timing_base = _run_causalaba_with_wall_timeout(
                    n_nodes,
                    str(facts_path),
                    weak_constraints=True,
                    search_for_models="first",
                    opt_mode=str(_om),
                    out_n=1,
                    skeleton_rules_reduction=True,
                    print_models=False,
                    return_statistics=True,
                    solve_timeout=float(timeout_sec),
                    wall_timeout=float(base_wall_budget),
                    threads=threads,
                )
                wall_base_solve = time.perf_counter() - t_base0
                build_time_base = None
                solve_time_base = None
                post_solve_base = 0.0
                try:
                    solve_time_base = float(timing_base.get("solve_sec_total", 0.0) or 0.0)
                except Exception:
                    solve_time_base = None
                try:
                    post_solve_base = float(timing_base.get("post_solve_sec_total", 0.0) or 0.0)
                except Exception:
                    post_solve_base = 0.0
                try:
                    if solve_time_base is not None:
                        build_time_base = max(
                            0.0,
                            float(wall_base_solve) - float(solve_time_base) - float(post_solve_base),
                        )
                except Exception:
                    build_time_base = None
                _progress(
                    "baseline_base_done",
                    rep=rep,
                    opt_mode=str(_om),
                    wall_solve_sec=round(float(wall_base_solve), 3),
                    remove_n=int(remove_n_base or 0),
                    timed_out=bool((timing_base or {}).get("timed_out", False)),
                )

                accepted_weight_base = None
                removed_keys_base: set[str] = set()
                try:
                    if remove_n_base and remove_n_base > 0:
                        rem_w = 0
                        for fact_str, _I, _is_correct in facts_sorted[-int(remove_n_base):]:
                            kw = _normalize_fact_line(f"ext_{fact_str}")
                            rem_w += int(weight_map.get(kw, 0))
                            if kw:
                                removed_keys_base.add(kw)
                        accepted_weight_base = float(max(0, total_weight - rem_w))
                    else:
                        accepted_weight_base = float(total_weight)
                except Exception:
                    pass

                accepted_keys_base = set(keys_in_file_order) - removed_keys_base
                ge_budget_base = float(graph_eval_timeout_sec)
                (
                    n_dags_base,
                    shd_avg_base,
                    f1_avg_base,
                    adj_f1_avg_base,
                    ah_f1_avg_base,
                    true_dag_in_compat_base,
                    n_cpdags_base,
                    true_cpdag_in_compat_base,
                    cpdag_shd_avg_base,
                    cpdag_f1_avg_base,
                    cpdag_adj_f1_avg_base,
                    cpdag_ah_f1_avg_base,
                    shd_best_base,
                    shd_worst_base,
                    f1_best_base,
                    f1_worst_base,
                    adj_f1_best_base,
                    adj_f1_worst_base,
                    ah_f1_best_base,
                    ah_f1_worst_base,
                    cpdag_shd_best_base,
                    cpdag_shd_worst_base,
                    cpdag_f1_best_base,
                    cpdag_f1_worst_base,
                    cpdag_adj_f1_best_base,
                    cpdag_adj_f1_worst_base,
                    cpdag_ah_f1_best_base,
                    cpdag_ah_f1_worst_base,
                    _timed_out_ge_base,
                    _ge_cache_hit_base,
                    ge_base_sec,
                ) = _graph_eval_from_accepted_wall(
                    accepted_keys=accepted_keys_base,
                    timeout_sec=float(ge_budget_base),
                )
                (
                    n_total,
                    n_true,
                    n_acc,
                    n_acc_true,
                    w_total,
                    w_true,
                    w_acc,
                    w_acc_true,
                ) = _compute_acceptance_metrics(
                    all_keys=keys_in_file_order,
                    accepted_keys=accepted_keys_base,
                    key_to_weight=key_to_weight,
                    key_to_is_true=key_to_is_true,
                )

                wall_base_total = time.perf_counter() - t_base0
                # `timed_out` is solve-timeout only; graph-eval has its own `graph_eval_timed_out` field.
                timed_out_base = bool((timing_base or {}).get("timed_out", False)) or (
                    wall_base_solve >= float(timeout_sec) + 1e-6
                )

                solve_used_base = float(solve_time_base) if solve_time_base is not None else None
                eval_time_base = None
                method_time_base = None
                try:
                    if build_time_base is not None and solve_used_base is not None:
                        eval_wall = float(wall_base_total) - float(build_time_base) - float(solve_used_base)
                        if eval_wall < 0.0:
                            build_time_base = max(0.0, float(build_time_base) + float(eval_wall))
                            eval_wall = 0.0

                        eval_attr = float(eval_wall)
                        try:
                            if bool(_ge_cache_hit_base):
                                eval_attr = max(eval_attr, float(ge_base_sec or 0.0))
                        except Exception:
                            pass

                        eval_time_base = float(eval_attr)
                        method_time_base = float(build_time_base) + float(solve_used_base) + float(eval_time_base)
                except Exception:
                    eval_time_base = None
                    method_time_base = None
                baseline_results.append(
                    BaselineResult(
                        rep=rep,
                        solver="causalaba",
                        opt_mode=str(_om),
                        wall_time_sec=float(wall_base_total),
                        method_time_sec=method_time_base,
                        build_time_sec=build_time_base,
                        solve_time_sec=solve_used_base,
                        eval_time_sec=eval_time_base,
                        timed_out=bool(timed_out_base),
                        removed=int(remove_n_base or 0),
                        models_after=len(models_after_base or []),
                        accepted_weight=accepted_weight_base,
                        n_tests_total=int(n_total),
                        n_tests_true=int(n_true),
                        n_tests_accepted=int(n_acc),
                        n_tests_accepted_true=int(n_acc_true),
                        weight_total=int(w_total),
                        weight_true=int(w_true),
                        weight_accepted_true=int(w_acc_true),
                        accepted_fact_f1=float(
                            _fact_f1(n_true=int(n_true), n_acc=int(n_acc), n_acc_true=int(n_acc_true))
                        ),
                        graph_eval_timed_out=bool(_timed_out_ge_base),
                        graph_eval_cache_hit=bool(_ge_cache_hit_base),
                        n_dags_compat=n_dags_base,
                        true_dag_in_compat=true_dag_in_compat_base,
                        n_cpdags_compat=n_cpdags_base,
                        true_cpdag_in_compat=true_cpdag_in_compat_base,
                        shd_avg=shd_avg_base,
                        f1_avg=f1_avg_base,
                        adjacency_f1_avg=adj_f1_avg_base,
                        arrowhead_f1_avg=ah_f1_avg_base,
                        shd_best=shd_best_base,
                        shd_worst=shd_worst_base,
                        f1_best=f1_best_base,
                        f1_worst=f1_worst_base,
                        adjacency_f1_best=adj_f1_best_base,
                        adjacency_f1_worst=adj_f1_worst_base,
                        arrowhead_f1_best=ah_f1_best_base,
                        arrowhead_f1_worst=ah_f1_worst_base,
                        cpdag_shd_avg=cpdag_shd_avg_base,
                        cpdag_f1_avg=cpdag_f1_avg_base,
                        cpdag_adjacency_f1_avg=cpdag_adj_f1_avg_base,
                        cpdag_arrowhead_f1_avg=cpdag_ah_f1_avg_base,
                        cpdag_shd_best=cpdag_shd_best_base,
                        cpdag_shd_worst=cpdag_shd_worst_base,
                        cpdag_f1_best=cpdag_f1_best_base,
                        cpdag_f1_worst=cpdag_f1_worst_base,
                        cpdag_adjacency_f1_best=cpdag_adj_f1_best_base,
                        cpdag_adjacency_f1_worst=cpdag_adj_f1_worst_base,
                        cpdag_arrowhead_f1_best=cpdag_ah_f1_best_base,
                        cpdag_arrowhead_f1_worst=cpdag_ah_f1_worst_base,
                    )
                )
                baseline_diag[("base", str(_om))] = {
                    "removed": int(remove_n_base or 0),
                    "removed_keys": sorted(removed_keys_base),
                    "accepted_keys": sorted(accepted_keys_base),
                    "models_after": len(models_after_base or []),
                    "timed_out": bool(timed_out_base),
                }
                done_baseline.add((rep, "causalaba", str(_om)))

                # Diagnostic: if base/inc disagree on removal, write a report.
                inc_key = ("inc", str(_om))
                base_key = ("base", str(_om))
                if inc_key in baseline_diag and base_key in baseline_diag:
                    inc_d = baseline_diag[inc_key]
                    base_d = baseline_diag[base_key]
                    removed_inc = int(inc_d.get("removed", 0))
                    removed_base = int(base_d.get("removed", 0))
                    timed_out_inc = bool(inc_d.get("timed_out", False))
                    timed_out_base = bool(base_d.get("timed_out", False))
                    accepted_inc = set(inc_d.get("accepted_keys", []) or [])
                    accepted_base = set(base_d.get("accepted_keys", []) or [])
                    mismatch = (removed_inc != removed_base) or (accepted_inc != accepted_base)
                    if mismatch:
                        diag_path = rep_out_dir / f"baseline_mismatch_{_om}.json"
                        diag_timeout = max(1.0, min(10.0, float(timeout_sec)))

                        def _write_diag_facts(accepted: list[str], *, label: str) -> Path:
                            diag_dir = rep_out_dir / "diag"
                            diag_dir.mkdir(parents=True, exist_ok=True)
                            out_path = diag_dir / f"facts_{label}.lp"
                            with out_path.open("w") as f:
                                for k in sorted(accepted):
                                    s = str(k).strip()
                                    if not s:
                                        continue
                                    if not s.endswith("."):
                                        s = s + "."
                                    f.write(f"#external {s}\n")
                            return out_path

                        def _sat_check(*, solver: str, facts_path: Path) -> dict[str, Any]:
                            try:
                                if solver == "base":
                                    models_after, _multiple, _stats, remove_n, timing = _run_causalaba_with_wall_timeout(
                                        n_nodes,
                                        str(facts_path),
                                        weak_constraints=False,
                                        search_for_models="first",
                                        opt_mode="ignore",
                                        out_n=1,
                                        skeleton_rules_reduction=True,
                                        print_models=False,
                                        return_statistics=True,
                                        solve_timeout=float(diag_timeout),
                                        wall_timeout=float(diag_timeout) + 0.5,
                                        threads=threads,
                                    )
                                    timed_out = bool((timing or {}).get("timed_out", False))
                                else:
                                    models_after, _multiple, _stats, remove_n, profile = _run_abapc_inc_with_wall_timeout(
                                        n_nodes,
                                        str(facts_path),
                                        weak_constraints=False,
                                        search_for_models="first",
                                        opt_mode="ignore",
                                        out_n=1,
                                        skeleton_rules_reduction=True,
                                        print_models=False,
                                        return_statistics=True,
                                        solve_timeout=float(diag_timeout),
                                        threads=threads,
                                        wall_timeout=float(diag_timeout) + 0.5,
                                    )
                                    timed_out = bool((profile or {}).get("timed_out", False))
                                n_models = len(models_after or [])
                                return {
                                    "sat": bool(n_models > 0 and int(remove_n or 0) == 0),
                                    "n_models": int(n_models),
                                    "remove_n": int(remove_n or 0),
                                    "timed_out": bool(timed_out),
                                }
                            except Exception as e:
                                return {"sat": False, "n_models": 0, "remove_n": None, "timed_out": False, "error": repr(e)}

                        diag_inc_facts = _write_diag_facts(inc_d.get("accepted_keys", []), label=f"inc_{_om}")
                        diag_base_facts = _write_diag_facts(base_d.get("accepted_keys", []), label=f"base_{_om}")
                        sat_base_on_inc = _sat_check(solver="base", facts_path=diag_inc_facts)
                        sat_inc_on_inc = _sat_check(solver="inc", facts_path=diag_inc_facts)
                        sat_base_on_base = _sat_check(solver="base", facts_path=diag_base_facts)
                        sat_inc_on_base = _sat_check(solver="inc", facts_path=diag_base_facts)
                        try:
                            payload = {
                                "rep": int(rep),
                                "opt_mode": str(_om),
                                "removed_inc": int(removed_inc),
                                "removed_base": int(removed_base),
                                "models_after_inc": int(inc_d.get("models_after", 0)),
                                "models_after_base": int(base_d.get("models_after", 0)),
                                "removed_keys_inc": inc_d.get("removed_keys", []),
                                "removed_keys_base": base_d.get("removed_keys", []),
                                "accepted_keys_inc": inc_d.get("accepted_keys", []),
                                "accepted_keys_base": base_d.get("accepted_keys", []),
                                "removed_fact_keys_from_inc": inc_d.get("removed_fact_keys_from_inc", None),
                                "timed_out_inc": bool(timed_out_inc),
                                "timed_out_base": bool(timed_out_base),
                                "sat_base_on_inc": sat_base_on_inc,
                                "sat_inc_on_inc": sat_inc_on_inc,
                                "sat_base_on_base": sat_base_on_base,
                                "sat_inc_on_base": sat_inc_on_base,
                            }
                            diag_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                        except Exception:
                            pass

                        # Hard fail unless this was caused by timeouts/termination.
                        if not timed_out_inc and not timed_out_base:
                            raise AssertionError(
                                "Baseline base/inc correspondence broke "
                                f"(rep={rep} opt_mode={_om} removed_inc={removed_inc} removed_base={removed_base} "
                                f"accepted_eq={accepted_inc == accepted_base}). "
                                f"Diagnostics: {diag_path}"
                            )

                _write_summary_snapshot(
                    path=partial_json,
                    is_partial=True,
                    total_weight_current=total_weight,
                )

        # For direct reification with base encoding, emit a full base program (no skeleton reduction).
        base_program_base_path = rep_lps_dir / f"abapc_base_base_rep{rep}.lp"
        if "direct" in reifications and "base" in encodings:
            _emit_full_base_program_for_direct(
                n_nodes=n_nodes,
                facts_path=facts_path,
                out_path=base_program_base_path,
            )

        def _harmonize_status(raw_status: str, *, timed_out: bool) -> str:
            """Normalize varying solver status strings into a small set.

            We treat optimization runs as:
            - OPT: optimality proven
            - SAT: satisfiable but optimality not proven
            - UNSAT: unsatisfiable
            - TIMEOUT: cancelled due to our timeout budget
            - UNKNOWN: solver returned unknown/interrupted for non-timeout reasons
            - ERROR: uncaught/explicit error
            """
            if timed_out:
                return "TIMEOUT"
            s = (raw_status or "").strip().upper()
            if not s:
                return "UNKNOWN"
            if s.startswith("ERROR"):
                return "ERROR"
            if "UNSAT" in s:
                return "UNSAT"
            # clingo JSON uses 'OPTIMUM FOUND'
            if "OPT" in s:
                return "OPT"
            if "SAT" in s:
                return "SAT"
            if "UNKNOWN" in s:
                return "UNKNOWN"
            if "INTERRUPT" in s:
                return "UNKNOWN"
            return "UNKNOWN"

        # WC optimization sweeps
        if not bool(getattr(args, "skip_wc", False)):
            for enc in encodings:
                for reif in reifications:
                    for obj in objectives:
                        for opt in strategies:
                            for opt_mode in opt_modes:
                                if (rep, enc, obj, opt, opt_mode, reif) in done_wc:
                                    continue
                                emit_name = f"wc_{enc}_{reif}_{obj}_{opt}_{opt_mode}_r{rep}.lp"
                                wc_i += 1
                                _progress(
                                    "wc_run_start",
                                    rep=rep,
                                    encoding=enc,
                                    reif=reif,
                                    objective=obj,
                                    opt_strategy=opt,
                                    opt_mode=opt_mode,
                                    i=f"{wc_i}/{wc_total}" if wc_total else "?",
                                )
                                t0 = time.perf_counter()
                                if reif == "mus":
                                    # MUS reification: use CausalABA_WC with base program
                                    base_path_arg: str | None = None
                                    if enc == "inc":
                                        # For WC runs, require the FULL (no skeleton reduction) inc base dump.
                                        base_program_inc_path = base_program_inc_full_paths.get(opt_mode)
                                        if base_program_inc_path is None or (not base_program_inc_path.exists()):
                                            logging.warning(
                                                "[warn] Skipping WC MUS run: missing inc base program dump. "
                                                f"rep={rep} opt_mode={opt_mode} enc={enc} path={base_program_inc_path}"
                                            )
                                            wc_results.append(
                                                StrategyResult(
                                                    rep=rep,
                                                    encoding=str(enc),
                                                    objective=str(obj),
                                                    opt_strategy=str(opt),
                                                    opt_mode=str(opt_mode),
                                                    reification=str(reif),
                                                    timed_out=True,
                                                    wall_time_sec=0.0,
                                                    status="SKIPPED_MISSING_BASE_DUMP",
                                                    objective_costs=[],
                                                )
                                            )
                                            done_wc.add((rep, enc, obj, opt, opt_mode, reif))
                                            _write_summary_snapshot(
                                                path=partial_json,
                                                is_partial=True,
                                                total_weight_current=total_weight,
                                            )
                                            continue
                                        base_path_arg = str(base_program_inc_path)
                                    # For base encoding, pass None to trigger compile-and-ground inside CausalABA_WC

                                    # Solve timeout is independent from graph-eval timeout.
                                    solve_timeout_sec = float(timeout_sec)

                                    wc_kwargs = {
                                        "n_nodes": n_nodes,
                                        "facts_location": str(facts_path),
                                        "facts_wc_location": str(wc_path),
                                        "solve_timeout": float(solve_timeout_sec),
                                        "opt_strategy": opt,
                                        "opt_mode": opt_mode,
                                        "objective": obj,
                                        "emit_lp": str(rep_lps_dir / emit_name),
                                        "base_program_path": base_path_arg,
                                    }

                                    isolate_wc_mus = bool(
                                        getattr(args, "isolate_wc_mus", False) or getattr(args, "isolate_wc", False)
                                    )
                                    if isolate_wc_mus:
                                        # Give a small overhead beyond solve_timeout for setup/teardown.
                                        _st, _payload, _exitcode, _err = _run_causalaba_wc_isolated(
                                            wall_timeout=float(solve_timeout_sec) + 3.0,
                                            kwargs=wc_kwargs,
                                        )
                                        if _st == "ok" and isinstance(_payload, dict):
                                            res = _payload
                                        else:
                                            wall = time.perf_counter() - t0
                                            exit_sig = None
                                            if isinstance(_exitcode, int) and _exitcode < 0:
                                                exit_sig = int(-_exitcode)
                                            reason = ""
                                            if _st == "timeout":
                                                reason = "TIMEOUT"
                                            elif _st == "killed":
                                                reason = f"KILLED_SIG{exit_sig}" if exit_sig is not None else "KILLED"
                                            else:
                                                reason = "FAILED"

                                            logging.warning(
                                                f"⚠ CausalABA_WC isolated run did not complete: status={_st} exitcode={_exitcode}"
                                            )
                                            if _err:
                                                logging.warning(_err)

                                            # Record a placeholder row so resume skips it, then continue.
                                            wc_results.append(
                                                StrategyResult(
                                                    rep=rep,
                                                    encoding=enc,
                                                    objective=obj,
                                                    opt_strategy=opt,
                                                    opt_mode=opt_mode,
                                                    reification=reif,
                                                    timed_out=bool(_st == "timeout"),
                                                    wall_time_sec=float(wall),
                                                    status=f"ERROR_{reason}",
                                                    objective_costs=[],
                                                    cut_weight=None,
                                                    selected_count=None,
                                                    build_time_sec=None,
                                                    solve_time_sec=None,
                                                    eval_time_sec=None,
                                                    accepted_weight=None,
                                                    n_tests_total=int(len(keys_in_file_order)),
                                                    n_tests_true=int(sum(1 for k in keys_in_file_order if key_to_is_true.get(k, False))),
                                                    n_tests_accepted=0,
                                                    n_tests_accepted_true=0,
                                                    weight_total=int(total_weight),
                                                    weight_true=int(total_true_weight),
                                                    weight_accepted_true=0,
                                                    accepted_fact_f1=None,
                                                    n_dags_compat=None,
                                                    true_dag_in_compat=None,
                                                    n_cpdags_compat=None,
                                                    true_cpdag_in_compat=None,
                                                    shd_avg=None,
                                                    f1_avg=None,
                                                    adjacency_f1_avg=None,
                                                    arrowhead_f1_avg=None,
                                                    cpdag_shd_avg=None,
                                                    cpdag_f1_avg=None,
                                                    cpdag_adjacency_f1_avg=None,
                                                    cpdag_arrowhead_f1_avg=None,
                                            )
                                        )
                                        done_wc.add((rep, enc, obj, opt, opt_mode, reif))
                                        _write_summary_snapshot(
                                            path=partial_json,
                                            is_partial=True,
                                            total_weight_current=total_weight,
                                        )
                                        continue
                                    else:
                                        res = CausalABA_WC(**wc_kwargs)
                                    wall = time.perf_counter() - t0

                                    # If compile-and-ground (inside MUS program build) hit the wall-time
                                    # budget, CausalABA_WC returns a TIMEOUT_BUILD sentinel. Treat this as
                                    # a timeout run with only timing populated (no acceptance/graph metrics).
                                    if (
                                        isinstance(res, dict)
                                        and str(res.get("status", "")).strip().upper() == "TIMEOUT_BUILD"
                                    ):
                                        build_time_wc = float(res.get("wc_build_time", 0.0) or 0.0)
                                        if build_time_wc <= 0.0:
                                            build_time_wc = float(wall)
                                        status_wc = _harmonize_status(
                                            str(res.get("status", "UNKNOWN")), timed_out=True
                                        )
                                        wc_results.append(
                                            StrategyResult(
                                                rep=rep,
                                                encoding=enc,
                                                objective=obj,
                                                opt_strategy=opt,
                                                opt_mode=opt_mode,
                                                reification=reif,
                                                timed_out=True,
                                                wall_time_sec=float(wall),
                                                method_time_sec=float(build_time_wc),
                                                status=status_wc,
                                                objective_costs=[],
                                                cut_weight=None,
                                                selected_count=None,
                                                build_time_sec=float(build_time_wc),
                                                solve_time_sec=None,
                                                eval_time_sec=None,
                                                accepted_weight=None,
                                                n_tests_total=int(len(keys_in_file_order)),
                                                n_tests_true=int(
                                                    sum(1 for k in keys_in_file_order if key_to_is_true.get(k, False))
                                                ),
                                                n_tests_accepted=0,
                                                n_tests_accepted_true=0,
                                                weight_total=int(total_weight),
                                                weight_true=int(total_true_weight),
                                                weight_accepted_true=0,
                                                accepted_fact_f1=None,
                                                graph_eval_timed_out=None,
                                                graph_eval_cache_hit=None,
                                                n_dags_compat=None,
                                                true_dag_in_compat=None,
                                                n_cpdags_compat=None,
                                                true_cpdag_in_compat=None,
                                                shd_avg=None,
                                                f1_avg=None,
                                                adjacency_f1_avg=None,
                                                arrowhead_f1_avg=None,
                                                shd_best=None,
                                                shd_worst=None,
                                                f1_best=None,
                                                f1_worst=None,
                                                adjacency_f1_best=None,
                                                adjacency_f1_worst=None,
                                                arrowhead_f1_best=None,
                                                arrowhead_f1_worst=None,
                                                cpdag_shd_avg=None,
                                                cpdag_f1_avg=None,
                                                cpdag_adjacency_f1_avg=None,
                                                cpdag_arrowhead_f1_avg=None,
                                                cpdag_shd_best=None,
                                                cpdag_shd_worst=None,
                                                cpdag_f1_best=None,
                                                cpdag_f1_worst=None,
                                                cpdag_adjacency_f1_best=None,
                                                cpdag_adjacency_f1_worst=None,
                                                cpdag_arrowhead_f1_best=None,
                                                cpdag_arrowhead_f1_worst=None,
                                            )
                                        )
                                        done_wc.add((rep, enc, obj, opt, opt_mode, reif))
                                        _progress(
                                            "wc_run_done",
                                            rep=rep,
                                            encoding=enc,
                                            reif=reif,
                                            objective=obj,
                                            opt_strategy=opt,
                                            opt_mode=opt_mode,
                                            wall_time_sec=round(float(wall), 3),
                                            status=str(res.get("status")),
                                            timed_out=True,
                                            graph_eval_timed_out=False,
                                        )
                                        _write_summary_snapshot(
                                            path=partial_json,
                                            is_partial=True,
                                            total_weight_current=total_weight,
                                        )
                                        continue

                                    # Derive accepted set from selected_mus indices.
                                    selected_mus = [int(x) for x in (res.get("selected_mus", []) or [])]
                                    accepted_keys_mus: set[str] = set()
                                    for idx in selected_mus:
                                        if 1 <= idx <= len(keys_in_file_order):
                                            accepted_keys_mus.add(keys_in_file_order[idx - 1])

                                    (
                                        n_total,
                                        n_true,
                                        n_acc,
                                        n_acc_true,
                                        w_total,
                                        w_true,
                                        w_acc,
                                        w_acc_true,
                                    ) = _compute_acceptance_metrics(
                                        all_keys=keys_in_file_order,
                                        accepted_keys=accepted_keys_mus,
                                        key_to_weight=key_to_weight,
                                        key_to_is_true=key_to_is_true,
                                    )

                                    (
                                        n_dags,
                                        shd_avg,
                                        f1_avg,
                                        adj_f1_avg,
                                        ah_f1_avg,
                                        true_dag_in_compat,
                                        n_cpdags,
                                        true_cpdag_in_compat,
                                        cpdag_shd_avg,
                                        cpdag_f1_avg,
                                        cpdag_adj_f1_avg,
                                        cpdag_ah_f1_avg,
                                        shd_best,
                                        shd_worst,
                                        f1_best,
                                        f1_worst,
                                        adj_f1_best,
                                        adj_f1_worst,
                                        ah_f1_best,
                                        ah_f1_worst,
                                        cpdag_shd_best,
                                        cpdag_shd_worst,
                                        cpdag_f1_best,
                                        cpdag_f1_worst,
                                        cpdag_adj_f1_best,
                                        cpdag_adj_f1_worst,
                                        cpdag_ah_f1_best,
                                        cpdag_ah_f1_worst,
                                        _timed_out_ge,
                                    ) = (
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        None,
                                        False,
                                    )

                                    # Run graph-eval under remaining budget; if it times out, treat the whole method as timed out.
                                    import math

                                    remaining_eval = float(graph_eval_timeout_sec)
                                    (
                                        n_dags,
                                        shd_avg,
                                        f1_avg,
                                        adj_f1_avg,
                                        ah_f1_avg,
                                        true_dag_in_compat,
                                        n_cpdags,
                                        true_cpdag_in_compat,
                                        cpdag_shd_avg,
                                        cpdag_f1_avg,
                                        cpdag_adj_f1_avg,
                                        cpdag_ah_f1_avg,
                                        shd_best,
                                        shd_worst,
                                        f1_best,
                                        f1_worst,
                                        adj_f1_best,
                                        adj_f1_worst,
                                        ah_f1_best,
                                        ah_f1_worst,
                                        cpdag_shd_best,
                                        cpdag_shd_worst,
                                        cpdag_f1_best,
                                        cpdag_f1_worst,
                                        cpdag_adj_f1_best,
                                        cpdag_adj_f1_worst,
                                        cpdag_ah_f1_best,
                                        cpdag_ah_f1_worst,
                                        _timed_out_ge,
                                        _ge_cache_hit,
                                        eval_time_sec,
                                    ) = _graph_eval_from_accepted_wall(
                                        accepted_keys=accepted_keys_mus,
                                        timeout_sec=float(remaining_eval),
                                    )

                                    wall = time.perf_counter() - t0

                                    acc_w = float(w_acc)
                                    cw = res.get("cut_weight", None) if res is not None else None
                                    timed_out_solver = bool(res.get("timed_out", False)) if res is not None else False
                                    timed_out_wc = bool(timed_out_solver)
                                    build_time_wc = None if timed_out_solver else float(res.get("wc_build_time", 0.0) or 0.0)
                                    solve_time_wc = None if timed_out_solver else float(res.get("solve_time", 0.0) or 0.0)
                                    eval_after_sec = None
                                    try:
                                        if build_time_wc is not None and solve_time_wc is not None:
                                            eval_wall = float(wall) - float(build_time_wc) - float(solve_time_wc)
                                            if eval_wall < 0.0:
                                                build_time_wc = max(0.0, float(build_time_wc) + float(eval_wall))
                                                eval_wall = 0.0

                                            eval_attr = float(eval_wall)
                                            try:
                                                if bool(_ge_cache_hit):
                                                    eval_attr = max(eval_attr, float(eval_time_sec or 0.0))
                                            except Exception:
                                                pass
                                            eval_after_sec = float(eval_attr)
                                    except Exception:
                                        eval_after_sec = None
                                    method_time_wc = None
                                    try:
                                        if build_time_wc is not None and solve_time_wc is not None and eval_after_sec is not None:
                                            method_time_wc = float(build_time_wc) + float(solve_time_wc) + float(eval_after_sec)
                                    except Exception:
                                        method_time_wc = None
                                    status_wc = _harmonize_status(str(res.get("status", "UNKNOWN")), timed_out=timed_out_wc)
                                    wc_results.append(
                                        StrategyResult(
                                            rep=rep,
                                            encoding=enc,
                                            objective=obj,
                                            opt_strategy=opt,
                                            opt_mode=opt_mode,
                                            reification=reif,
                                            timed_out=timed_out_wc,
                                            wall_time_sec=float(wall),
                                            method_time_sec=method_time_wc,
                                            status=status_wc,
                                            objective_costs=[int(x) for x in (res.get("costs", []) or [])],
                                            cut_weight=int(res.get("cut_weight", 0) or 0) if res is not None else None,
                                            selected_count=len(selected_mus or []),
                                            build_time_sec=build_time_wc,
                                            solve_time_sec=solve_time_wc,
                                            eval_time_sec=eval_after_sec,
                                            accepted_weight=acc_w,
                                            n_tests_total=int(n_total),
                                            n_tests_true=int(n_true),
                                            n_tests_accepted=int(n_acc),
                                            n_tests_accepted_true=int(n_acc_true),
                                            weight_total=int(w_total),
                                            weight_true=int(w_true),
                                            weight_accepted_true=int(w_acc_true),
                                            accepted_fact_f1=float(
                                                _fact_f1(n_true=int(n_true), n_acc=int(n_acc), n_acc_true=int(n_acc_true))
                                            ),
                                            graph_eval_timed_out=bool(_timed_out_ge),
                                            graph_eval_cache_hit=bool(_ge_cache_hit),
                                            n_dags_compat=n_dags,
                                            true_dag_in_compat=int(true_dag_in_compat) if true_dag_in_compat is not None else None,
                                            n_cpdags_compat=int(n_cpdags) if n_cpdags is not None else None,
                                            true_cpdag_in_compat=int(true_cpdag_in_compat) if true_cpdag_in_compat is not None else None,
                                            shd_avg=shd_avg,
                                            f1_avg=f1_avg,
                                            adjacency_f1_avg=adj_f1_avg,
                                            arrowhead_f1_avg=ah_f1_avg,
                                            shd_best=shd_best,
                                            shd_worst=shd_worst,
                                            f1_best=f1_best,
                                            f1_worst=f1_worst,
                                            adjacency_f1_best=adj_f1_best,
                                            adjacency_f1_worst=adj_f1_worst,
                                            arrowhead_f1_best=ah_f1_best,
                                            arrowhead_f1_worst=ah_f1_worst,
                                            cpdag_shd_avg=cpdag_shd_avg,
                                            cpdag_f1_avg=cpdag_f1_avg,
                                            cpdag_adjacency_f1_avg=cpdag_adj_f1_avg,
                                            cpdag_arrowhead_f1_avg=cpdag_ah_f1_avg,
                                            cpdag_shd_best=cpdag_shd_best,
                                            cpdag_shd_worst=cpdag_shd_worst,
                                            cpdag_f1_best=cpdag_f1_best,
                                            cpdag_f1_worst=cpdag_f1_worst,
                                            cpdag_adjacency_f1_best=cpdag_adj_f1_best,
                                            cpdag_adjacency_f1_worst=cpdag_adj_f1_worst,
                                            cpdag_arrowhead_f1_best=cpdag_ah_f1_best,
                                            cpdag_arrowhead_f1_worst=cpdag_ah_f1_worst,
                                        )
                                    )
                                    done_wc.add((rep, enc, obj, opt, opt_mode, reif))

                                    _progress(
                                        "wc_run_done",
                                        rep=rep,
                                        encoding=enc,
                                        reif=reif,
                                        objective=obj,
                                        opt_strategy=opt,
                                        opt_mode=opt_mode,
                                        wall_time_sec=round(float(wall), 3),
                                        status=str(res.get("status")) if isinstance(res, dict) else "?",
                                        timed_out=bool(res.get("timed_out", False)) if isinstance(res, dict) else "?",
                                        graph_eval_timed_out=bool(_timed_out_ge),
                                    )

                                    _write_summary_snapshot(
                                        path=partial_json,
                                        is_partial=True,
                                        total_weight_current=total_weight,
                                    )

                            if reif != "direct":
                                continue

                            # Select base program dump
                            if enc == "inc":
                                base_program_inc_path = base_program_inc_paths.get(opt_mode)
                                if base_program_inc_path is None or (not base_program_inc_path.exists()):
                                    logging.warning(
                                        "[warn] Skipping WC direct run: missing inc base program dump. "
                                        f"rep={rep} opt_mode={opt_mode} enc={enc} path={base_program_inc_path}"
                                    )
                                    wc_results.append(
                                        StrategyResult(
                                            rep=rep,
                                            encoding=str(enc),
                                            objective=str(obj),
                                            opt_strategy=str(opt),
                                            opt_mode=str(opt_mode),
                                            reification=str(reif),
                                            timed_out=True,
                                            wall_time_sec=0.0,
                                            status="SKIPPED_MISSING_BASE_DUMP",
                                            objective_costs=[],
                                        )
                                    )
                                    done_wc.add((rep, enc, obj, opt, opt_mode, reif))
                                    _write_summary_snapshot(
                                        path=partial_json,
                                        is_partial=True,
                                        total_weight_current=total_weight,
                                    )
                                    continue
                                base_program_path = base_program_inc_path
                            else:
                                base_program_path = base_program_base_path
                            
                            # Build clingo command with base program + facts + weak constraints
                            wc_path = facts_path.parent / f"{facts_path.stem}_wc.lp"
                            wc_text = _build_direct_wc_text(
                                keys_in_file_order=keys_in_file_order,
                                key_to_weight=key_to_weight,
                                objective=obj,
                            )
                            
                            build_t0 = time.perf_counter()
                            
                            # Load base program and strip skeleton-reduction artifacts (block_edge) for consistency
                            # when using inc encoding dump with direct reification.
                            with open(str(base_program_path), 'r') as f:
                                base_program_text = f.read()
                            normalized_base_program = _normalize_base_program_for_analysis(base_program_text, n_nodes=n_nodes)

                            # Persist a runnable program artifact for parity with MUS runs.
                            # (This is not used by the solver; it's for inspection/debugging.)
                            try:
                                direct_emit_path = rep_lps_dir / emit_name
                                direct_emit_path.write_text(
                                    normalized_base_program
                                    + "\n\n% facts (externals)\n"
                                    + facts_path.read_text()
                                    + "\n\n% weak constraints\n"
                                    + wc_text
                                    + "\n"
                                )
                            except Exception:
                                pass
                            
                            # Write normalized base program to a temp file for clingo
                            solve_t0 = time.perf_counter()
                            status_direct = "UNKNOWN"
                            costs_direct: list[int] = []
                            timed_out_direct = False
                            accepted_keys_direct: set[str] = set()
                            isolate_wc_all = bool(getattr(args, "isolate_wc", False))
                            if isolate_wc_all:
                                build_sec = time.perf_counter() - build_t0
                                direct_kwargs = {
                                    "n_nodes": int(n_nodes),
                                    "base_program_path": str(base_program_path),
                                    "facts_path": str(facts_path),
                                    "keys_in_file_order": list(keys_in_file_order),
                                    "key_to_weight": dict(key_to_weight),
                                    "objective": str(obj),
                                    "opt_mode": str(opt_mode),
                                    "opt_strategy": str(opt),
                                    "threads": int(threads),
                                    "solve_timeout": float(timeout_sec),
                                }
                                _st, _payload, _exitcode, _err = _run_direct_wc_isolated(
                                    wall_timeout=float(timeout_sec) + 0.5,
                                    kwargs=direct_kwargs,
                                )
                                if _st == "ok" and isinstance(_payload, dict):
                                    status_direct = str(_payload.get("status", "UNKNOWN"))
                                    costs_direct = list(_payload.get("costs") or [])
                                    timed_out_direct = bool(_payload.get("timed_out", False))
                                    accepted_keys_direct = set(_payload.get("accepted_keys") or [])
                                else:
                                    exit_sig = None
                                    if isinstance(_exitcode, int) and _exitcode < 0:
                                        exit_sig = int(-_exitcode)
                                    reason = "FAILED"
                                    if _st == "timeout":
                                        reason = "TIMEOUT"
                                        timed_out_direct = True
                                    elif _st == "killed":
                                        reason = f"KILLED_SIG{exit_sig}" if exit_sig is not None else "KILLED"
                                    if _err:
                                        logging.warning(_err)
                                    status_direct = f"ERROR_{reason}"
                                    costs_direct = []
                                    accepted_keys_direct = set()
                                solve_sec = time.perf_counter() - solve_t0
                            else:
                                # Write normalized base program to a temp file for clingo
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.lp', delete=False) as tmp:
                                    tmp.write(normalized_base_program)
                                    normalized_base_path = tmp.name
                                build_sec = time.perf_counter() - build_t0

                                # Run clingo using Python API (supports assign_external(..., None)).
                                import clingo

                                try:
                                    ctl = clingo.Control(
                                        [
                                            f"--opt-mode={opt_mode}",
                                            f"--opt-strategy={opt}",
                                            "-n",
                                            "1",
                                            "-t",
                                            str(threads),
                                            "--warn=none",
                                        ]
                                    )
                                    ctl.load(str(normalized_base_path))
                                    # facts.lp contains `#external ext_...` declarations
                                    ctl.load(str(facts_path))
                                    if wc_text.strip():
                                        ctl.add("base", [], wc_text)
                                    ctl.ground([("base", [])])

                                    key_syms: list[tuple[str, clingo.Symbol]] = []
                                    for k in keys_in_file_order:
                                        try:
                                            key_syms.append((k, clingo.parse_term(k)))
                                        except Exception:
                                            continue

                                    # Make each external "free" so clingo can choose it.
                                    for _k, sym in key_syms:
                                        try:
                                            ctl.assign_external(sym, None)
                                        except Exception:
                                            pass

                                    last_cost: list[int] = []
                                    last_keys: set[str] = set()
                                    last_optimality_proven: bool = False

                                    def _on_model(m: clingo.Model) -> None:
                                        nonlocal last_cost, last_keys, last_optimality_proven
                                        try:
                                            last_cost = [int(x) for x in (m.cost or [])]
                                        except Exception:
                                            last_cost = []
                                        try:
                                            last_optimality_proven = bool(getattr(m, "optimality_proven", False))
                                        except Exception:
                                            last_optimality_proven = False
                                        chosen: set[str] = set()
                                        for kk, ss in key_syms:
                                            try:
                                                if m.contains(ss):
                                                    chosen.add(kk)
                                            except Exception:
                                                continue
                                        last_keys = chosen

                                    # Enforce a single wall-time budget across the whole run.
                                    solve_timeout_sec = float(timeout_sec)
                                    if solve_timeout_sec <= 0.0:
                                        timed_out_direct = True
                                    else:
                                        handle = ctl.solve(async_=True, on_model=_on_model)
                                        finished = handle.wait(timeout=float(solve_timeout_sec))
                                        if not finished:
                                            timed_out_direct = True
                                            try:
                                                handle.cancel()
                                            except Exception:
                                                pass
                                            try:
                                                # Bounded grace period; never block indefinitely after cancel.
                                                finished = handle.wait(timeout=1.0)
                                            except Exception:
                                                finished = False

                                    res = None
                                    if not timed_out_direct and finished:
                                        try:
                                            res = handle.get()
                                        except Exception:
                                            res = None

                                    if timed_out_direct:
                                        status_direct = "TIMEOUT"
                                    elif res is not None and bool(getattr(res, "unsatisfiable", False)):
                                        status_direct = "UNSAT"
                                    elif res is not None and bool(getattr(res, "satisfiable", False)):
                                        status_direct = "OPT" if last_optimality_proven else "SAT"
                                    elif res is not None and bool(getattr(res, "unknown", False)):
                                        status_direct = "UNKNOWN"
                                    elif res is not None and bool(getattr(res, "interrupted", False)):
                                        status_direct = "UNKNOWN"
                                    else:
                                        status_direct = "UNKNOWN"

                                    costs_direct = list(last_cost or [])
                                    accepted_keys_direct = set(last_keys or set())

                                except Exception:
                                    logging.exception("Direct solve failed")
                                    status_direct = "ERROR"
                                    costs_direct = []
                                    accepted_keys_direct = set()
                                    timed_out_direct = False

                                solve_sec = time.perf_counter() - solve_t0
                                
                                # Clean up temp file
                                try:
                                    os.remove(normalized_base_path)
                                except Exception:
                                    pass
                            
                            # If the solver stage timed out, hide build/solve split since it's incomplete.
                            if timed_out_direct:
                                build_sec = None
                                solve_sec = None

                            (
                                n_total,
                                n_true,
                                n_acc,
                                n_acc_true,
                                w_total,
                                w_true,
                                w_acc,
                                w_acc_true,
                            ) = _compute_acceptance_metrics(
                                all_keys=keys_in_file_order,
                                accepted_keys=accepted_keys_direct,
                                key_to_weight=key_to_weight,
                                key_to_is_true=key_to_is_true,
                            )

                            # For direct, define accepted-weight as sum(weights of ext_* atoms that are true).
                            acc_w = float(w_acc)
                            cut_w = int(max(0, w_total - w_acc))

                            # Graph eval under remaining wall-time budget.
                            import math

                            remaining_eval = float(graph_eval_timeout_sec)
                            (
                                n_dags,
                                shd_avg,
                                f1_avg,
                                adj_f1_avg,
                                ah_f1_avg,
                                true_dag_in_compat,
                                n_cpdags,
                                true_cpdag_in_compat,
                                cpdag_shd_avg,
                                cpdag_f1_avg,
                                cpdag_adj_f1_avg,
                                cpdag_ah_f1_avg,
                                shd_best,
                                shd_worst,
                                f1_best,
                                f1_worst,
                                adj_f1_best,
                                adj_f1_worst,
                                ah_f1_best,
                                ah_f1_worst,
                                cpdag_shd_best,
                                cpdag_shd_worst,
                                cpdag_f1_best,
                                cpdag_f1_worst,
                                cpdag_adj_f1_best,
                                cpdag_adj_f1_worst,
                                cpdag_ah_f1_best,
                                cpdag_ah_f1_worst,
                                _timed_out_ge,
                                _ge_cache_hit,
                                eval_time_sec,
                            ) = _graph_eval_from_accepted_wall(
                                accepted_keys=accepted_keys_direct,
                                timeout_sec=float(remaining_eval),
                            )

                            wall = time.perf_counter() - t0
                            timed_out_total = bool(timed_out_direct)

                            eval_after_sec = None
                            try:
                                if build_sec is not None and solve_sec is not None:
                                    eval_wall = float(wall) - float(build_sec) - float(solve_sec)
                                    if eval_wall < 0.0:
                                        build_sec = max(0.0, float(build_sec) + float(eval_wall))
                                        eval_wall = 0.0

                                    eval_attr = float(eval_wall)
                                    try:
                                        if bool(_ge_cache_hit):
                                            eval_attr = max(eval_attr, float(eval_time_sec or 0.0))
                                    except Exception:
                                        pass
                                    eval_after_sec = float(eval_attr)
                            except Exception:
                                eval_after_sec = None

                            method_time_wc = None
                            try:
                                if build_sec is not None and solve_sec is not None and eval_after_sec is not None:
                                    method_time_wc = float(build_sec) + float(solve_sec) + float(eval_after_sec)
                            except Exception:
                                method_time_wc = None
                            
                            wc_results.append(
                                StrategyResult(
                                    rep=rep,
                                    encoding=enc,
                                    objective=obj,
                                    opt_strategy=opt,
                                    opt_mode=opt_mode,
                                    reification=reif,
                                    timed_out=bool(timed_out_total),
                                    wall_time_sec=float(wall),
                                    method_time_sec=method_time_wc,
                                    status=status_direct,
                                    objective_costs=costs_direct,
                                    cut_weight=cut_w,
                                    selected_count=int(n_acc),
                                    build_time_sec=build_sec,
                                    solve_time_sec=solve_sec,
                                    eval_time_sec=eval_after_sec,
                                    accepted_weight=acc_w,
                                    n_tests_total=int(n_total),
                                    n_tests_true=int(n_true),
                                    n_tests_accepted=int(n_acc),
                                    n_tests_accepted_true=int(n_acc_true),
                                    weight_total=int(w_total),
                                    weight_true=int(w_true),
                                    weight_accepted_true=int(w_acc_true),
                                    accepted_fact_f1=float(_fact_f1(n_true=int(n_true), n_acc=int(n_acc), n_acc_true=int(n_acc_true))),
                                    graph_eval_timed_out=bool(_timed_out_ge),
                                    graph_eval_cache_hit=bool(_ge_cache_hit),
                                    n_dags_compat=n_dags,
                                    true_dag_in_compat=int(true_dag_in_compat) if true_dag_in_compat is not None else None,
                                    n_cpdags_compat=int(n_cpdags) if n_cpdags is not None else None,
                                    true_cpdag_in_compat=int(true_cpdag_in_compat) if true_cpdag_in_compat is not None else None,
                                    shd_avg=shd_avg,
                                    f1_avg=f1_avg,
                                    adjacency_f1_avg=adj_f1_avg,
                                    arrowhead_f1_avg=ah_f1_avg,
                                    shd_best=shd_best,
                                    shd_worst=shd_worst,
                                    f1_best=f1_best,
                                    f1_worst=f1_worst,
                                    adjacency_f1_best=adj_f1_best,
                                    adjacency_f1_worst=adj_f1_worst,
                                    arrowhead_f1_best=ah_f1_best,
                                    arrowhead_f1_worst=ah_f1_worst,
                                    cpdag_shd_avg=cpdag_shd_avg,
                                    cpdag_f1_avg=cpdag_f1_avg,
                                    cpdag_adjacency_f1_avg=cpdag_adj_f1_avg,
                                    cpdag_arrowhead_f1_avg=cpdag_ah_f1_avg,
                                    cpdag_shd_best=cpdag_shd_best,
                                    cpdag_shd_worst=cpdag_shd_worst,
                                    cpdag_f1_best=cpdag_f1_best,
                                    cpdag_f1_worst=cpdag_f1_worst,
                                    cpdag_adjacency_f1_best=cpdag_adj_f1_best,
                                    cpdag_adjacency_f1_worst=cpdag_adj_f1_worst,
                                    cpdag_arrowhead_f1_best=cpdag_ah_f1_best,
                                    cpdag_arrowhead_f1_worst=cpdag_ah_f1_worst,
                                )
                            )
                            done_wc.add((rep, enc, obj, opt, opt_mode, reif))

                            _write_summary_snapshot(
                                path=partial_json,
                                is_partial=True,
                                total_weight_current=total_weight,
                            )

    # Write JSON summary
    out_json = out_dir / "summary.json"

    def _baseline_to_json(r: BaselineResult) -> dict[str, Any]:
        return {
            "rep": r.rep,
            "solver": r.solver,
            "opt_mode": r.opt_mode,
            "wall_time_sec": r.wall_time_sec,
            "method_time_sec": r.method_time_sec,
            "build_time_sec": r.build_time_sec,
            "solve_time_sec": r.solve_time_sec,
            "eval_time_sec": r.eval_time_sec,
            "timed_out": r.timed_out,
            "graph_eval_timed_out": r.graph_eval_timed_out,
            "graph_eval_cache_hit": r.graph_eval_cache_hit,
            "removed": r.removed,
            "models_after": r.models_after,
            "base_program_path": r.base_program_path,
            "accepted_weight": r.accepted_weight,
            "n_tests_total": r.n_tests_total,
            "n_tests_true": r.n_tests_true,
            "n_tests_accepted": r.n_tests_accepted,
            "n_tests_accepted_true": r.n_tests_accepted_true,
            "weight_total": r.weight_total,
            "weight_true": r.weight_true,
            "weight_accepted": int(r.accepted_weight) if r.accepted_weight is not None else None,
            "weight_accepted_true": r.weight_accepted_true,
            "accepted_fact_f1": r.accepted_fact_f1,
            "n_dags_compat": r.n_dags_compat,
            "true_dag_in_compat": r.true_dag_in_compat,
            "n_cpdags_compat": r.n_cpdags_compat,
            "true_cpdag_in_compat": r.true_cpdag_in_compat,
            "shd_avg": r.shd_avg,
            "f1_avg": r.f1_avg,
            "adjacency_f1_avg": r.adjacency_f1_avg,
            "arrowhead_f1_avg": r.arrowhead_f1_avg,
            "shd_best": r.shd_best,
            "shd_worst": r.shd_worst,
            "f1_best": r.f1_best,
            "f1_worst": r.f1_worst,
            "adjacency_f1_best": r.adjacency_f1_best,
            "adjacency_f1_worst": r.adjacency_f1_worst,
            "arrowhead_f1_best": r.arrowhead_f1_best,
            "arrowhead_f1_worst": r.arrowhead_f1_worst,
            "cpdag_shd_avg": r.cpdag_shd_avg,
            "cpdag_f1_avg": r.cpdag_f1_avg,
            "cpdag_adjacency_f1_avg": r.cpdag_adjacency_f1_avg,
            "cpdag_arrowhead_f1_avg": r.cpdag_arrowhead_f1_avg,
            "cpdag_shd_best": r.cpdag_shd_best,
            "cpdag_shd_worst": r.cpdag_shd_worst,
            "cpdag_f1_best": r.cpdag_f1_best,
            "cpdag_f1_worst": r.cpdag_f1_worst,
            "cpdag_adjacency_f1_best": r.cpdag_adjacency_f1_best,
            "cpdag_adjacency_f1_worst": r.cpdag_adjacency_f1_worst,
            "cpdag_arrowhead_f1_best": r.cpdag_arrowhead_f1_best,
            "cpdag_arrowhead_f1_worst": r.cpdag_arrowhead_f1_worst,
        }

    def _wc_to_json(r: StrategyResult) -> dict[str, Any]:
        # Intentionally omit objective_costs and cut_weight from the JSON output.
        return {
            "rep": r.rep,
            "encoding": r.encoding,
            "objective": r.objective,
            "opt_strategy": r.opt_strategy,
            "opt_mode": r.opt_mode,
            "reification": r.reification,
            "timed_out": r.timed_out,
            "graph_eval_timed_out": r.graph_eval_timed_out,
            "graph_eval_cache_hit": r.graph_eval_cache_hit,
            "wall_time_sec": r.wall_time_sec,
            "method_time_sec": r.method_time_sec,
            "status": r.status,
            "selected_count": r.selected_count,
            "build_time_sec": r.build_time_sec,
            "solve_time_sec": r.solve_time_sec,
            "eval_time_sec": r.eval_time_sec,
            "accepted_weight": r.accepted_weight,
            "n_tests_total": r.n_tests_total,
            "n_tests_true": r.n_tests_true,
            "n_tests_accepted": r.n_tests_accepted,
            "n_tests_accepted_true": r.n_tests_accepted_true,
            "weight_total": r.weight_total,
            "weight_true": r.weight_true,
            "weight_accepted": int(r.accepted_weight) if r.accepted_weight is not None else None,
            "weight_accepted_true": r.weight_accepted_true,
            "accepted_fact_f1": r.accepted_fact_f1,
            "n_dags_compat": r.n_dags_compat,
            "true_dag_in_compat": r.true_dag_in_compat,
            "n_cpdags_compat": r.n_cpdags_compat,
            "true_cpdag_in_compat": r.true_cpdag_in_compat,
            "shd_avg": r.shd_avg,
            "f1_avg": r.f1_avg,
            "adjacency_f1_avg": r.adjacency_f1_avg,
            "arrowhead_f1_avg": r.arrowhead_f1_avg,
            "shd_best": r.shd_best,
            "shd_worst": r.shd_worst,
            "f1_best": r.f1_best,
            "f1_worst": r.f1_worst,
            "adjacency_f1_best": r.adjacency_f1_best,
            "adjacency_f1_worst": r.adjacency_f1_worst,
            "arrowhead_f1_best": r.arrowhead_f1_best,
            "arrowhead_f1_worst": r.arrowhead_f1_worst,
            "cpdag_shd_avg": r.cpdag_shd_avg,
            "cpdag_f1_avg": r.cpdag_f1_avg,
            "cpdag_adjacency_f1_avg": r.cpdag_adjacency_f1_avg,
            "cpdag_arrowhead_f1_avg": r.cpdag_arrowhead_f1_avg,
            "cpdag_shd_best": r.cpdag_shd_best,
            "cpdag_shd_worst": r.cpdag_shd_worst,
            "cpdag_f1_best": r.cpdag_f1_best,
            "cpdag_f1_worst": r.cpdag_f1_worst,
            "cpdag_adjacency_f1_best": r.cpdag_adjacency_f1_best,
            "cpdag_adjacency_f1_worst": r.cpdag_adjacency_f1_worst,
            "cpdag_arrowhead_f1_best": r.cpdag_arrowhead_f1_best,
            "cpdag_arrowhead_f1_worst": r.cpdag_arrowhead_f1_worst,
        }

    out_json.write_text(
        json.dumps(
            {
                "n_nodes": n_nodes,
                "seed": seed,
                "timeout_sec": timeout_sec,
                "pct_wrong_facts": pct_wrong_facts,
                "wrong_fact_counts": wrong_fact_counts,
                "strategies": strategies,
                "opt_modes": opt_modes,
                "encodings": encodings,
                "objectives": objectives,
                "reifications": reifications,
                "reps": reps,
                "total_weight": total_weight,
                "baseline_results": [_baseline_to_json(r) for r in baseline_results],
                "wc_results": [_wc_to_json(r) for r in wc_results],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    def _norm_status(status: str, *, timed_out: bool) -> str:
        if timed_out:
            return "TIMEOUT"
        s = (status or "").strip().upper()
        if not s:
            return "UNKNOWN"
        if s.startswith("ERROR"):
            return "ERROR"
        if "UNSAT" in s:
            return "UNSAT"
        if "OPT" in s:
            return "OPT"
        if "SAT" in s:
            return "SAT"
        if "UNKNOWN" in s or "INTERRUPT" in s:
            return "UNKNOWN"
        return "UNKNOWN"

    def _assert_direct_mus_opt_eq() -> None:
        if not bool(getattr(args, "assert_direct_mus_opt_eq", False)):
            return

        # Group by config (rep + method fields excluding reif)
        by_key: dict[tuple[int, str, str, str, str], dict[str, StrategyResult]] = {}
        for r in wc_results:
            k = (int(r.rep), str(r.encoding), str(r.objective), str(r.opt_strategy), str(r.opt_mode))
            by_key.setdefault(k, {})[str(r.reification)] = r

        def _obj_sig(r: StrategyResult) -> tuple[int, ...] | None:
            # Prefer explicit fields if present.
            try:
                oc = getattr(r, "objective_costs", None)
            except Exception:
                oc = None
            if oc is not None:
                try:
                    return tuple(int(x) for x in oc)
                except Exception:
                    return None
            try:
                cw = getattr(r, "cut_weight", None)
            except Exception:
                cw = None
            if cw is not None:
                try:
                    return (int(cw),)
                except Exception:
                    return None
            # Fallback: cut_weight := total_weight - wA
            if total_weight is not None and r.accepted_weight is not None:
                try:
                    return (int(total_weight) - int(r.accepted_weight),)
                except Exception:
                    return None
            return None

        hard: list[tuple[tuple[int, str, str, str, str], StrategyResult, StrategyResult]] = []
        soft: list[
            tuple[
                tuple[int, str, str, str, str],
                StrategyResult,
                StrategyResult,
                str,
                bool,
                bool,
            ]
        ] = []
        soft_status_mismatches = 0
        soft_weight_mismatches = 0
        for k, d in sorted(by_key.items()):
            if "direct" not in d or "mus" not in d:
                continue
            r_dir = d["direct"]
            r_mus = d["mus"]
            s_dir = _norm_status(r_dir.status, timed_out=bool(r_dir.timed_out))
            s_mus = _norm_status(r_mus.status, timed_out=bool(r_mus.timed_out))
            sig_dir = _obj_sig(r_dir)
            sig_mus = _obj_sig(r_mus)

            if s_dir == "OPT" and s_mus == "OPT" and sig_dir != sig_mus:
                hard.append((k, r_dir, r_mus))
                continue

            status_diff = (s_dir != s_mus)
            w_diff = (
                (r_dir.accepted_weight is not None and r_mus.accepted_weight is not None)
                and (int(r_dir.accepted_weight) != int(r_mus.accepted_weight))
            )

            reasons: list[str] = []
            if status_diff:
                reasons.append(f"status {s_dir} vs {s_mus}")
            if w_diff:
                reasons.append(f"wA {int(r_dir.accepted_weight)} vs {int(r_mus.accepted_weight)}")
            if sig_dir is not None and sig_mus is not None and sig_dir != sig_mus:
                reasons.append(f"obj_sig {sig_dir} vs {sig_mus}")
            if reasons:
                if status_diff:
                    soft_status_mismatches += 1
                if w_diff:
                    soft_weight_mismatches += 1
                soft.append((k, r_dir, r_mus, "; ".join(reasons), status_diff, w_diff))

        if not hard and not soft:
            return

        def _resolve_wc_lp_path(
            root: Path,
            *,
            rep: int,
            encoding: str,
            reification: str,
            objective: str,
            opt_strategy: str,
            opt_mode: str,
        ) -> Path:
            fn = f"wc_{encoding}_{reification}_{objective}_{opt_strategy}_{opt_mode}_r{rep}.lp"
            candidates = [
                root / f"rep{rep}" / "lps" / fn,
                root / f"rep{rep}" / fn,
                root / fn,
            ]
            for p in candidates:
                try:
                    if p.exists():
                        return p
                except Exception:
                    pass
            return candidates[0]

        lines: list[str] = []
        lines.append("\n[assert-direct-mus-opt-eq] direct vs mus divergence report")
        lines.append(f"out_dir={out_dir}")
        lines.append(f"total_weight={total_weight}")
        lines.append(f"hard_mismatches={len(hard)}  (OPT/OPT but objective differs)")
        lines.append(f"soft_mismatches={len(soft)}  (any divergence outside OPT/OPT cost mismatch)")
        lines.append(f"soft_status_mismatches={soft_status_mismatches}  (status differs)")
        lines.append(f"soft_weight_mismatches={soft_weight_mismatches}  (accepted_weight differs)")
        lines.append("\nEach row shows the expected .lp files to inspect:")

        if hard:
            lines.append("\n=== HARD (OPT/OPT objective mismatch) ===")
            for (rep, enc, obj, strat, optm), r_dir, r_mus in hard:
                lp_dir = _resolve_wc_lp_path(
                    out_dir,
                    rep=rep,
                    encoding=enc,
                    reification="direct",
                    objective=obj,
                    opt_strategy=strat,
                    opt_mode=optm,
                )
                lp_mus = _resolve_wc_lp_path(
                    out_dir,
                    rep=rep,
                    encoding=enc,
                    reification="mus",
                    objective=obj,
                    opt_strategy=strat,
                    opt_mode=optm,
                )
                lines.append(
                    "\n"
                    + f"rep={rep} enc={enc} obj={obj} strategy={strat} optm={optm}\n"
                    + f"  direct: status=OPT to={int(bool(r_dir.timed_out))} wA={r_dir.accepted_weight} sig={_obj_sig(r_dir)}\n"
                    + f"  mus:    status=OPT to={int(bool(r_mus.timed_out))} wA={r_mus.accepted_weight} sig={_obj_sig(r_mus)}\n"
                    + f"  lp_direct={lp_dir}\n"
                    + f"  lp_mus   ={lp_mus}"
                )

        if soft:
            lines.append("\n=== SOFT (printing only weight mismatches) ===")
            soft_print = [t for t in soft if bool(t[5])]
            if not soft_print:
                lines.append("\n(no soft weight mismatches to print)")
            for (rep, enc, obj, strat, optm), r_dir, r_mus, reason, _sd, _wd in soft_print[:200]:
                lp_dir = _resolve_wc_lp_path(
                    out_dir,
                    rep=rep,
                    encoding=enc,
                    reification="direct",
                    objective=obj,
                    opt_strategy=strat,
                    opt_mode=optm,
                )
                lp_mus = _resolve_wc_lp_path(
                    out_dir,
                    rep=rep,
                    encoding=enc,
                    reification="mus",
                    objective=obj,
                    opt_strategy=strat,
                    opt_mode=optm,
                )
                lines.append(
                    "\n"
                    + f"rep={rep} enc={enc} obj={obj} strategy={strat} optm={optm}  ({reason})\n"
                    + f"  direct: status={_norm_status(r_dir.status, timed_out=bool(r_dir.timed_out))} to={int(bool(r_dir.timed_out))} wA={r_dir.accepted_weight} sig={_obj_sig(r_dir)} time={r_dir.wall_time_sec:.3f}\n"
                    + f"  mus:    status={_norm_status(r_mus.status, timed_out=bool(r_mus.timed_out))} to={int(bool(r_mus.timed_out))} wA={r_mus.accepted_weight} sig={_obj_sig(r_mus)} time={r_mus.wall_time_sec:.3f}\n"
                    + f"  lp_direct={lp_dir}\n"
                    + f"  lp_mus   ={lp_mus}"
                )
            if len(soft_print) > 200:
                lines.append(f"\n... truncated soft weight mismatches: showing 200 of {len(soft_print)}")

        raise SystemExit("\n".join(lines))

    # Optional: fail fast (after summary.json is written) if direct/mus diverge.
    _assert_direct_mus_opt_eq()

    def _is_number(x: Any) -> bool:
        try:
            import math

            return math.isfinite(float(x))
        except Exception:
            return False

    def _rank_map(
        method_to_value: dict[tuple[str, str, str, str, str], Any], *, higher_is_better: bool
    ) -> dict[tuple[str, str, str, str, str], float]:
        """Return per-method ranks (1..N, lower=better). Missing/non-finite values get worst rank."""
        methods_all = list(method_to_value.keys())
        n_methods = len(methods_all)

        present: list[tuple[tuple[str, str, str, str, str], float]] = []
        for m, v in method_to_value.items():
            if _is_number(v):
                present.append((m, float(v)))
        present.sort(key=lambda kv: kv[1], reverse=higher_is_better)

        ranks: dict[tuple[str, str, str, str, str], float] = {m: float(n_methods) for m in methods_all}
        i = 0
        while i < len(present):
            j = i + 1
            while j < len(present) and present[j][1] == present[i][1]:
                j += 1
            rank_lo = i + 1
            rank_hi = j
            rank_avg = (rank_lo + rank_hi) / 2.0
            for k in range(i, j):
                ranks[present[k][0]] = float(rank_avg)
            i = j
        return ranks

    def _write_metric_ranks_json(out_dir: Path) -> Path | None:
        """Persist per-method avg ranks (computed per-rep) to metric_ranks.json.
        """

        def _method_id(m: tuple[str, str, str, str, str]) -> str:
            return "|".join(m)

        # method -> rep -> result object (BaselineResult or StrategyResult)
        method_rep: dict[tuple[str, str, str, str, str], dict[int, Any]] = {}
        reps_seen: set[int] = set()

        for b in baseline_results:
            if int(b.rep) <= 0:
                continue
            enc_label = "inc" if b.solver == "causalaba_increm" else "base"
            optm = str(getattr(b, "opt_mode", None) or "optN")
            mkey = (enc_label, "-", "baseline", optm, "-")
            method_rep.setdefault(mkey, {})[int(b.rep)] = b
            reps_seen.add(int(b.rep))

        for r in wc_results:
            if int(r.rep) <= 0:
                continue
            mkey = (str(r.encoding), str(r.objective), str(r.opt_strategy), str(r.opt_mode), str(r.reification))
            method_rep.setdefault(mkey, {})[int(r.rep)] = r
            reps_seen.add(int(r.rep))

        methods_sorted = sorted(method_rep.keys())
        reps_sorted = sorted(reps_seen)
        if not methods_sorted or not reps_sorted:
            return None

        def _time_for_rank(it: Any) -> float | None:
            mt = getattr(it, "method_time_sec", None)
            if mt is not None:
                return mt
            b = getattr(it, "build_time_sec", None)
            s = getattr(it, "solve_time_sec", None)
            e = getattr(it, "eval_time_sec", None)
            if b is not None and s is not None and e is not None:
                return float(b) + float(s) + float(e)
            if b is not None and s is not None:
                return float(b) + float(s)
            return getattr(it, "wall_time_sec", None)

        def _time_build_solve_for_rank(it: Any) -> float | None:
            b = getattr(it, "build_time_sec", None)
            s = getattr(it, "solve_time_sec", None)
            if b is None or s is None:
                return None
            try:
                return float(b) + float(s)
            except Exception:
                return None

        def _solve_for_rank(it: Any) -> float | None:
            s = getattr(it, "solve_time_sec", None)
            if s is not None:
                return float(s)
            # Back-compat for any older in-memory row shapes.
            s2 = getattr(it, "solve_sec", None)
            return float(s2) if s2 is not None else None

        metric_defs: list[tuple[str, bool, Any]] = [
            ("timeR", False, _time_for_rank),
            ("timeBSR", False, _time_build_solve_for_rank),
            ("solveR", False, _solve_for_rank),
            ("toR", False, lambda it: int(bool(getattr(it, "timed_out", False))) if hasattr(it, "timed_out") else None),
            ("nATR", True, lambda it: getattr(it, "n_tests_accepted_true", None)),
            ("wATR", True, lambda it: getattr(it, "weight_accepted_true", None)),
            ("nDR", False, lambda it: getattr(it, "n_dags_compat", None)),
            ("shdR", False, lambda it: getattr(it, "shd_avg", None)),
            ("F1R", True, lambda it: getattr(it, "f1_avg", None)),
            ("adjR", True, lambda it: getattr(it, "adjacency_f1_avg", None)),
            ("ahR", True, lambda it: getattr(it, "arrowhead_f1_avg", None)),
            ("factR", True, lambda it: getattr(it, "accepted_fact_f1", None)),
            ("cshdR", False, lambda it: getattr(it, "cpdag_shd_avg", None)),
            ("cF1R", True, lambda it: getattr(it, "cpdag_f1_avg", None)),
            ("cadjR", True, lambda it: getattr(it, "cpdag_adjacency_f1_avg", None)),
            ("cahR", True, lambda it: getattr(it, "cpdag_arrowhead_f1_avg", None)),
        ]

        ranks_acc: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = {
            m: {name: [] for (name, _hib, _g) in metric_defs} for m in methods_sorted
        }

        for rep in reps_sorted:
            for name, higher_is_better, getter in metric_defs:
                vals: dict[tuple[str, str, str, str, str], Any] = {}
                for m in methods_sorted:
                    it = method_rep.get(m, {}).get(rep)
                    if it is None:
                        vals[m] = None
                        continue
                    try:
                        vals[m] = getter(it)
                    except Exception:
                        vals[m] = None
                rm = _rank_map(vals, higher_is_better=higher_is_better)
                for m, rnk in rm.items():
                    ranks_acc[m][name].append(float(rnk))

        def _mean(xs: list[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else float("nan")

        payload: dict[str, Any] = {
            "results_dir": str(out_dir),
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "reps": reps_sorted,
            "opt_modes": list(opt_modes),
            "metric_defs": [{"name": n, "higher_is_better": hib} for (n, hib, _g) in metric_defs],
            "methods": [
                {"id": _method_id(m), "enc": m[0], "obj": m[1], "strategy": m[2], "opt_mode": m[3], "reif": m[4]}
                for m in methods_sorted
            ],
            "ranks_avg": {
                _method_id(m): {name: _mean(ranks_acc[m][name]) for (name, _hib, _g) in metric_defs}
                for m in methods_sorted
            },
        }

        out_path = out_dir / "metric_ranks.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return out_path

    # Persist full metric ranks so wc_sweep_report can show them “natively”.
    try:
        ranks_path = _write_metric_ranks_json(out_dir)
        if ranks_path is not None:
            logging.info("Wrote metric ranks to %s", ranks_path)
    except Exception:
        logging.exception("Failed to write metric_ranks.json")

    def _stats(vals: list[float]) -> tuple[float, float, float]:
        vals_f = [float(v) for v in vals if v is not None]
        if not vals_f:
            return float("nan"), float("nan"), float("nan")
        return sum(vals_f) / len(vals_f), min(vals_f), max(vals_f)

    def _stdev(vals: list[float]) -> float:
        """Sample standard deviation (ddof=1); 0.0 for <=1 value.

        Ignores None.
        """
        vals_f = [float(v) for v in vals if v is not None]
        if len(vals_f) <= 1:
            return 0.0
        try:
            import statistics as _statistics

            return float(_statistics.stdev(vals_f))
        except Exception:
            # Fallback: naive two-pass stdev
            mu = sum(vals_f) / float(len(vals_f))
            var = sum((x - mu) ** 2 for x in vals_f) / float(max(1, len(vals_f) - 1))
            return float(var**0.5)

    def _print_baseline_table(rows: list[BaselineResult]) -> None:
        if not rows:
            print("baseline skipped")
            return
        # Keep compact and aligned with WC table for easy comparison.
        # NOTE: `mode` must be wide enough for 'causalaba_increm' to avoid shifting columns.
        mode_w = 16

        def _bw_pair(best: float | None, worst: float | None, *, prec: int = 3, width: int = 13) -> str:
            if best is None or worst is None:
                return "".rjust(width)
            return f"[{float(best):.{prec}f},{float(worst):.{prec}f}]".rjust(width)

        header = (
            f"{'rep':>3} {'enc':<4} {'obj':<4} {'strategy':<12} {'reif':<8} {'optm':<4} {'mode':<{mode_w}} "
            f"{'rm':>2} {'time':>7} {'build':>7} {'solve':>7} {'eval':>8} {'to':>3} "
            f"{'nT':>3} {'nTr':>3} {'nA':>4} {'nAT':>4} "
            f"{'wA':>11} {'wAT':>11} {'nD':>4} {'shd':>7} {'F1':>6} {'adjF1':>7} {'ahF1':>7} "
            f"{'shd[b,w]':>13} {'F1[b,w]':>13} {'adj[b,w]':>13} {'ah[b,w]':>13} "
            f"{'cshd[b,w]':>13} {'cF1[b,w]':>13} {'cadj[b,w]':>13} {'cah[b,w]':>13}"
        )
        print(header)
        for b in sorted(rows, key=lambda x: (int(x.rep), str(x.opt_mode or ""), str(x.solver))):
            enc = "inc" if b.solver == "causalaba_increm" else "base"
            w_acc = "" if b.accepted_weight is None else f"{b.accepted_weight:.0f}"
            nd = "" if b.n_dags_compat is None else f"{int(b.n_dags_compat):d}"
            shd = "" if b.shd_avg is None else f"{float(b.shd_avg):.3f}"
            f1 = "" if b.f1_avg is None else f"{float(b.f1_avg):.3f}"
            adjf1 = "" if b.adjacency_f1_avg is None else f"{float(b.adjacency_f1_avg):.3f}"
            ahf1 = "" if b.arrowhead_f1_avg is None else f"{float(b.arrowhead_f1_avg):.3f}"
            build_sec = "" if b.build_time_sec is None else f"{float(b.build_time_sec):.3f}"
            solve_sec = "" if b.solve_time_sec is None else f"{float(b.solve_time_sec):.3f}"
            eval_sec = "" if b.eval_time_sec is None else f"{float(b.eval_time_sec):.3f}"
            t_disp = float(b.method_time_sec) if b.method_time_sec is not None else float(b.wall_time_sec)
            shd_bw = _bw_pair(b.shd_best, b.shd_worst)
            f1_bw = _bw_pair(b.f1_best, b.f1_worst)
            adj_bw = _bw_pair(b.adjacency_f1_best, b.adjacency_f1_worst)
            ah_bw = _bw_pair(b.arrowhead_f1_best, b.arrowhead_f1_worst)
            cshd_bw = _bw_pair(b.cpdag_shd_best, b.cpdag_shd_worst)
            cf1_bw = _bw_pair(b.cpdag_f1_best, b.cpdag_f1_worst)
            cadj_bw = _bw_pair(b.cpdag_adjacency_f1_best, b.cpdag_adjacency_f1_worst)
            cah_bw = _bw_pair(b.cpdag_arrowhead_f1_best, b.cpdag_arrowhead_f1_worst)
            print(
                f"{b.rep:>3d} {enc:<4} {'-':<4} {'baseline':<12} {'-':<8} {str(b.opt_mode or ''):<4} {b.solver:<{mode_w}} "
                f"{b.removed:>2d} {t_disp:7.3f} {build_sec:>7} {solve_sec:>7} {eval_sec:>8} {int(b.timed_out):>3d} "
                f"{b.n_tests_total:>3d} {b.n_tests_true:>3d} {b.n_tests_accepted:>4d} {b.n_tests_accepted_true:>4d} "
                f"{w_acc:>11} {int(b.weight_accepted_true):>11d} "
                f"{nd:>4} {shd:>7} {f1:>6} {adjf1:>7} {ahf1:>7} "
                f"{shd_bw} {f1_bw} {adj_bw} {ah_bw} {cshd_bw} {cf1_bw} {cadj_bw} {cah_bw}"
            )

    def _print_baseline_summary(rows: list[BaselineResult]) -> None:
        if not rows:
            return
        header_avg = (
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'time_sec':>10} {'build':>10} {'solve':>10} {'eval':>10} {'timeouts':>9} "
            f"{'n_total':>10} {'n_true':>10} {'n_acc':>10} {'n_accT':>10} "
            f"{'w_acc':>12} {'w_accT':>12} {'n_dags':>10} {'shd':>8} {'F1':>8} {'adjF1':>8} {'ahF1':>8}"
        )
        header_minmax = (
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'time_sec':>14} {'build':>14} {'solve':>14} {'timeouts':>9} "
            f"{'n_total':>14} {'n_true':>14} {'n_acc':>14} {'n_accT':>14} "
            f"{'w_acc':>16} {'w_accT':>16} {'n_dags':>14} {'shd':>14} {'F1':>14} {'adjF1':>14} {'ahF1':>14}"
        )
        by_solver: dict[tuple[str, str], list[BaselineResult]] = {}
        for r in rows:
            by_solver.setdefault((r.solver, str(r.opt_mode or "optN")), []).append(r)

        def _solver_sort_key(k: tuple[str, str]) -> tuple[str, str]:
            solver, optm = k
            enc_label = "base" if solver == "causalaba" else "inc"
            return (enc_label, optm)

        print("\nBaseline summary (avg)")
        print(header_avg)
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            t_avg, t_min, t_max = _stats([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            timeouts = sum(1 for i in items if i.timed_out)
            n_total_avg, _, _ = _stats([i.n_tests_total for i in items])
            n_true_avg, _, _ = _stats([i.n_tests_true for i in items])
            n_acc_avg, n_acc_min, n_acc_max = _stats([i.n_tests_accepted for i in items])
            n_accT_avg, n_accT_min, n_accT_max = _stats([i.n_tests_accepted_true for i in items])
            w_acc_avg = float("nan")
            w_accT_avg = float("nan")
            w_acc_vals = [i.accepted_weight for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            if w_acc_vals:
                w_acc_avg, _, _ = _stats(w_acc_vals)
            if w_accT_vals:
                w_accT_avg, _, _ = _stats(w_accT_vals)
            enc_label = "base" if solver_key == "causalaba" else "inc"
            build_avg = _stats([i.build_time_sec for i in items if i.build_time_sec is not None])[0]
            solve_avg = _stats([i.solve_time_sec for i in items if i.solve_time_sec is not None])[0]
            eval_avg = _stats([i.eval_time_sec for i in items if i.eval_time_sec is not None])[0]

            n_dags_avg, _, _ = _stats([float(i.n_dags_compat) for i in items if i.n_dags_compat is not None])
            shd_avg, _, _ = _stats([float(i.shd_avg) for i in items if i.shd_avg is not None])
            f1_avg, _, _ = _stats([float(i.f1_avg) for i in items if i.f1_avg is not None])
            adj_f1_avg, _, _ = _stats([float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None])
            ah_f1_avg, _, _ = _stats([float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None])
            nd_disp = "" if n_dags_avg != n_dags_avg else f"{n_dags_avg:0.1f}"
            shd_disp = "" if shd_avg != shd_avg else f"{shd_avg:0.3f}"
            f1_disp = "" if f1_avg != f1_avg else f"{f1_avg:0.3f}"
            adj_f1_disp = "" if adj_f1_avg != adj_f1_avg else f"{adj_f1_avg:0.3f}"
            ah_f1_disp = "" if ah_f1_avg != ah_f1_avg else f"{ah_f1_avg:0.3f}"

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{t_avg:>10.3f} {build_avg:>10.3f} {solve_avg:>10.3f} {eval_avg:>10.3f} {timeouts:>9} "
                f"{n_total_avg:>10.1f} {n_true_avg:>10.1f} {n_acc_avg:>10.1f} {n_accT_avg:>10.1f} "
                f"{(int(round(w_acc_avg)) if w_acc_vals else ''):>12} {int(round(w_accT_avg)):>12} "
                f"{nd_disp:>10} {shd_disp:>8} {f1_disp:>8} {adj_f1_disp:>8} {ah_f1_disp:>8}"
            )

        print("\nBaseline summary (std)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'time_sd':>10} {'build_sd':>10} {'solve_sd':>10} {'eval_sd':>10} {'timeouts':>9} "
            f"{'n_total':>10} {'n_true':>10} {'n_acc':>10} {'n_accT':>10} "
            f"{'w_acc':>12} {'w_accT':>12} {'n_dags':>10} {'shd':>8} {'F1':>8} {'adjF1':>8} {'ahF1':>8}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"
            t_sd = _stdev([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            build_sd = _stdev([i.build_time_sec for i in items if i.build_time_sec is not None])
            solve_sd = _stdev([i.solve_time_sec for i in items if i.solve_time_sec is not None])
            eval_sd = _stdev([i.eval_time_sec for i in items if i.eval_time_sec is not None])
            timeouts = sum(1 for i in items if i.timed_out)

            n_total_sd = _stdev([float(i.n_tests_total) for i in items])
            n_true_sd = _stdev([float(i.n_tests_true) for i in items])
            n_acc_sd = _stdev([float(i.n_tests_accepted) for i in items])
            n_accT_sd = _stdev([float(i.n_tests_accepted_true) for i in items])

            w_acc_vals = [float(i.accepted_weight) for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            w_acc_sd = _stdev(w_acc_vals) if w_acc_vals else float("nan")
            w_accT_sd = _stdev(w_accT_vals) if w_accT_vals else float("nan")

            n_dags_sd = _stdev([float(i.n_dags_compat) for i in items if i.n_dags_compat is not None])
            shd_sd = _stdev([float(i.shd_avg) for i in items if i.shd_avg is not None])
            f1_sd = _stdev([float(i.f1_avg) for i in items if i.f1_avg is not None])
            adj_f1_sd = _stdev([float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None])
            ah_f1_sd = _stdev([float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None])

            def _disp(x: float, fmt: str) -> str:
                return "" if x != x else format(x, fmt)

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{t_sd:>10.3f} {build_sd:>10.3f} {solve_sd:>10.3f} {eval_sd:>10.3f} {timeouts:>9} "
                f"{n_total_sd:>10.1f} {n_true_sd:>10.1f} {n_acc_sd:>10.1f} {n_accT_sd:>10.1f} "
                f"{_disp(w_acc_sd,'0.1f'):>12} {(_disp(w_accT_sd,'0.1f')):>12} "
                f"{_disp(n_dags_sd,'0.1f'):>10} {(_disp(shd_sd,'0.3f')):>8} {(_disp(f1_sd,'0.3f')):>8} {(_disp(adj_f1_sd,'0.3f')):>8} {(_disp(ah_f1_sd,'0.3f')):>8}"
            )

        # Facts + CPDAG summary (avg)
        print("\nBaseline summary (avg) (facts + CPDAG)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'factF1':>10} {'inT':>6} {'inCP':>6} {'nCPD':>8} {'cshd':>10} {'cF1':>10} {'cadjF1':>10} {'cahF1':>10}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"
            fact_f1_avg, _, _ = _stats([float(i.accepted_fact_f1) for i in items if i.accepted_fact_f1 is not None])
            inT_avg, _, _ = _stats([float(i.true_dag_in_compat) for i in items if i.true_dag_in_compat is not None])
            inCP_avg, _, _ = _stats([float(i.true_cpdag_in_compat) for i in items if i.true_cpdag_in_compat is not None])
            ncpd_avg, _, _ = _stats([float(i.n_cpdags_compat) for i in items if i.n_cpdags_compat is not None])
            cshd_avg, _, _ = _stats([float(i.cpdag_shd_avg) for i in items if i.cpdag_shd_avg is not None])
            cf1_avg, _, _ = _stats([float(i.cpdag_f1_avg) for i in items if i.cpdag_f1_avg is not None])
            cadj_avg, _, _ = _stats([float(i.cpdag_adjacency_f1_avg) for i in items if i.cpdag_adjacency_f1_avg is not None])
            cah_avg, _, _ = _stats([float(i.cpdag_arrowhead_f1_avg) for i in items if i.cpdag_arrowhead_f1_avg is not None])

            def _disp(x: float, fmt: str) -> str:
                return "" if x != x else format(x, fmt)

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{_disp(fact_f1_avg,'0.3f'):>10} {(_disp(inT_avg,'0.3f') if inT_avg == inT_avg else ''):>6} {(_disp(inCP_avg,'0.3f') if inCP_avg == inCP_avg else ''):>6} "
                f"{_disp(ncpd_avg,'0.1f'):>8} {(_disp(cshd_avg,'0.3f')):>10} {(_disp(cf1_avg,'0.3f')):>10} {(_disp(cadj_avg,'0.3f')):>10} {(_disp(cah_avg,'0.3f')):>10}"
            )

        print("\nBaseline summary (std) (facts + CPDAG)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'factF1_sd':>10} {'inT_sd':>8} {'inCP_sd':>8} {'nCPD_sd':>10} {'cshd_sd':>10} {'cF1_sd':>10} {'cadj_sd':>10} {'cah_sd':>10}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"

            fact_f1_sd = _stdev([float(i.accepted_fact_f1) for i in items if i.accepted_fact_f1 is not None])
            inT_sd = _stdev([float(i.true_dag_in_compat) for i in items if i.true_dag_in_compat is not None])
            inCP_sd = _stdev([float(i.true_cpdag_in_compat) for i in items if i.true_cpdag_in_compat is not None])
            ncpd_sd = _stdev([float(i.n_cpdags_compat) for i in items if i.n_cpdags_compat is not None])
            cshd_sd = _stdev([float(i.cpdag_shd_avg) for i in items if i.cpdag_shd_avg is not None])
            cf1_sd = _stdev([float(i.cpdag_f1_avg) for i in items if i.cpdag_f1_avg is not None])
            cadj_sd = _stdev([float(i.cpdag_adjacency_f1_avg) for i in items if i.cpdag_adjacency_f1_avg is not None])
            cah_sd = _stdev([float(i.cpdag_arrowhead_f1_avg) for i in items if i.cpdag_arrowhead_f1_avg is not None])

            def _disp(x: float, fmt: str) -> str:
                return "" if x != x else format(x, fmt)

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{_disp(fact_f1_sd,'0.3f'):>10} {(_disp(inT_sd,'0.3f') if inT_sd == inT_sd else ''):>8} {(_disp(inCP_sd,'0.3f') if inCP_sd == inCP_sd else ''):>8} "
                f"{_disp(ncpd_sd,'0.1f'):>10} {(_disp(cshd_sd,'0.3f')):>10} {(_disp(cf1_sd,'0.3f')):>10} {(_disp(cadj_sd,'0.3f')):>10} {(_disp(cah_sd,'0.3f')):>10}"
            )

        print("\nBaseline summary (min/max)")
        # Split min/max into two compact blocks to avoid terminal wrapping.
        print("(times + counts)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'time':>14} {'build':>14} {'solve':>14} {'eval':>14} {'to':>4} {'nT':>12} {'nTr':>12} {'nA':>12} {'nAT':>12}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            _, t_min, t_max = _stats([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            timeouts = sum(1 for i in items if i.timed_out)
            _, b_min, b_max = _stats([i.build_time_sec for i in items if i.build_time_sec is not None])
            _, s_min, s_max = _stats([i.solve_time_sec for i in items if i.solve_time_sec is not None])
            _, e_min, e_max = _stats([i.eval_time_sec for i in items if i.eval_time_sec is not None])
            _, n_total_min, n_total_max = _stats([i.n_tests_total for i in items])
            _, n_true_min, n_true_max = _stats([i.n_tests_true for i in items])
            _, n_acc_min, n_acc_max = _stats([i.n_tests_accepted for i in items])
            _, n_accT_min, n_accT_max = _stats([i.n_tests_accepted_true for i in items])
            w_acc_vals = [i.accepted_weight for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            _, w_acc_min, w_acc_max = _stats(w_acc_vals) if w_acc_vals else (float("nan"), float("nan"), float("nan"))
            _, w_accT_min, w_accT_max = _stats(w_accT_vals) if w_accT_vals else (float("nan"), float("nan"), float("nan"))
            enc_label = "base" if solver_key == "causalaba" else "inc"
            time_range = f"[{t_min:.3f},{t_max:.3f}]".rjust(14)
            build_range = f"[{b_min:.3f},{b_max:.3f}]".rjust(14)
            solve_range = f"[{s_min:.3f},{s_max:.3f}]".rjust(14)
            eval_range = f"[{e_min:.3f},{e_max:.3f}]".rjust(14)
            n_total_range = f"[{n_total_min:.1f},{n_total_max:.1f}]".rjust(12)
            n_true_range = f"[{n_true_min:.1f},{n_true_max:.1f}]".rjust(12)
            n_acc_range = f"[{n_acc_min:.1f},{n_acc_max:.1f}]".rjust(12)
            n_accT_range = f"[{n_accT_min:.1f},{n_accT_max:.1f}]".rjust(12)

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{time_range} {build_range} {solve_range} {eval_range} {timeouts:>4d} {n_total_range} {n_true_range} {n_acc_range} {n_accT_range}"
            )

        print("\n(weights + graph)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'wA':>22} {'wAT':>22} {'nD':>12} {'shd':>14} {'F1':>12} {'adjF1':>12} {'ahF1':>12}"
        )

        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"
            w_acc_vals = [i.accepted_weight for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            _, w_acc_min, w_acc_max = _stats(w_acc_vals) if w_acc_vals else (float("nan"), float("nan"), float("nan"))
            _, w_accT_min, w_accT_max = _stats(w_accT_vals) if w_accT_vals else (float("nan"), float("nan"), float("nan"))
            w_acc_range = (f"[{w_acc_min:.0f},{w_acc_max:.0f}]".rjust(22) if w_acc_vals else "".rjust(22))
            w_accT_range = (f"[{w_accT_min:.0f},{w_accT_max:.0f}]".rjust(22) if w_accT_vals else "".rjust(22))

            nd_vals = [float(i.n_dags_compat) for i in items if i.n_dags_compat is not None]
            shd_vals = [float(i.shd_avg) for i in items if i.shd_avg is not None]
            f1_vals = [float(i.f1_avg) for i in items if i.f1_avg is not None]
            adj_f1_vals = [float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None]
            ah_f1_vals = [float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None]
            _, nd_min, nd_max = _stats(nd_vals) if nd_vals else (float("nan"), float("nan"), float("nan"))
            _, shd_min, shd_max = _stats(shd_vals) if shd_vals else (float("nan"), float("nan"), float("nan"))
            _, f1_min, f1_max = _stats(f1_vals) if f1_vals else (float("nan"), float("nan"), float("nan"))
            _, adj_f1_min, adj_f1_max = _stats(adj_f1_vals) if adj_f1_vals else (float("nan"), float("nan"), float("nan"))
            _, ah_f1_min, ah_f1_max = _stats(ah_f1_vals) if ah_f1_vals else (float("nan"), float("nan"), float("nan"))
            nd_range = (f"[{nd_min:.1f},{nd_max:.1f}]".rjust(12) if nd_vals else "".rjust(12))
            shd_range = (f"[{shd_min:.3f},{shd_max:.3f}]".rjust(14) if shd_vals else "".rjust(14))
            f1_range = (f"[{f1_min:.3f},{f1_max:.3f}]".rjust(12) if f1_vals else "".rjust(12))
            adj_f1_range = (f"[{adj_f1_min:.3f},{adj_f1_max:.3f}]".rjust(12) if adj_f1_vals else "".rjust(12))
            ah_f1_range = (f"[{ah_f1_min:.3f},{ah_f1_max:.3f}]".rjust(12) if ah_f1_vals else "".rjust(12))

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{w_acc_range} {w_accT_range} {nd_range} {shd_range} {f1_range} {adj_f1_range} {ah_f1_range}"
            )

        print("\n(facts + CPDAG)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'factF1':>14} {'inT':>10} {'inCP':>10} {'nCPD':>12} {'cshd':>14} {'cF1':>14} {'cadjF1':>14} {'cahF1':>14}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"
            f1_vals = [float(i.accepted_fact_f1) for i in items if i.accepted_fact_f1 is not None]
            in_vals = [float(i.true_dag_in_compat) for i in items if i.true_dag_in_compat is not None]
            incp_vals = [float(i.true_cpdag_in_compat) for i in items if i.true_cpdag_in_compat is not None]
            ncpd_vals = [float(i.n_cpdags_compat) for i in items if i.n_cpdags_compat is not None]
            cshd_vals = [float(i.cpdag_shd_avg) for i in items if i.cpdag_shd_avg is not None]
            cf1_vals = [float(i.cpdag_f1_avg) for i in items if i.cpdag_f1_avg is not None]
            cadj_vals = [float(i.cpdag_adjacency_f1_avg) for i in items if i.cpdag_adjacency_f1_avg is not None]
            cah_vals = [float(i.cpdag_arrowhead_f1_avg) for i in items if i.cpdag_arrowhead_f1_avg is not None]

            _, f1_min, f1_max = _stats(f1_vals) if f1_vals else (float("nan"), float("nan"), float("nan"))
            _, in_min, in_max = _stats(in_vals) if in_vals else (float("nan"), float("nan"), float("nan"))
            _, incp_min, incp_max = _stats(incp_vals) if incp_vals else (float("nan"), float("nan"), float("nan"))
            _, ncpd_min, ncpd_max = _stats(ncpd_vals) if ncpd_vals else (float("nan"), float("nan"), float("nan"))
            _, cshd_min, cshd_max = _stats(cshd_vals) if cshd_vals else (float("nan"), float("nan"), float("nan"))
            _, cf1_min, cf1_max = _stats(cf1_vals) if cf1_vals else (float("nan"), float("nan"), float("nan"))
            _, cadj_min, cadj_max = _stats(cadj_vals) if cadj_vals else (float("nan"), float("nan"), float("nan"))
            _, cah_min, cah_max = _stats(cah_vals) if cah_vals else (float("nan"), float("nan"), float("nan"))

            factF1_range = (f"[{f1_min:.3f},{f1_max:.3f}]".rjust(14) if f1_vals else "".rjust(14))
            inT_range = (f"[{int(in_min):d},{int(in_max):d}]".rjust(10) if in_vals else "".rjust(10))
            inCP_range = (f"[{int(incp_min):d},{int(incp_max):d}]".rjust(10) if incp_vals else "".rjust(10))
            ncpd_range = (f"[{int(ncpd_min):d},{int(ncpd_max):d}]".rjust(12) if ncpd_vals else "".rjust(12))
            cshd_range = (f"[{cshd_min:.3f},{cshd_max:.3f}]".rjust(14) if cshd_vals else "".rjust(14))
            cF1_range = (f"[{cf1_min:.3f},{cf1_max:.3f}]".rjust(14) if cf1_vals else "".rjust(14))
            cadj_range = (f"[{cadj_min:.3f},{cadj_max:.3f}]".rjust(14) if cadj_vals else "".rjust(14))
            cah_range = (f"[{cah_min:.3f},{cah_max:.3f}]".rjust(14) if cah_vals else "".rjust(14))

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{factF1_range} {inT_range} {inCP_range} {ncpd_range} {cshd_range} {cF1_range} {cadj_range} {cah_range}"
            )

        print("\n(graph best/worst over compatible DAGs) (min/max over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
            f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"

            shd_best_vals = [float(i.shd_best) for i in items if i.shd_best is not None]
            shd_worst_vals = [float(i.shd_worst) for i in items if i.shd_worst is not None]
            f1_best_vals = [float(i.f1_best) for i in items if i.f1_best is not None]
            f1_worst_vals = [float(i.f1_worst) for i in items if i.f1_worst is not None]
            adj_best_vals = [float(i.adjacency_f1_best) for i in items if i.adjacency_f1_best is not None]
            adj_worst_vals = [float(i.adjacency_f1_worst) for i in items if i.adjacency_f1_worst is not None]
            ah_best_vals = [float(i.arrowhead_f1_best) for i in items if i.arrowhead_f1_best is not None]
            ah_worst_vals = [float(i.arrowhead_f1_worst) for i in items if i.arrowhead_f1_worst is not None]

            _, shd_best_min, shd_best_max = _stats(shd_best_vals) if shd_best_vals else (float("nan"), float("nan"), float("nan"))
            _, shd_worst_min, shd_worst_max = _stats(shd_worst_vals) if shd_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, f1_best_min, f1_best_max = _stats(f1_best_vals) if f1_best_vals else (float("nan"), float("nan"), float("nan"))
            _, f1_worst_min, f1_worst_max = _stats(f1_worst_vals) if f1_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, adj_best_min, adj_best_max = _stats(adj_best_vals) if adj_best_vals else (float("nan"), float("nan"), float("nan"))
            _, adj_worst_min, adj_worst_max = _stats(adj_worst_vals) if adj_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, ah_best_min, ah_best_max = _stats(ah_best_vals) if ah_best_vals else (float("nan"), float("nan"), float("nan"))
            _, ah_worst_min, ah_worst_max = _stats(ah_worst_vals) if ah_worst_vals else (float("nan"), float("nan"), float("nan"))

            shd_best_range = (f"[{shd_best_min:.3f},{shd_best_max:.3f}]".rjust(14) if shd_best_vals else "".rjust(14))
            shd_worst_range = (f"[{shd_worst_min:.3f},{shd_worst_max:.3f}]".rjust(14) if shd_worst_vals else "".rjust(14))
            f1_best_range = (f"[{f1_best_min:.3f},{f1_best_max:.3f}]".rjust(14) if f1_best_vals else "".rjust(14))
            f1_worst_range = (f"[{f1_worst_min:.3f},{f1_worst_max:.3f}]".rjust(14) if f1_worst_vals else "".rjust(14))
            adj_best_range = (f"[{adj_best_min:.3f},{adj_best_max:.3f}]".rjust(14) if adj_best_vals else "".rjust(14))
            adj_worst_range = (f"[{adj_worst_min:.3f},{adj_worst_max:.3f}]".rjust(14) if adj_worst_vals else "".rjust(14))
            ah_best_range = (f"[{ah_best_min:.3f},{ah_best_max:.3f}]".rjust(14) if ah_best_vals else "".rjust(14))
            ah_worst_range = (f"[{ah_worst_min:.3f},{ah_worst_max:.3f}]".rjust(14) if ah_worst_vals else "".rjust(14))

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{shd_best_range} {shd_worst_range} {f1_best_range} {f1_worst_range} "
                f"{adj_best_range} {adj_worst_range} {ah_best_range} {ah_worst_range}"
            )

        print("\n(graph best/worst over compatible DAGs) (avg over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
            f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"

            shd_best_vals = [float(i.shd_best) for i in items if i.shd_best is not None]
            shd_worst_vals = [float(i.shd_worst) for i in items if i.shd_worst is not None]
            f1_best_vals = [float(i.f1_best) for i in items if i.f1_best is not None]
            f1_worst_vals = [float(i.f1_worst) for i in items if i.f1_worst is not None]
            adj_best_vals = [float(i.adjacency_f1_best) for i in items if i.adjacency_f1_best is not None]
            adj_worst_vals = [float(i.adjacency_f1_worst) for i in items if i.adjacency_f1_worst is not None]
            ah_best_vals = [float(i.arrowhead_f1_best) for i in items if i.arrowhead_f1_best is not None]
            ah_worst_vals = [float(i.arrowhead_f1_worst) for i in items if i.arrowhead_f1_worst is not None]

            shd_best_avg, _, _ = _stats(shd_best_vals) if shd_best_vals else (float("nan"), float("nan"), float("nan"))
            shd_worst_avg, _, _ = _stats(shd_worst_vals) if shd_worst_vals else (float("nan"), float("nan"), float("nan"))
            f1_best_avg, _, _ = _stats(f1_best_vals) if f1_best_vals else (float("nan"), float("nan"), float("nan"))
            f1_worst_avg, _, _ = _stats(f1_worst_vals) if f1_worst_vals else (float("nan"), float("nan"), float("nan"))
            adj_best_avg, _, _ = _stats(adj_best_vals) if adj_best_vals else (float("nan"), float("nan"), float("nan"))
            adj_worst_avg, _, _ = _stats(adj_worst_vals) if adj_worst_vals else (float("nan"), float("nan"), float("nan"))
            ah_best_avg, _, _ = _stats(ah_best_vals) if ah_best_vals else (float("nan"), float("nan"), float("nan"))
            ah_worst_avg, _, _ = _stats(ah_worst_vals) if ah_worst_vals else (float("nan"), float("nan"), float("nan"))

            def _disp(x: float, fmt: str) -> str:
                return "" if x != x else format(x, fmt)

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{_disp(shd_best_avg,'0.3f'):>14} {_disp(shd_worst_avg,'0.3f'):>14} "
                f"{_disp(f1_best_avg,'0.3f'):>14} {_disp(f1_worst_avg,'0.3f'):>14} "
                f"{_disp(adj_best_avg,'0.3f'):>14} {_disp(adj_worst_avg,'0.3f'):>14} "
                f"{_disp(ah_best_avg,'0.3f'):>14} {_disp(ah_worst_avg,'0.3f'):>14}"
            )

        print("\n(CPDAG best/worst over compatible CPDAGs) (min/max over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
            f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"

            cshd_best_vals = [float(i.cpdag_shd_best) for i in items if i.cpdag_shd_best is not None]
            cshd_worst_vals = [float(i.cpdag_shd_worst) for i in items if i.cpdag_shd_worst is not None]
            cf1_best_vals = [float(i.cpdag_f1_best) for i in items if i.cpdag_f1_best is not None]
            cf1_worst_vals = [float(i.cpdag_f1_worst) for i in items if i.cpdag_f1_worst is not None]
            cadj_best_vals = [float(i.cpdag_adjacency_f1_best) for i in items if i.cpdag_adjacency_f1_best is not None]
            cadj_worst_vals = [float(i.cpdag_adjacency_f1_worst) for i in items if i.cpdag_adjacency_f1_worst is not None]
            cah_best_vals = [float(i.cpdag_arrowhead_f1_best) for i in items if i.cpdag_arrowhead_f1_best is not None]
            cah_worst_vals = [float(i.cpdag_arrowhead_f1_worst) for i in items if i.cpdag_arrowhead_f1_worst is not None]

            _, cshd_best_min, cshd_best_max = _stats(cshd_best_vals) if cshd_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cshd_worst_min, cshd_worst_max = _stats(cshd_worst_vals) if cshd_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, cf1_best_min, cf1_best_max = _stats(cf1_best_vals) if cf1_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cf1_worst_min, cf1_worst_max = _stats(cf1_worst_vals) if cf1_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, cadj_best_min, cadj_best_max = _stats(cadj_best_vals) if cadj_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cadj_worst_min, cadj_worst_max = _stats(cadj_worst_vals) if cadj_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, cah_best_min, cah_best_max = _stats(cah_best_vals) if cah_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cah_worst_min, cah_worst_max = _stats(cah_worst_vals) if cah_worst_vals else (float("nan"), float("nan"), float("nan"))

            cshd_best_range = (f"[{cshd_best_min:.3f},{cshd_best_max:.3f}]".rjust(14) if cshd_best_vals else "".rjust(14))
            cshd_worst_range = (f"[{cshd_worst_min:.3f},{cshd_worst_max:.3f}]".rjust(14) if cshd_worst_vals else "".rjust(14))
            cf1_best_range = (f"[{cf1_best_min:.3f},{cf1_best_max:.3f}]".rjust(14) if cf1_best_vals else "".rjust(14))
            cf1_worst_range = (f"[{cf1_worst_min:.3f},{cf1_worst_max:.3f}]".rjust(14) if cf1_worst_vals else "".rjust(14))
            cadj_best_range = (f"[{cadj_best_min:.3f},{cadj_best_max:.3f}]".rjust(14) if cadj_best_vals else "".rjust(14))
            cadj_worst_range = (f"[{cadj_worst_min:.3f},{cadj_worst_max:.3f}]".rjust(14) if cadj_worst_vals else "".rjust(14))
            cah_best_range = (f"[{cah_best_min:.3f},{cah_best_max:.3f}]".rjust(14) if cah_best_vals else "".rjust(14))
            cah_worst_range = (f"[{cah_worst_min:.3f},{cah_worst_max:.3f}]".rjust(14) if cah_worst_vals else "".rjust(14))

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{cshd_best_range} {cshd_worst_range} {cf1_best_range} {cf1_worst_range} "
                f"{cadj_best_range} {cadj_worst_range} {cah_best_range} {cah_worst_range}"
            )

        print("\n(CPDAG best/worst over compatible CPDAGs) (avg over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
            f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
        )
        for solver_key, optm in sorted(by_solver.keys(), key=_solver_sort_key):
            items = by_solver[(solver_key, optm)]
            enc_label = "base" if solver_key == "causalaba" else "inc"

            cshd_best_vals = [float(i.cpdag_shd_best) for i in items if i.cpdag_shd_best is not None]
            cshd_worst_vals = [float(i.cpdag_shd_worst) for i in items if i.cpdag_shd_worst is not None]
            cf1_best_vals = [float(i.cpdag_f1_best) for i in items if i.cpdag_f1_best is not None]
            cf1_worst_vals = [float(i.cpdag_f1_worst) for i in items if i.cpdag_f1_worst is not None]
            cadj_best_vals = [float(i.cpdag_adjacency_f1_best) for i in items if i.cpdag_adjacency_f1_best is not None]
            cadj_worst_vals = [float(i.cpdag_adjacency_f1_worst) for i in items if i.cpdag_adjacency_f1_worst is not None]
            cah_best_vals = [float(i.cpdag_arrowhead_f1_best) for i in items if i.cpdag_arrowhead_f1_best is not None]
            cah_worst_vals = [float(i.cpdag_arrowhead_f1_worst) for i in items if i.cpdag_arrowhead_f1_worst is not None]

            cshd_best_avg, _, _ = _stats(cshd_best_vals) if cshd_best_vals else (float("nan"), float("nan"), float("nan"))
            cshd_worst_avg, _, _ = _stats(cshd_worst_vals) if cshd_worst_vals else (float("nan"), float("nan"), float("nan"))
            cf1_best_avg, _, _ = _stats(cf1_best_vals) if cf1_best_vals else (float("nan"), float("nan"), float("nan"))
            cf1_worst_avg, _, _ = _stats(cf1_worst_vals) if cf1_worst_vals else (float("nan"), float("nan"), float("nan"))
            cadj_best_avg, _, _ = _stats(cadj_best_vals) if cadj_best_vals else (float("nan"), float("nan"), float("nan"))
            cadj_worst_avg, _, _ = _stats(cadj_worst_vals) if cadj_worst_vals else (float("nan"), float("nan"), float("nan"))
            cah_best_avg, _, _ = _stats(cah_best_vals) if cah_best_vals else (float("nan"), float("nan"), float("nan"))
            cah_worst_avg, _, _ = _stats(cah_worst_vals) if cah_worst_vals else (float("nan"), float("nan"), float("nan"))

            def _disp(x: float, fmt: str) -> str:
                return "" if x != x else format(x, fmt)

            print(
                f"{enc_label:<4} {'-':<4} {'baseline':<12} {optm:<4} {'-':<8} {len(items):>6} "
                f"{_disp(cshd_best_avg,'0.3f'):>14} {_disp(cshd_worst_avg,'0.3f'):>14} "
                f"{_disp(cf1_best_avg,'0.3f'):>14} {_disp(cf1_worst_avg,'0.3f'):>14} "
                f"{_disp(cadj_best_avg,'0.3f'):>14} {_disp(cadj_worst_avg,'0.3f'):>14} "
                f"{_disp(cah_best_avg,'0.3f'):>14} {_disp(cah_worst_avg,'0.3f'):>14}"
            )

    def _print_wc_table(rows: list[StrategyResult]) -> None:
        if not rows:
            print("no WC results")
            return

        def _bw_pair(best: float | None, worst: float | None, *, prec: int = 3, width: int = 13) -> str:
            if best is None or worst is None:
                return "".rjust(width)
            return f"[{float(best):.{prec}f},{float(worst):.{prec}f}]".rjust(width)

        print(
            " rep enc  obj  strategy     optm reif     mode         rm    time  build  solve   eval   to "
            " nT  nTr   nA  nAT         wA        wAT  nD    shd    F1  adjF1   ahF1"
            "      shd[b,w]       F1[b,w]      adj[b,w]       ah[b,w]     cshd[b,w]      cF1[b,w]     cadj[b,w]      cah[b,w]"
        )
        for r in sorted(rows, key=lambda x: (x.encoding, x.objective, x.opt_strategy, x.opt_mode, x.reification, x.rep)):
            w_acc = "" if r.accepted_weight is None else f"{r.accepted_weight:.0f}"
            nd = "" if r.n_dags_compat is None else f"{int(r.n_dags_compat):d}"
            shd = "" if r.shd_avg is None else f"{float(r.shd_avg):.3f}"
            f1 = "" if r.f1_avg is None else f"{float(r.f1_avg):.3f}"
            adjf1 = "" if r.adjacency_f1_avg is None else f"{float(r.adjacency_f1_avg):.3f}"
            ahf1 = "" if r.arrowhead_f1_avg is None else f"{float(r.arrowhead_f1_avg):.3f}"
            rm_disp = "" if r.selected_count is None else f"{int(r.selected_count):d}"
            build_sec = "" if r.build_time_sec is None else f"{float(r.build_time_sec):.3f}"
            solve_sec = "" if r.solve_time_sec is None else f"{float(r.solve_time_sec):.3f}"
            eval_sec = "" if r.eval_time_sec is None else f"{float(r.eval_time_sec):.3f}"
            t_disp = float(r.method_time_sec) if r.method_time_sec is not None else float(r.wall_time_sec)
            shd_bw = _bw_pair(r.shd_best, r.shd_worst)
            f1_bw = _bw_pair(r.f1_best, r.f1_worst)
            adj_bw = _bw_pair(r.adjacency_f1_best, r.adjacency_f1_worst)
            ah_bw = _bw_pair(r.arrowhead_f1_best, r.arrowhead_f1_worst)
            cshd_bw = _bw_pair(r.cpdag_shd_best, r.cpdag_shd_worst)
            cf1_bw = _bw_pair(r.cpdag_f1_best, r.cpdag_f1_worst)
            cadj_bw = _bw_pair(r.cpdag_adjacency_f1_best, r.cpdag_adjacency_f1_worst)
            cah_bw = _bw_pair(r.cpdag_arrowhead_f1_best, r.cpdag_arrowhead_f1_worst)
            print(
                f"{r.rep:4d} {r.encoding:<4} {r.objective:<4} {r.opt_strategy:<12} {r.opt_mode:<4} {r.reification:<8} {r.status:<12} "
                f"{rm_disp:>2} {t_disp:7.3f} {build_sec:>6} {solve_sec:>6} {eval_sec:>6} {int(r.timed_out):>3d} "
                f"{r.n_tests_total:>3d} {r.n_tests_true:>3d} {r.n_tests_accepted:>4d} {r.n_tests_accepted_true:>4d} "
                f"{w_acc:>11} {int(r.weight_accepted_true):>11d} "
                f"{nd:>3} {shd:>6} {f1:>5} {adjf1:>6} {ahf1:>6} "
                f"{shd_bw} {f1_bw} {adj_bw} {ah_bw} {cshd_bw} {cf1_bw} {cadj_bw} {cah_bw}"
            )

    def _print_wc_summary(rows: list[StrategyResult]) -> None:
        if not rows:
            return

        def _iter_baseline_summary_rows() -> list[tuple[str, str, str, str, str, list[BaselineResult]]]:
            if not baseline_results:
                return []
            by_solver: dict[tuple[str, str], list[BaselineResult]] = {}
            for r in baseline_results:
                by_solver.setdefault((r.solver, str(r.opt_mode or "optN")), []).append(r)
            out: list[tuple[str, str, str, str, str, list[BaselineResult]]] = []
            # Match the labels used elsewhere, split by opt_mode.
            def _enc_label(solver: str) -> str:
                return "inc" if solver == "causalaba_increm" else "base"

            def _sort_key(k: tuple[str, str]) -> tuple[str, str]:
                solver, optm = k
                return (_enc_label(solver), optm)

            for (solver, optm) in sorted(by_solver.keys(), key=_sort_key):
                out.append((_enc_label(solver), "-", "baseline", optm, "-", by_solver[(solver, optm)]))
            return out
        header_avg = (
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'time_sec':>10} {'build':>10} {'solve':>10} {'eval':>10} {'timeouts':>9} "
            f"{'n_total':>10} {'n_true':>10} {'n_acc':>10} {'n_accT':>10} "
            f"{'w_acc':>12} {'w_accT':>12} {'n_dags':>10} {'shd':>8} {'F1':>8} {'adjF1':>8} {'ahF1':>8}"
        )
        header_minmax = (
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'time_sec':>14} {'build':>14} {'solve':>14} {'eval':>14} {'timeouts':>9} "
            f"{'n_total':>14} {'n_true':>14} {'n_acc':>14} {'n_accT':>14} "
            f"{'w_acc':>16} {'w_accT':>16} {'n_dags':>14} {'shd':>14} {'F1':>14} {'adjF1':>14} {'ahF1':>14}"
        )
        grouped: dict[tuple[str, str, str, str, str], list[StrategyResult]] = {}
        for r in rows:
            key = (r.encoding, r.objective, r.opt_strategy, r.opt_mode, r.reification)
            grouped.setdefault(key, []).append(r)

        print("\nWC summary (avg)")
        print(f"run: n_nodes={n_nodes}, seed={seed}, reps={reps}, timeout_sec={timeout_sec}, pct_wrong_facts={pct_wrong_facts}")
        print(header_avg)

        # Prepend baseline rows for easy comparison.
        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            t_avg, _, _ = _stats([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            timeouts = sum(1 for i in items if i.timed_out)
            build_avg = _stats([i.build_time_sec for i in items if i.build_time_sec is not None])[0]
            solve_avg = _stats([i.solve_time_sec for i in items if i.solve_time_sec is not None])[0]
            eval_avg = _stats([i.eval_time_sec for i in items if i.eval_time_sec is not None])[0]
            n_total_avg, _, _ = _stats([i.n_tests_total for i in items])
            n_true_avg, _, _ = _stats([i.n_tests_true for i in items])
            n_acc_avg, _, _ = _stats([i.n_tests_accepted for i in items])
            n_accT_avg, _, _ = _stats([i.n_tests_accepted_true for i in items])

            w_acc_vals = [i.accepted_weight for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            w_acc_avg = float("nan")
            w_accT_avg = float("nan")
            if w_acc_vals:
                w_acc_avg, _, _ = _stats(w_acc_vals)
            if w_accT_vals:
                w_accT_avg, _, _ = _stats(w_accT_vals)

            n_dags_avg, _, _ = _stats([float(i.n_dags_compat) for i in items if i.n_dags_compat is not None])
            shd_avg, _, _ = _stats([float(i.shd_avg) for i in items if i.shd_avg is not None])
            f1_avg, _, _ = _stats([float(i.f1_avg) for i in items if i.f1_avg is not None])
            adj_f1_avg, _, _ = _stats([float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None])
            ah_f1_avg, _, _ = _stats([float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None])
            nd_disp = "" if n_dags_avg != n_dags_avg else f"{n_dags_avg:0.1f}"
            shd_disp = "" if shd_avg != shd_avg else f"{shd_avg:0.3f}"
            f1_disp = "" if f1_avg != f1_avg else f"{f1_avg:0.3f}"
            adj_f1_disp = "" if adj_f1_avg != adj_f1_avg else f"{adj_f1_avg:0.3f}"
            ah_f1_disp = "" if ah_f1_avg != ah_f1_avg else f"{ah_f1_avg:0.3f}"

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{t_avg:>10.3f} {build_avg:>10.3f} {solve_avg:>10.3f} {eval_avg:>10.3f} {timeouts:>9} "
                f"{n_total_avg:>10.1f} {n_true_avg:>10.1f} {n_acc_avg:>10.1f} {n_accT_avg:>10.1f} "
                f"{(int(round(w_acc_avg)) if w_acc_vals else ''):>12} {int(round(w_accT_avg)):>12} "
                f"{nd_disp:>10} {shd_disp:>8} {f1_disp:>8} {adj_f1_disp:>8} {ah_f1_disp:>8}"
            )

        for key, items in sorted(grouped.items()):
            enc, obj, opt, optm, reif = key
            t_avg, _, _ = _stats([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            b_avg, _, _ = _stats([i.build_time_sec for i in items if not i.timed_out])
            s_avg, _, _ = _stats([i.solve_time_sec for i in items if not i.timed_out])
            e_avg, _, _ = _stats([i.eval_time_sec for i in items if not i.timed_out])
            timeouts = sum(1 for i in items if i.timed_out)
            n_total_avg, _, _ = _stats([i.n_tests_total for i in items])
            n_true_avg, _, _ = _stats([i.n_tests_true for i in items])
            n_acc_avg, _, _ = _stats([i.n_tests_accepted for i in items])
            n_accT_avg, _, _ = _stats([i.n_tests_accepted_true for i in items])

            w_acc_vals = [i.accepted_weight for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            w_acc_avg = float("nan")
            w_accT_avg = float("nan")
            if w_acc_vals:
                w_acc_avg, _, _ = _stats(w_acc_vals)
            if w_accT_vals:
                w_accT_avg, _, _ = _stats(w_accT_vals)

            n_dags_avg, _, _ = _stats([float(i.n_dags_compat) for i in items if i.n_dags_compat is not None])
            shd_avg, _, _ = _stats([float(i.shd_avg) for i in items if i.shd_avg is not None])
            f1_avg, _, _ = _stats([float(i.f1_avg) for i in items if i.f1_avg is not None])
            adj_f1_avg, _, _ = _stats([float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None])
            ah_f1_avg, _, _ = _stats([float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None])
            nd_disp = "" if n_dags_avg != n_dags_avg else f"{n_dags_avg:0.1f}"
            shd_disp = "" if shd_avg != shd_avg else f"{shd_avg:0.3f}"
            f1_disp = "" if f1_avg != f1_avg else f"{f1_avg:0.3f}"
            adj_f1_disp = "" if adj_f1_avg != adj_f1_avg else f"{adj_f1_avg:0.3f}"
            ah_f1_disp = "" if ah_f1_avg != ah_f1_avg else f"{ah_f1_avg:0.3f}"

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{t_avg:>10.3f} {b_avg:>10.3f} {s_avg:>10.3f} {e_avg:>10.3f} {timeouts:>9} "
                f"{n_total_avg:>10.1f} {n_true_avg:>10.1f} {n_acc_avg:>10.1f} {n_accT_avg:>10.1f} "
                f"{(int(round(w_acc_avg)) if w_acc_vals else ''):>12} {int(round(w_accT_avg)):>12} "
                f"{nd_disp:>10} {shd_disp:>8} {f1_disp:>8} {adj_f1_disp:>8} {ah_f1_disp:>8}"
            )

        print("\nWC summary (std)")
        print(header_avg.replace('time_sec', 'time_sd').replace('build', 'build_sd').replace('solve', 'solve_sd').replace('eval', 'eval_sd'))

        def _disp(x: float, fmt: str) -> str:
            return "" if x != x else format(x, fmt)

        # Baselines
        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            t_sd = _stdev([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            b_sd = _stdev([i.build_time_sec for i in items if i.build_time_sec is not None])
            s_sd = _stdev([i.solve_time_sec for i in items if i.solve_time_sec is not None])
            e_sd = _stdev([i.eval_time_sec for i in items if i.eval_time_sec is not None])
            timeouts = sum(1 for i in items if i.timed_out)

            n_total_sd = _stdev([float(i.n_tests_total) for i in items])
            n_true_sd = _stdev([float(i.n_tests_true) for i in items])
            n_acc_sd = _stdev([float(i.n_tests_accepted) for i in items])
            n_accT_sd = _stdev([float(i.n_tests_accepted_true) for i in items])

            w_acc_vals = [float(i.accepted_weight) for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            w_acc_sd = _stdev(w_acc_vals) if w_acc_vals else float('nan')
            w_accT_sd = _stdev(w_accT_vals) if w_accT_vals else float('nan')

            n_dags_sd = _stdev([float(i.n_dags_compat) for i in items if i.n_dags_compat is not None])
            shd_sd = _stdev([float(i.shd_avg) for i in items if i.shd_avg is not None])
            f1_sd = _stdev([float(i.f1_avg) for i in items if i.f1_avg is not None])
            adj_sd = _stdev([float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None])
            ah_sd = _stdev([float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None])

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{t_sd:>10.3f} {b_sd:>10.3f} {s_sd:>10.3f} {e_sd:>10.3f} {timeouts:>9} "
                f"{n_total_sd:>10.1f} {n_true_sd:>10.1f} {n_acc_sd:>10.1f} {n_accT_sd:>10.1f} "
                f"{_disp(w_acc_sd,'0.1f'):>12} {(_disp(w_accT_sd,'0.1f')):>12} "
                f"{_disp(n_dags_sd,'0.1f'):>10} {(_disp(shd_sd,'0.3f')):>8} {(_disp(f1_sd,'0.3f')):>8} {(_disp(adj_sd,'0.3f')):>8} {(_disp(ah_sd,'0.3f')):>8}"
            )

        # WC groups
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            t_sd = _stdev([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            b_sd = _stdev([i.build_time_sec for i in items if not i.timed_out and i.build_time_sec is not None])
            s_sd = _stdev([i.solve_time_sec for i in items if not i.timed_out and i.solve_time_sec is not None])
            e_sd = _stdev([i.eval_time_sec for i in items if not i.timed_out and i.eval_time_sec is not None])
            timeouts = sum(1 for i in items if i.timed_out)

            n_total_sd = _stdev([float(i.n_tests_total) for i in items])
            n_true_sd = _stdev([float(i.n_tests_true) for i in items])
            n_acc_sd = _stdev([float(i.n_tests_accepted) for i in items])
            n_accT_sd = _stdev([float(i.n_tests_accepted_true) for i in items])

            w_acc_vals = [float(i.accepted_weight) for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            w_acc_sd = _stdev(w_acc_vals) if w_acc_vals else float('nan')
            w_accT_sd = _stdev(w_accT_vals) if w_accT_vals else float('nan')

            n_dags_sd = _stdev([float(i.n_dags_compat) for i in items if i.n_dags_compat is not None])
            shd_sd = _stdev([float(i.shd_avg) for i in items if i.shd_avg is not None])
            f1_sd = _stdev([float(i.f1_avg) for i in items if i.f1_avg is not None])
            adj_sd = _stdev([float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None])
            ah_sd = _stdev([float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None])

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{t_sd:>10.3f} {b_sd:>10.3f} {s_sd:>10.3f} {e_sd:>10.3f} {timeouts:>9} "
                f"{n_total_sd:>10.1f} {n_true_sd:>10.1f} {n_acc_sd:>10.1f} {n_accT_sd:>10.1f} "
                f"{_disp(w_acc_sd,'0.1f'):>12} {(_disp(w_accT_sd,'0.1f')):>12} "
                f"{_disp(n_dags_sd,'0.1f'):>10} {(_disp(shd_sd,'0.3f')):>8} {(_disp(f1_sd,'0.3f')):>8} {(_disp(adj_sd,'0.3f')):>8} {(_disp(ah_sd,'0.3f')):>8}"
            )

        # Facts + CPDAG summary (avg)
        print("\nWC summary (avg) (facts + CPDAG)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'factF1':>10} {'inT':>6} {'inCP':>6} {'nCPD':>8} {'cshd':>10} {'cF1':>10} {'cadjF1':>10} {'cahF1':>10}"
        )

        def _disp(x: float, fmt: str) -> str:
            return "" if x != x else format(x, fmt)

        # Baselines
        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            fact_f1_avg, _, _ = _stats([float(i.accepted_fact_f1) for i in items if i.accepted_fact_f1 is not None])
            inT_avg, _, _ = _stats([float(i.true_dag_in_compat) for i in items if i.true_dag_in_compat is not None])
            inCP_avg, _, _ = _stats([float(i.true_cpdag_in_compat) for i in items if i.true_cpdag_in_compat is not None])
            ncpd_avg, _, _ = _stats([float(i.n_cpdags_compat) for i in items if i.n_cpdags_compat is not None])
            cshd_avg, _, _ = _stats([float(i.cpdag_shd_avg) for i in items if i.cpdag_shd_avg is not None])
            cf1_avg, _, _ = _stats([float(i.cpdag_f1_avg) for i in items if i.cpdag_f1_avg is not None])
            cadj_avg, _, _ = _stats([float(i.cpdag_adjacency_f1_avg) for i in items if i.cpdag_adjacency_f1_avg is not None])
            cah_avg, _, _ = _stats([float(i.cpdag_arrowhead_f1_avg) for i in items if i.cpdag_arrowhead_f1_avg is not None])
            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{_disp(fact_f1_avg,'0.3f'):>10} {(_disp(inT_avg,'0.3f') if inT_avg == inT_avg else ''):>6} {(_disp(inCP_avg,'0.3f') if inCP_avg == inCP_avg else ''):>6} "
                f"{_disp(ncpd_avg,'0.1f'):>8} {(_disp(cshd_avg,'0.3f')):>10} {(_disp(cf1_avg,'0.3f')):>10} {(_disp(cadj_avg,'0.3f')):>10} {(_disp(cah_avg,'0.3f')):>10}"
            )

        # WC groups
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            fact_f1_avg, _, _ = _stats([float(i.accepted_fact_f1) for i in items if i.accepted_fact_f1 is not None])
            inT_avg, _, _ = _stats([float(i.true_dag_in_compat) for i in items if i.true_dag_in_compat is not None])
            inCP_avg, _, _ = _stats([float(i.true_cpdag_in_compat) for i in items if i.true_cpdag_in_compat is not None])
            ncpd_avg, _, _ = _stats([float(i.n_cpdags_compat) for i in items if i.n_cpdags_compat is not None])
            cshd_avg, _, _ = _stats([float(i.cpdag_shd_avg) for i in items if i.cpdag_shd_avg is not None])
            cf1_avg, _, _ = _stats([float(i.cpdag_f1_avg) for i in items if i.cpdag_f1_avg is not None])
            cadj_avg, _, _ = _stats([float(i.cpdag_adjacency_f1_avg) for i in items if i.cpdag_adjacency_f1_avg is not None])
            cah_avg, _, _ = _stats([float(i.cpdag_arrowhead_f1_avg) for i in items if i.cpdag_arrowhead_f1_avg is not None])
            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{_disp(fact_f1_avg,'0.3f'):>10} {(_disp(inT_avg,'0.3f') if inT_avg == inT_avg else ''):>6} {(_disp(inCP_avg,'0.3f') if inCP_avg == inCP_avg else ''):>6} "
                f"{_disp(ncpd_avg,'0.1f'):>8} {(_disp(cshd_avg,'0.3f')):>10} {(_disp(cf1_avg,'0.3f')):>10} {(_disp(cadj_avg,'0.3f')):>10} {(_disp(cah_avg,'0.3f')):>10}"
            )

        print("\nWC summary (std) (facts + CPDAG)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'factF1_sd':>10} {'inT_sd':>8} {'inCP_sd':>8} {'nCPD_sd':>10} {'cshd_sd':>10} {'cF1_sd':>10} {'cadj_sd':>10} {'cah_sd':>10}"
        )

        def _disp2(x: float, fmt: str) -> str:
            return "" if x != x else format(x, fmt)

        # Baselines
        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            fact_f1_sd = _stdev([float(i.accepted_fact_f1) for i in items if i.accepted_fact_f1 is not None])
            inT_sd = _stdev([float(i.true_dag_in_compat) for i in items if i.true_dag_in_compat is not None])
            inCP_sd = _stdev([float(i.true_cpdag_in_compat) for i in items if i.true_cpdag_in_compat is not None])
            ncpd_sd = _stdev([float(i.n_cpdags_compat) for i in items if i.n_cpdags_compat is not None])
            cshd_sd = _stdev([float(i.cpdag_shd_avg) for i in items if i.cpdag_shd_avg is not None])
            cf1_sd = _stdev([float(i.cpdag_f1_avg) for i in items if i.cpdag_f1_avg is not None])
            cadj_sd = _stdev([float(i.cpdag_adjacency_f1_avg) for i in items if i.cpdag_adjacency_f1_avg is not None])
            cah_sd = _stdev([float(i.cpdag_arrowhead_f1_avg) for i in items if i.cpdag_arrowhead_f1_avg is not None])

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{_disp2(fact_f1_sd,'0.3f'):>10} {(_disp2(inT_sd,'0.3f') if inT_sd == inT_sd else ''):>8} {(_disp2(inCP_sd,'0.3f') if inCP_sd == inCP_sd else ''):>8} "
                f"{_disp2(ncpd_sd,'0.1f'):>10} {(_disp2(cshd_sd,'0.3f')):>10} {(_disp2(cf1_sd,'0.3f')):>10} {(_disp2(cadj_sd,'0.3f')):>10} {(_disp2(cah_sd,'0.3f')):>10}"
            )

        # WC groups
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            fact_f1_sd = _stdev([float(i.accepted_fact_f1) for i in items if i.accepted_fact_f1 is not None])
            inT_sd = _stdev([float(i.true_dag_in_compat) for i in items if i.true_dag_in_compat is not None])
            inCP_sd = _stdev([float(i.true_cpdag_in_compat) for i in items if i.true_cpdag_in_compat is not None])
            ncpd_sd = _stdev([float(i.n_cpdags_compat) for i in items if i.n_cpdags_compat is not None])
            cshd_sd = _stdev([float(i.cpdag_shd_avg) for i in items if i.cpdag_shd_avg is not None])
            cf1_sd = _stdev([float(i.cpdag_f1_avg) for i in items if i.cpdag_f1_avg is not None])
            cadj_sd = _stdev([float(i.cpdag_adjacency_f1_avg) for i in items if i.cpdag_adjacency_f1_avg is not None])
            cah_sd = _stdev([float(i.cpdag_arrowhead_f1_avg) for i in items if i.cpdag_arrowhead_f1_avg is not None])

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{_disp2(fact_f1_sd,'0.3f'):>10} {(_disp2(inT_sd,'0.3f') if inT_sd == inT_sd else ''):>8} {(_disp2(inCP_sd,'0.3f') if inCP_sd == inCP_sd else ''):>8} "
                f"{_disp2(ncpd_sd,'0.1f'):>10} {(_disp2(cshd_sd,'0.3f')):>10} {(_disp2(cf1_sd,'0.3f')):>10} {(_disp2(cadj_sd,'0.3f')):>10} {(_disp2(cah_sd,'0.3f')):>10}"
            )

        def _is_finite_number(x: Any) -> bool:
            try:
                import math as _math

                return bool(_math.isfinite(float(x)))
            except Exception:
                return False

        def _rank_map(method_to_value: dict[tuple[str, str, str, str, str], Any], *, higher_is_better: bool) -> dict[tuple[str, str, str, str, str], float]:
            """Return per-method ranks (1..N, lower=better). Missing/non-finite values get worst rank."""
            methods_all = list(method_to_value.keys())
            n_methods = len(methods_all)
            present: list[tuple[tuple[str, str, str, str, str], float]] = []
            for m, v in method_to_value.items():
                if _is_finite_number(v):
                    present.append((m, float(v)))
            present.sort(key=lambda kv: kv[1], reverse=higher_is_better)

            ranks: dict[tuple[str, str, str, str, str], float] = {m: float(n_methods) for m in methods_all}
            i = 0
            while i < len(present):
                j = i + 1
                # Group ties by exact numeric equality; values here are small/integers so this is ok.
                while j < len(present) and present[j][1] == present[i][1]:
                    j += 1
                # Ranks are 1-indexed; ties get average rank.
                rank_lo = i + 1
                rank_hi = j
                rank_avg = (rank_lo + rank_hi) / 2.0
                for k in range(i, j):
                    ranks[present[k][0]] = float(rank_avg)
                i = j
            return ranks

        def _print_metric_ranks() -> None:
            """Ranks methods per-rep on each metric, then averages ranks across reps."""
            # Define all methods to include (baselines + WC combinations)
            methods: list[tuple[str, str, str, str, str]] = []
            rep_set: set[int] = set()
            method_rep: dict[tuple[str, str, str, str, str], dict[int, Any]] = {}

            # Baselines
            for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
                mkey = (enc, obj, opt, optm, reif)
                methods.append(mkey)
                md: dict[int, Any] = {}
                for it in items:
                    md[int(it.rep)] = it
                    rep_set.add(int(it.rep))
                method_rep[mkey] = md

            # WC
            for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
                mkey = (enc, obj, opt, optm, reif)
                methods.append(mkey)
                md = method_rep.setdefault(mkey, {})
                for it in items:
                    md[int(it.rep)] = it
                    rep_set.add(int(it.rep))

            reps_sorted = sorted(rep_set)
            if not methods or not reps_sorted:
                return

            def _time_for_rank(it: Any) -> float | None:
                mt = getattr(it, "method_time_sec", None)
                if mt is not None:
                    return mt
                b = getattr(it, "build_time_sec", None)
                s = getattr(it, "solve_time_sec", None)
                if b is not None and s is not None:
                    return float(b) + float(s)
                # Back-compat for any older in-memory row shapes.
                b2 = getattr(it, "build_sec", None)
                s2 = getattr(it, "solve_sec", None)
                if b2 is not None and s2 is not None:
                    return float(b2) + float(s2)
                return getattr(it, "wall_time_sec", None)

            def _solve_for_rank(it: Any) -> float | None:
                s = getattr(it, "solve_time_sec", None)
                if s is not None:
                    return float(s)
                s2 = getattr(it, "solve_sec", None)
                return float(s2) if s2 is not None else None

            metric_defs: list[tuple[str, bool, Any]] = [
                ("timeR", False, _time_for_rank),
                ("solveR", False, _solve_for_rank),
                ("toR", False, lambda it: int(bool(getattr(it, "timed_out", False))) if hasattr(it, "timed_out") else None),
                ("nATR", True, lambda it: getattr(it, "n_tests_accepted_true", None)),
                ("wATR", True, lambda it: getattr(it, "weight_accepted_true", None)),
                ("nDR", False, lambda it: getattr(it, "n_dags_compat", None)),
                ("shdR", False, lambda it: getattr(it, "shd_avg", None)),
                ("F1R", True, lambda it: getattr(it, "f1_avg", None)),
                ("adjR", True, lambda it: getattr(it, "adjacency_f1_avg", None)),
                ("ahR", True, lambda it: getattr(it, "arrowhead_f1_avg", None)),
            ]

            # ranks_acc[method][metric] -> list of ranks over reps
            ranks_acc: dict[tuple[str, str, str, str, str], dict[str, list[float]]] = {
                m: {name: [] for (name, _, _) in metric_defs} for m in methods
            }

            for rep in reps_sorted:
                for name, higher_is_better, getter in metric_defs:
                    vals: dict[tuple[str, str, str, str, str], Any] = {}
                    for m in methods:
                        it = method_rep.get(m, {}).get(rep, None)
                        v = None
                        if it is not None:
                            try:
                                v = getter(it)
                            except Exception:
                                v = None
                        vals[m] = v
                    rm = _rank_map(vals, higher_is_better=higher_is_better)
                    for m, rnk in rm.items():
                        ranks_acc[m][name].append(float(rnk))

            print("\nWC metric ranks (avg)  (lower is better)")
            print(
                f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
                + " ".join(f"{name:>6}" for (name, _, _) in metric_defs)
            )
            for m in methods:
                enc, obj, opt, optm, reif = m
                out_fields: list[str] = []
                for name, _, _ in metric_defs:
                    xs = ranks_acc[m][name]
                    out_fields.append(f"{(sum(xs)/len(xs)) if xs else float('nan'):6.2f}")
                print(f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(reps_sorted):>6} " + " ".join(out_fields))

        _print_metric_ranks()

        print("\nWC summary (min/max)")
        # Split min/max into two compact blocks to avoid terminal wrapping.
        print("(times + counts)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'time':>14} {'build':>14} {'solve':>14} {'eval':>14} {'to':>4} {'nT':>12} {'nTr':>12} {'nA':>12} {'nAT':>12}"
        )

        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            _, t_min, t_max = _stats([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            timeouts = sum(1 for i in items if i.timed_out)
            _, b_min, b_max = _stats([i.build_time_sec for i in items if i.build_time_sec is not None])
            _, s_min, s_max = _stats([i.solve_time_sec for i in items if i.solve_time_sec is not None])
            _, e_min, e_max = _stats([i.eval_time_sec for i in items if i.eval_time_sec is not None])
            _, n_total_min, n_total_max = _stats([i.n_tests_total for i in items])
            _, n_true_min, n_true_max = _stats([i.n_tests_true for i in items])
            _, n_acc_min, n_acc_max = _stats([i.n_tests_accepted for i in items])
            _, n_accT_min, n_accT_max = _stats([i.n_tests_accepted_true for i in items])
            time_range = f"[{t_min:.3f},{t_max:.3f}]".rjust(14)
            build_range = (f"[{b_min:.3f},{b_max:.3f}]".rjust(14) if b_min == b_min else "".rjust(14))
            solve_range = (f"[{s_min:.3f},{s_max:.3f}]".rjust(14) if s_min == s_min else "".rjust(14))
            eval_range = (f"[{e_min:.3f},{e_max:.3f}]".rjust(14) if e_min == e_min else "".rjust(14))
            n_total_range = f"[{n_total_min:.1f},{n_total_max:.1f}]".rjust(12)
            n_true_range = f"[{n_true_min:.1f},{n_true_max:.1f}]".rjust(12)
            n_acc_range = f"[{n_acc_min:.1f},{n_acc_max:.1f}]".rjust(12)
            n_accT_range = f"[{n_accT_min:.1f},{n_accT_max:.1f}]".rjust(12)

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{time_range} {build_range} {solve_range} {eval_range} {timeouts:>4d} "
                f"{n_total_range} {n_true_range} {n_acc_range} {n_accT_range}"
            )
        for key, items in sorted(grouped.items()):
            enc, obj, opt, optm, reif = key
            _, t_min, t_max = _stats([(i.method_time_sec if i.method_time_sec is not None else i.wall_time_sec) for i in items])
            _, b_min, b_max = _stats([i.build_time_sec for i in items if not i.timed_out])
            _, s_min, s_max = _stats([i.solve_time_sec for i in items if not i.timed_out])
            _, e_min, e_max = _stats([i.eval_time_sec for i in items if not i.timed_out])
            timeouts = sum(1 for i in items if i.timed_out)
            _, n_total_min, n_total_max = _stats([i.n_tests_total for i in items])
            _, n_true_min, n_true_max = _stats([i.n_tests_true for i in items])
            _, n_acc_min, n_acc_max = _stats([i.n_tests_accepted for i in items])
            _, n_accT_min, n_accT_max = _stats([i.n_tests_accepted_true for i in items])
            w_acc_vals = [i.accepted_weight for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            _, w_acc_min, w_acc_max = _stats(w_acc_vals) if w_acc_vals else (float("nan"), float("nan"), float("nan"))
            _, w_accT_min, w_accT_max = _stats(w_accT_vals) if w_accT_vals else (float("nan"), float("nan"), float("nan"))
            time_range = f"[{t_min:.3f},{t_max:.3f}]".rjust(14)
            build_range = (f"[{b_min:.3f},{b_max:.3f}]".rjust(14) if b_min == b_min else "".rjust(14))
            solve_range = (f"[{s_min:.3f},{s_max:.3f}]".rjust(14) if s_min == s_min else "".rjust(14))
            eval_range = (f"[{e_min:.3f},{e_max:.3f}]".rjust(14) if e_min == e_min else "".rjust(14))
            n_total_range = f"[{n_total_min:.1f},{n_total_max:.1f}]".rjust(12)
            n_true_range = f"[{n_true_min:.1f},{n_true_max:.1f}]".rjust(12)
            n_acc_range = f"[{n_acc_min:.1f},{n_acc_max:.1f}]".rjust(12)
            n_accT_range = f"[{n_accT_min:.1f},{n_accT_max:.1f}]".rjust(12)

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{time_range} {build_range} {solve_range} {eval_range} {timeouts:>4d} "
                f"{n_total_range} {n_true_range} {n_acc_range} {n_accT_range}"
            )

        print("\n(weights + graph)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'wA':>22} {'wAT':>22} {'nD':>12} {'shd':>14} {'F1':>12} {'adjF1':>12} {'ahF1':>12}"
        )

        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            w_acc_vals = [i.accepted_weight for i in items if i.accepted_weight is not None]
            w_accT_vals = [float(i.weight_accepted_true) for i in items]
            _, w_acc_min, w_acc_max = _stats(w_acc_vals) if w_acc_vals else (float("nan"), float("nan"), float("nan"))
            _, w_accT_min, w_accT_max = _stats(w_accT_vals) if w_accT_vals else (float("nan"), float("nan"), float("nan"))
            w_acc_range = (f"[{w_acc_min:.0f},{w_acc_max:.0f}]".rjust(22) if w_acc_vals else "".rjust(22))
            w_accT_range = (f"[{w_accT_min:.0f},{w_accT_max:.0f}]".rjust(22) if w_accT_vals else "".rjust(22))

            nd_vals = [float(i.n_dags_compat) for i in items if i.n_dags_compat is not None]
            shd_vals = [float(i.shd_avg) for i in items if i.shd_avg is not None]
            f1_vals = [float(i.f1_avg) for i in items if i.f1_avg is not None]
            adj_f1_vals = [float(i.adjacency_f1_avg) for i in items if i.adjacency_f1_avg is not None]
            ah_f1_vals = [float(i.arrowhead_f1_avg) for i in items if i.arrowhead_f1_avg is not None]
            _, nd_min, nd_max = _stats(nd_vals) if nd_vals else (float("nan"), float("nan"), float("nan"))
            _, shd_min, shd_max = _stats(shd_vals) if shd_vals else (float("nan"), float("nan"), float("nan"))
            _, f1_min, f1_max = _stats(f1_vals) if f1_vals else (float("nan"), float("nan"), float("nan"))
            _, adj_f1_min, adj_f1_max = _stats(adj_f1_vals) if adj_f1_vals else (float("nan"), float("nan"), float("nan"))
            _, ah_f1_min, ah_f1_max = _stats(ah_f1_vals) if ah_f1_vals else (float("nan"), float("nan"), float("nan"))
            nd_range = (f"[{nd_min:.1f},{nd_max:.1f}]".rjust(12) if nd_vals else "".rjust(12))
            shd_range = (f"[{shd_min:.3f},{shd_max:.3f}]".rjust(14) if shd_vals else "".rjust(14))
            f1_range = (f"[{f1_min:.3f},{f1_max:.3f}]".rjust(12) if f1_vals else "".rjust(12))
            adj_f1_range = (f"[{adj_f1_min:.3f},{adj_f1_max:.3f}]".rjust(12) if adj_f1_vals else "".rjust(12))
            ah_f1_range = (f"[{ah_f1_min:.3f},{ah_f1_max:.3f}]".rjust(12) if ah_f1_vals else "".rjust(12))

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{w_acc_range} {w_accT_range} {nd_range} {shd_range} {f1_range} {adj_f1_range} {ah_f1_range}"
            )

        print("\n(facts + CPDAG)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'factF1':>14} {'inT':>10} {'nCPD':>12} {'cshd':>14} {'cF1':>14} {'cadjF1':>14} {'cahF1':>14}"
        )

        def _print_facts_cpdag_minmax(enc: str, obj: str, opt: str, optm: str, reif: str, items: list[Any]) -> None:
            f1_vals = [float(i.accepted_fact_f1) for i in items if getattr(i, 'accepted_fact_f1', None) is not None]
            in_vals = [float(i.true_dag_in_compat) for i in items if getattr(i, 'true_dag_in_compat', None) is not None]
            ncpd_vals = [float(i.n_cpdags_compat) for i in items if getattr(i, 'n_cpdags_compat', None) is not None]
            cshd_vals = [float(i.cpdag_shd_avg) for i in items if getattr(i, 'cpdag_shd_avg', None) is not None]
            cf1_vals = [float(i.cpdag_f1_avg) for i in items if getattr(i, 'cpdag_f1_avg', None) is not None]
            cadj_vals = [float(i.cpdag_adjacency_f1_avg) for i in items if getattr(i, 'cpdag_adjacency_f1_avg', None) is not None]
            cah_vals = [float(i.cpdag_arrowhead_f1_avg) for i in items if getattr(i, 'cpdag_arrowhead_f1_avg', None) is not None]

            _, f1_min, f1_max = _stats(f1_vals) if f1_vals else (float('nan'), float('nan'), float('nan'))
            _, in_min, in_max = _stats(in_vals) if in_vals else (float('nan'), float('nan'), float('nan'))
            _, ncpd_min, ncpd_max = _stats(ncpd_vals) if ncpd_vals else (float('nan'), float('nan'), float('nan'))
            _, cshd_min, cshd_max = _stats(cshd_vals) if cshd_vals else (float('nan'), float('nan'), float('nan'))
            _, cf1_min, cf1_max = _stats(cf1_vals) if cf1_vals else (float('nan'), float('nan'), float('nan'))
            _, cadj_min, cadj_max = _stats(cadj_vals) if cadj_vals else (float('nan'), float('nan'), float('nan'))
            _, cah_min, cah_max = _stats(cah_vals) if cah_vals else (float('nan'), float('nan'), float('nan'))

            factF1_range = (f"[{f1_min:.3f},{f1_max:.3f}]".rjust(14) if f1_vals else "".rjust(14))
            inT_range = (f"[{int(in_min):d},{int(in_max):d}]".rjust(10) if in_vals else "".rjust(10))
            ncpd_range = (f"[{int(ncpd_min):d},{int(ncpd_max):d}]".rjust(12) if ncpd_vals else "".rjust(12))
            cshd_range = (f"[{cshd_min:.3f},{cshd_max:.3f}]".rjust(14) if cshd_vals else "".rjust(14))
            cF1_range = (f"[{cf1_min:.3f},{cf1_max:.3f}]".rjust(14) if cf1_vals else "".rjust(14))
            cadj_range = (f"[{cadj_min:.3f},{cadj_max:.3f}]".rjust(14) if cadj_vals else "".rjust(14))
            cah_range = (f"[{cah_min:.3f},{cah_max:.3f}]".rjust(14) if cah_vals else "".rjust(14))

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{factF1_range} {inT_range} {ncpd_range} {cshd_range} {cF1_range} {cadj_range} {cah_range}"
            )

        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            _print_facts_cpdag_minmax(enc, obj, opt, optm, reif, items)
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            _print_facts_cpdag_minmax(enc, obj, opt, optm, reif, items)

        print("\n(graph best/worst over compatible DAGs) (min/max over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
            f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
        )

        def _print_dag_best_worst_row(enc: str, obj: str, opt: str, optm: str, reif: str, items: list[Any]) -> None:
            shd_best_vals = [float(i.shd_best) for i in items if getattr(i, "shd_best", None) is not None]
            shd_worst_vals = [float(i.shd_worst) for i in items if getattr(i, "shd_worst", None) is not None]
            f1_best_vals = [float(i.f1_best) for i in items if getattr(i, "f1_best", None) is not None]
            f1_worst_vals = [float(i.f1_worst) for i in items if getattr(i, "f1_worst", None) is not None]
            adj_best_vals = [float(i.adjacency_f1_best) for i in items if getattr(i, "adjacency_f1_best", None) is not None]
            adj_worst_vals = [float(i.adjacency_f1_worst) for i in items if getattr(i, "adjacency_f1_worst", None) is not None]
            ah_best_vals = [float(i.arrowhead_f1_best) for i in items if getattr(i, "arrowhead_f1_best", None) is not None]
            ah_worst_vals = [float(i.arrowhead_f1_worst) for i in items if getattr(i, "arrowhead_f1_worst", None) is not None]

            _, shd_best_min, shd_best_max = _stats(shd_best_vals) if shd_best_vals else (float("nan"), float("nan"), float("nan"))
            _, shd_worst_min, shd_worst_max = _stats(shd_worst_vals) if shd_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, f1_best_min, f1_best_max = _stats(f1_best_vals) if f1_best_vals else (float("nan"), float("nan"), float("nan"))
            _, f1_worst_min, f1_worst_max = _stats(f1_worst_vals) if f1_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, adj_best_min, adj_best_max = _stats(adj_best_vals) if adj_best_vals else (float("nan"), float("nan"), float("nan"))
            _, adj_worst_min, adj_worst_max = _stats(adj_worst_vals) if adj_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, ah_best_min, ah_best_max = _stats(ah_best_vals) if ah_best_vals else (float("nan"), float("nan"), float("nan"))
            _, ah_worst_min, ah_worst_max = _stats(ah_worst_vals) if ah_worst_vals else (float("nan"), float("nan"), float("nan"))

            shd_best_range = (f"[{shd_best_min:.3f},{shd_best_max:.3f}]".rjust(14) if shd_best_vals else "".rjust(14))
            shd_worst_range = (f"[{shd_worst_min:.3f},{shd_worst_max:.3f}]".rjust(14) if shd_worst_vals else "".rjust(14))
            f1_best_range = (f"[{f1_best_min:.3f},{f1_best_max:.3f}]".rjust(14) if f1_best_vals else "".rjust(14))
            f1_worst_range = (f"[{f1_worst_min:.3f},{f1_worst_max:.3f}]".rjust(14) if f1_worst_vals else "".rjust(14))
            adj_best_range = (f"[{adj_best_min:.3f},{adj_best_max:.3f}]".rjust(14) if adj_best_vals else "".rjust(14))
            adj_worst_range = (f"[{adj_worst_min:.3f},{adj_worst_max:.3f}]".rjust(14) if adj_worst_vals else "".rjust(14))
            ah_best_range = (f"[{ah_best_min:.3f},{ah_best_max:.3f}]".rjust(14) if ah_best_vals else "".rjust(14))
            ah_worst_range = (f"[{ah_worst_min:.3f},{ah_worst_max:.3f}]".rjust(14) if ah_worst_vals else "".rjust(14))

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{shd_best_range} {shd_worst_range} {f1_best_range} {f1_worst_range} "
                f"{adj_best_range} {adj_worst_range} {ah_best_range} {ah_worst_range}"
            )

        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            _print_dag_best_worst_row(enc, obj, opt, optm, reif, items)
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            _print_dag_best_worst_row(enc, obj, opt, optm, reif, items)

        print("\n(graph best/worst over compatible DAGs) (avg over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'shd_best':>14} {'shd_worst':>14} {'F1_best':>14} {'F1_worst':>14} "
            f"{'adj_best':>14} {'adj_worst':>14} {'ah_best':>14} {'ah_worst':>14}"
        )

        def _print_dag_best_worst_avg_row(enc: str, obj: str, opt: str, optm: str, reif: str, items: list[Any]) -> None:
            shd_best_vals = [float(i.shd_best) for i in items if getattr(i, "shd_best", None) is not None]
            shd_worst_vals = [float(i.shd_worst) for i in items if getattr(i, "shd_worst", None) is not None]
            f1_best_vals = [float(i.f1_best) for i in items if getattr(i, "f1_best", None) is not None]
            f1_worst_vals = [float(i.f1_worst) for i in items if getattr(i, "f1_worst", None) is not None]
            adj_best_vals = [float(i.adjacency_f1_best) for i in items if getattr(i, "adjacency_f1_best", None) is not None]
            adj_worst_vals = [float(i.adjacency_f1_worst) for i in items if getattr(i, "adjacency_f1_worst", None) is not None]
            ah_best_vals = [float(i.arrowhead_f1_best) for i in items if getattr(i, "arrowhead_f1_best", None) is not None]
            ah_worst_vals = [float(i.arrowhead_f1_worst) for i in items if getattr(i, "arrowhead_f1_worst", None) is not None]

            shd_best_avg, _, _ = _stats(shd_best_vals) if shd_best_vals else (float("nan"), float("nan"), float("nan"))
            shd_worst_avg, _, _ = _stats(shd_worst_vals) if shd_worst_vals else (float("nan"), float("nan"), float("nan"))
            f1_best_avg, _, _ = _stats(f1_best_vals) if f1_best_vals else (float("nan"), float("nan"), float("nan"))
            f1_worst_avg, _, _ = _stats(f1_worst_vals) if f1_worst_vals else (float("nan"), float("nan"), float("nan"))
            adj_best_avg, _, _ = _stats(adj_best_vals) if adj_best_vals else (float("nan"), float("nan"), float("nan"))
            adj_worst_avg, _, _ = _stats(adj_worst_vals) if adj_worst_vals else (float("nan"), float("nan"), float("nan"))
            ah_best_avg, _, _ = _stats(ah_best_vals) if ah_best_vals else (float("nan"), float("nan"), float("nan"))
            ah_worst_avg, _, _ = _stats(ah_worst_vals) if ah_worst_vals else (float("nan"), float("nan"), float("nan"))

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{('' if shd_best_avg != shd_best_avg else f'{shd_best_avg:.3f}'):>14} "
                f"{('' if shd_worst_avg != shd_worst_avg else f'{shd_worst_avg:.3f}'):>14} "
                f"{('' if f1_best_avg != f1_best_avg else f'{f1_best_avg:.3f}'):>14} "
                f"{('' if f1_worst_avg != f1_worst_avg else f'{f1_worst_avg:.3f}'):>14} "
                f"{('' if adj_best_avg != adj_best_avg else f'{adj_best_avg:.3f}'):>14} "
                f"{('' if adj_worst_avg != adj_worst_avg else f'{adj_worst_avg:.3f}'):>14} "
                f"{('' if ah_best_avg != ah_best_avg else f'{ah_best_avg:.3f}'):>14} "
                f"{('' if ah_worst_avg != ah_worst_avg else f'{ah_worst_avg:.3f}'):>14}"
            )

        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            _print_dag_best_worst_avg_row(enc, obj, opt, optm, reif, items)
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            _print_dag_best_worst_avg_row(enc, obj, opt, optm, reif, items)

        print("\n(CPDAG best/worst over compatible CPDAGs) (min/max over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
            f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
        )

        def _print_cpdag_best_worst_row(enc: str, obj: str, opt: str, optm: str, reif: str, items: list[Any]) -> None:
            cshd_best_vals = [float(i.cpdag_shd_best) for i in items if getattr(i, "cpdag_shd_best", None) is not None]
            cshd_worst_vals = [float(i.cpdag_shd_worst) for i in items if getattr(i, "cpdag_shd_worst", None) is not None]
            cf1_best_vals = [float(i.cpdag_f1_best) for i in items if getattr(i, "cpdag_f1_best", None) is not None]
            cf1_worst_vals = [float(i.cpdag_f1_worst) for i in items if getattr(i, "cpdag_f1_worst", None) is not None]
            cadj_best_vals = [float(i.cpdag_adjacency_f1_best) for i in items if getattr(i, "cpdag_adjacency_f1_best", None) is not None]
            cadj_worst_vals = [float(i.cpdag_adjacency_f1_worst) for i in items if getattr(i, "cpdag_adjacency_f1_worst", None) is not None]
            cah_best_vals = [float(i.cpdag_arrowhead_f1_best) for i in items if getattr(i, "cpdag_arrowhead_f1_best", None) is not None]
            cah_worst_vals = [float(i.cpdag_arrowhead_f1_worst) for i in items if getattr(i, "cpdag_arrowhead_f1_worst", None) is not None]

            _, cshd_best_min, cshd_best_max = _stats(cshd_best_vals) if cshd_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cshd_worst_min, cshd_worst_max = _stats(cshd_worst_vals) if cshd_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, cf1_best_min, cf1_best_max = _stats(cf1_best_vals) if cf1_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cf1_worst_min, cf1_worst_max = _stats(cf1_worst_vals) if cf1_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, cadj_best_min, cadj_best_max = _stats(cadj_best_vals) if cadj_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cadj_worst_min, cadj_worst_max = _stats(cadj_worst_vals) if cadj_worst_vals else (float("nan"), float("nan"), float("nan"))
            _, cah_best_min, cah_best_max = _stats(cah_best_vals) if cah_best_vals else (float("nan"), float("nan"), float("nan"))
            _, cah_worst_min, cah_worst_max = _stats(cah_worst_vals) if cah_worst_vals else (float("nan"), float("nan"), float("nan"))

            cshd_best_range = (f"[{cshd_best_min:.3f},{cshd_best_max:.3f}]".rjust(14) if cshd_best_vals else "".rjust(14))
            cshd_worst_range = (f"[{cshd_worst_min:.3f},{cshd_worst_max:.3f}]".rjust(14) if cshd_worst_vals else "".rjust(14))
            cf1_best_range = (f"[{cf1_best_min:.3f},{cf1_best_max:.3f}]".rjust(14) if cf1_best_vals else "".rjust(14))
            cf1_worst_range = (f"[{cf1_worst_min:.3f},{cf1_worst_max:.3f}]".rjust(14) if cf1_worst_vals else "".rjust(14))
            cadj_best_range = (f"[{cadj_best_min:.3f},{cadj_best_max:.3f}]".rjust(14) if cadj_best_vals else "".rjust(14))
            cadj_worst_range = (f"[{cadj_worst_min:.3f},{cadj_worst_max:.3f}]".rjust(14) if cadj_worst_vals else "".rjust(14))
            cah_best_range = (f"[{cah_best_min:.3f},{cah_best_max:.3f}]".rjust(14) if cah_best_vals else "".rjust(14))
            cah_worst_range = (f"[{cah_worst_min:.3f},{cah_worst_max:.3f}]".rjust(14) if cah_worst_vals else "".rjust(14))

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{cshd_best_range} {cshd_worst_range} {cf1_best_range} {cf1_worst_range} "
                f"{cadj_best_range} {cadj_worst_range} {cah_best_range} {cah_worst_range}"
            )

        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            _print_cpdag_best_worst_row(enc, obj, opt, optm, reif, items)
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            _print_cpdag_best_worst_row(enc, obj, opt, optm, reif, items)

        print("\n(CPDAG best/worst over compatible CPDAGs) (avg over runs)")
        print(
            f"{'enc':<4} {'obj':<4} {'strategy':<12} {'optm':<4} {'reif':<8} {'runs':>6} "
            f"{'cshd_best':>14} {'cshd_worst':>14} {'cF1_best':>14} {'cF1_worst':>14} "
            f"{'cadj_best':>14} {'cadj_worst':>14} {'cah_best':>14} {'cah_worst':>14}"
        )

        def _print_cpdag_best_worst_avg_row(enc: str, obj: str, opt: str, optm: str, reif: str, items: list[Any]) -> None:
            cshd_best_vals = [float(i.cpdag_shd_best) for i in items if getattr(i, "cpdag_shd_best", None) is not None]
            cshd_worst_vals = [float(i.cpdag_shd_worst) for i in items if getattr(i, "cpdag_shd_worst", None) is not None]
            cf1_best_vals = [float(i.cpdag_f1_best) for i in items if getattr(i, "cpdag_f1_best", None) is not None]
            cf1_worst_vals = [float(i.cpdag_f1_worst) for i in items if getattr(i, "cpdag_f1_worst", None) is not None]
            cadj_best_vals = [float(i.cpdag_adjacency_f1_best) for i in items if getattr(i, "cpdag_adjacency_f1_best", None) is not None]
            cadj_worst_vals = [float(i.cpdag_adjacency_f1_worst) for i in items if getattr(i, "cpdag_adjacency_f1_worst", None) is not None]
            cah_best_vals = [float(i.cpdag_arrowhead_f1_best) for i in items if getattr(i, "cpdag_arrowhead_f1_best", None) is not None]
            cah_worst_vals = [float(i.cpdag_arrowhead_f1_worst) for i in items if getattr(i, "cpdag_arrowhead_f1_worst", None) is not None]

            cshd_best_avg, _, _ = _stats(cshd_best_vals) if cshd_best_vals else (float("nan"), float("nan"), float("nan"))
            cshd_worst_avg, _, _ = _stats(cshd_worst_vals) if cshd_worst_vals else (float("nan"), float("nan"), float("nan"))
            cf1_best_avg, _, _ = _stats(cf1_best_vals) if cf1_best_vals else (float("nan"), float("nan"), float("nan"))
            cf1_worst_avg, _, _ = _stats(cf1_worst_vals) if cf1_worst_vals else (float("nan"), float("nan"), float("nan"))
            cadj_best_avg, _, _ = _stats(cadj_best_vals) if cadj_best_vals else (float("nan"), float("nan"), float("nan"))
            cadj_worst_avg, _, _ = _stats(cadj_worst_vals) if cadj_worst_vals else (float("nan"), float("nan"), float("nan"))
            cah_best_avg, _, _ = _stats(cah_best_vals) if cah_best_vals else (float("nan"), float("nan"), float("nan"))
            cah_worst_avg, _, _ = _stats(cah_worst_vals) if cah_worst_vals else (float("nan"), float("nan"), float("nan"))

            print(
                f"{enc:<4} {obj:<4} {opt:<12} {optm:<4} {reif:<8} {len(items):>6} "
                f"{('' if cshd_best_avg != cshd_best_avg else f'{cshd_best_avg:.3f}'):>14} "
                f"{('' if cshd_worst_avg != cshd_worst_avg else f'{cshd_worst_avg:.3f}'):>14} "
                f"{('' if cf1_best_avg != cf1_best_avg else f'{cf1_best_avg:.3f}'):>14} "
                f"{('' if cf1_worst_avg != cf1_worst_avg else f'{cf1_worst_avg:.3f}'):>14} "
                f"{('' if cadj_best_avg != cadj_best_avg else f'{cadj_best_avg:.3f}'):>14} "
                f"{('' if cadj_worst_avg != cadj_worst_avg else f'{cadj_worst_avg:.3f}'):>14} "
                f"{('' if cah_best_avg != cah_best_avg else f'{cah_best_avg:.3f}'):>14} "
                f"{('' if cah_worst_avg != cah_worst_avg else f'{cah_worst_avg:.3f}'):>14}"
            )

        for enc, obj, opt, optm, reif, items in _iter_baseline_summary_rows():
            _print_cpdag_best_worst_avg_row(enc, obj, opt, optm, reif, items)
        for (enc, obj, opt, optm, reif), items in sorted(grouped.items()):
            _print_cpdag_best_worst_avg_row(enc, obj, opt, optm, reif, items)

    # Print summary tables to console
    print("\n" + "="*80)
    print("=== Baseline removal runs ===")
    print("="*80)
    print(f"out_dir={out_dir}")
    print(f"timeout_sec={timeout_sec}")
    print(f"reps={reps}")
    print(f"n_nodes={n_nodes}, seed={seed}")
    print(f"pct_wrong_facts={pct_wrong_facts}")
    print(f"encodings={encodings}, objectives={objectives}, reifications={reifications}, strategies={strategies}")
    _print_baseline_table(baseline_results)
    _print_baseline_summary(baseline_results)

    print("\n" + "="*80)
    print("=== WC opt-strategy sweep ===")
    print("="*80)
    
    # Count problematic runs for summary
    unsat_count = sum(1 for r in wc_results if r.status == "UNSATISFIABLE")
    timeout_count = sum(1 for r in wc_results if r.timed_out)
    if unsat_count > 0 or timeout_count > 0:
        print(f"NOTE: {unsat_count} UNSATISFIABLE runs (encoding failed to find solution), "
              f"{timeout_count} timeouts")
    
    _print_wc_table(wc_results)
    _print_wc_summary(wc_results)
    
    print(f"\nSummary written to {out_json}")
    print("="*80)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
