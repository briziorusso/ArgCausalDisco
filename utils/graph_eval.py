"""Graph evaluation utilities shared with the WC sweep.

This module intentionally mirrors the graph-evaluation logic used in
`scripts/wc_opt_strategy_sweep.py` so worked examples (e.g. notebooks/ex4.py)
report identical metrics.

The main entry point is `graph_eval_from_accepted`, which enumerates compatible
DAGs implied by a set of accepted external CI tests and summarizes accuracy
metrics vs a known ground-truth DAG.
"""

from __future__ import annotations

import math
import multiprocessing
import os
import tempfile
import time
import traceback
from typing import Any, Callable


_GE_BASE_LEN = 13
_GE_EXT_LEN = 29


def _empty_graph_eval(*, timed_out: bool, include_extrema: bool) -> tuple[Any, ...]:
    base: tuple[Any, ...] = (
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
    )
    if not include_extrema:
        return (*base, bool(timed_out))
    # Extended payload appends per-run best/worst for DAG and CPDAG metrics:
    # shd, f1, adjacency_f1, arrowhead_f1, cpdag_shd, cpdag_f1, cpdag_adjacency_f1, cpdag_arrowhead_f1.
    extrema = (None,) * 16
    return (*base, *extrema, bool(timed_out))


def _best_worst(vals: list[float], *, higher_is_better: bool) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    if higher_is_better:
        return float(max(vals)), float(min(vals))
    return float(min(vals)), float(max(vals))


def graph_eval_from_accepted(
    *,
    n_nodes: int,
    G_true1: Any,
    keys_in_file_order: list[str],
    accepted_keys: set[str],
    timeout_sec: float,
    threads: int,
    cache: dict[frozenset[str], tuple[Any, ...]],
    include_extrema: bool = False,
    progress_cb: Callable[[str, dict[str, Any]], None] | None = None,
    progress_snapshot_every: int = 25,
) -> tuple[Any, ...]:
    """Return sweep-compatible graph-eval summary for accepted constraints.

    Output tuple layout (same as sweep `_graph_eval_from_accepted`):

    DAG-level:
      0. n_dags_compat
      1. dag_shd_avg
      2. dag_f1_avg
      3. dag_adjacency_f1_avg
      4. dag_arrowhead_f1_avg
      5. true_dag_in_compat (1/0)

    CPDAG-level:
      6. n_cpdags_compat
      7. true_cpdag_in_compat (1/0/None)
      8. cpdag_shd_avg
      9. cpdag_f1_avg
      10. cpdag_adjacency_f1_avg
      11. cpdag_arrowhead_f1_avg

    12. timed_out

    Optional extended layout (`include_extrema=True`):
      13. dag_shd_best (min over compatible DAGs)
      14. dag_shd_worst (max over compatible DAGs)
      15. dag_f1_best (max)
      16. dag_f1_worst (min)
      17. dag_adjacency_f1_best (max)
      18. dag_adjacency_f1_worst (min)
      19. dag_arrowhead_f1_best (max)
      20. dag_arrowhead_f1_worst (min)
      21. cpdag_shd_best (min over unique compatible CPDAGs)
      22. cpdag_shd_worst (max)
      23. cpdag_f1_best (max)
      24. cpdag_f1_worst (min)
      25. cpdag_adjacency_f1_best (max)
      26. cpdag_adjacency_f1_worst (min)
      27. cpdag_arrowhead_f1_best (max)
      28. cpdag_arrowhead_f1_worst (min)
      29. timed_out

    If enumeration times out or fails, returns (None, ..., timed_out=True/False).

    Optional incremental progress:
      - `progress_cb("graph_eval_dag_done", {"n_dags": ..., "n_cpdags": ...})`
        is emitted for every compatible DAG processed.
      - `progress_cb("graph_eval_partial", {"out": ...})` is emitted periodically
        every `progress_snapshot_every` DAGs (and once at the end), so callers
        can keep partial aggregates when hard wall-time termination is used.
    """

    key = frozenset(accepted_keys or set())
    if key in cache:
        out_cached = cache[key]
        if (not include_extrema and len(out_cached) == _GE_BASE_LEN) or (
            include_extrema and len(out_cached) == _GE_EXT_LEN
        ):
            return out_cached
    if not accepted_keys:
        out = _empty_graph_eval(timed_out=False, include_extrema=include_extrema)
        cache[key] = out
        return out

    # No budget: do not cache.
    try:
        if float(timeout_sec) <= 0.0:
            return _empty_graph_eval(timed_out=True, include_extrema=include_extrema)
    except Exception:
        return _empty_graph_eval(timed_out=True, include_extrema=include_extrema)

    import numpy as np
    import networkx as nx

    from causalaba import compile_and_ground
    from utils.graph_utils import DAGMetrics, dag2cpdag, extract_test_elements_from_symbol, model_to_set_of_arrows

    try:
        B_true = nx.to_numpy_array(G_true1, nodelist=list(range(n_nodes)), dtype=int)
    except Exception:
        out = _empty_graph_eval(timed_out=False, include_extrema=include_extrema)
        cache[key] = out
        return out

    # True CPDAG for CPDAG-level evaluation.
    try:
        B_true_cpdag = dag2cpdag(B_true.copy())
    except Exception:
        B_true_cpdag = None

    # Precompute true directed edges + skeleton.
    true_arrows: set[tuple[int, int]] = set()
    true_skel: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            try:
                if int(B_true[i, j]) == 1:
                    true_arrows.add((i, j))
                    a, b = (i, j) if i < j else (j, i)
                    true_skel.add((a, b))
            except Exception:
                continue

    def _f1_from_pr(p: float, r: float) -> float:
        return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    def _f1_from_sets(pred: set[tuple[int, int]], true: set[tuple[int, int]]) -> float:
        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)
        p = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        return float(_f1_from_pr(float(p), float(r)))

    # True CPDAG skeleton + directed arrows
    true_cp_skel: set[tuple[int, int]] = set()
    true_cp_arrows: set[tuple[int, int]] = set()
    if B_true_cpdag is not None:
        # Only count compelled orientations (directed edges) as arrowheads.
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                try:
                    if int(B_true_cpdag[i, j]) == 1 and int(B_true_cpdag[j, i]) == 0:
                        true_cp_arrows.add((i, j))
                except Exception:
                    continue
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                try:
                    if int(B_true_cpdag[i, j]) != 0 or int(B_true_cpdag[j, i]) != 0:
                        true_cp_skel.add((i, j))
                except Exception:
                    continue

    def _cpdag_skeleton(C: "np.ndarray") -> set[tuple[int, int]]:
        s: set[tuple[int, int]] = set()
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                try:
                    if int(C[i, j]) != 0 or int(C[j, i]) != 0:
                        s.add((i, j))
                except Exception:
                    continue
        return s

    def _cpdag_directed_arrows(C: "np.ndarray") -> set[tuple[int, int]]:
        """Return only directed CPDAG edges i->j (ignore undirected edges)."""
        arrows: set[tuple[int, int]] = set()
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                try:
                    if int(C[i, j]) == 1 and int(C[j, i]) == 0:
                        arrows.add((i, j))
                except Exception:
                    continue
        return arrows

    def _canonical_cpdag_key(C: "np.ndarray") -> tuple[int, ...]:
        """Canonicalize CPDAG matrix to a hashable key.

        - Keep directed edges as 1/0.
        - Represent undirected edges with -1 only in upper triangle (i<j).
        """
        Cc = np.array(C, dtype=int, copy=True)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                a = int(Cc[i, j])
                b = int(Cc[j, i])
                # directed i->j
                if a == 1 and b == 0:
                    continue
                # directed j->i
                if b == 1 and a == 0:
                    continue
                # undirected (or any non-zero symmetric form)
                if a != 0 or b != 0:
                    Cc[i, j] = -1
                    Cc[j, i] = 0
        return tuple(int(x) for x in Cc.flatten())

    fd_eval, eval_facts_file = tempfile.mkstemp(suffix="_accepted_remaining.lp", text=True)
    os.close(fd_eval)
    try:
        def _emit(event: str, **payload: Any) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(event, payload)
            except Exception:
                pass

        with open(eval_facts_file, "w") as f:
            # Keep graph-eval robust to encoding-side var-domain tweaks
            # (e.g. var(0..n_vars) vs var(0..n_vars-1)).
            for i in range(int(n_nodes)):
                f.write(f"var({i}).\n")
            for k in keys_in_file_order:
                if k in accepted_keys:
                    f.write(f"{k}.\n")

        import clingo

        deadline = time.perf_counter() + float(timeout_sec)
        indep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        dep_facts: dict[tuple[int, int], set[tuple[int, ...]]] = {}
        for k in accepted_keys:
            ks = str(k or "").strip()
            if not ks or not (ks.startswith("ext_dep(") or ks.startswith("ext_indep(")):
                continue
            try:
                # extractor expects a trailing dot (e.g. ext_dep(...).)
                parse_symbol = ks if ks.endswith(".") else f"{ks}."
                X, S, Y, dep_type = extract_test_elements_from_symbol(parse_symbol)
            except Exception:
                continue
            try:
                X_i, Y_i = int(X), int(Y)
            except Exception:
                continue
            try:
                S_tup = tuple(int(s) for s in S)
            except Exception:
                try:
                    S_tup = tuple(S)
                except Exception:
                    S_tup = ()
            group = indep_facts if str(dep_type) == "ext_indep" else dep_facts
            group.setdefault((X_i, Y_i), set()).add(tuple(S_tup))

        timing: dict[str, Any] = {}
        try:
            ctl = compile_and_ground(
                n_nodes,
                eval_facts_file,
                skeleton_rules_reduction=True,
                weak_constraints=False,
                indep_facts=indep_facts,
                dep_facts=dep_facts,
                opt_mode="ignore",
                out_n=0,
                show=["arrow"],
                pre_grounding=False,
                # IMPORTANT: constrain only the (X,Y,S) tests that are actually
                # present in `accepted_keys`. Without `ext_flag=True`,
                # compile_and_ground will add generic dep/indep rules over the
                # global `set(S)` domain, unintentionally constraining untested
                # conditioning sets and often yielding 0 compatible DAGs.
                ext_flag=True,
                prior_knowledge=None,
                max_path_length=None,
                max_conditioning_size=None,
                collider_tree_depth=None,
                cycle_length=None,
                dump_specific=None,
                threads=int(threads),
                deadline=deadline,
                timing_recorder=timing,
            )
        except TimeoutError:
            return _empty_graph_eval(timed_out=True, include_extrema=include_extrema)
        except Exception:
            out = _empty_graph_eval(timed_out=False, include_extrema=include_extrema)
            cache[key] = out
            return out

        seen: set[frozenset[tuple[int, int]]] = set()
        shds: list[float] = []
        f1s: list[float] = []
        adj_f1s: list[float] = []
        ah_f1s: list[float] = []

        true_dag_key = frozenset((int(a), int(b)) for (a, b) in true_arrows)
        true_dag_in_compat = 0

        cp_seen: set[tuple[int, ...]] = set()
        true_cp_key: tuple[int, ...] | None = None
        true_cpdag_in_compat: int | None = None
        if B_true_cpdag is not None:
            try:
                true_cp_key = _canonical_cpdag_key(B_true_cpdag)
                true_cpdag_in_compat = 0
            except Exception:
                true_cp_key = None
                true_cpdag_in_compat = None
        cp_shds: list[float] = []
        cp_f1s: list[float] = []
        cp_adj_f1s: list[float] = []
        cp_ah_f1s: list[float] = []

        timed_out = False

        try:
            snapshot_every = int(progress_snapshot_every)
        except Exception:
            snapshot_every = 25
        snapshot_every = max(1, snapshot_every)

        def _build_out(timed_out_flag: bool) -> tuple[Any, ...]:
            if not seen:
                base = (
                    0,
                    None,
                    None,
                    None,
                    None,
                    0,
                    0,
                    (int(true_cpdag_in_compat) if true_cpdag_in_compat is not None else None),
                    None,
                    None,
                    None,
                    None,
                )
                if not include_extrema:
                    return (*base, timed_out_flag)
                return (*base, *((None,) * 16), timed_out_flag)

            shd_avg = (sum(shds) / len(shds)) if shds else None
            f1_avg = (sum(f1s) / len(f1s)) if f1s else None
            adj_f1_avg = (sum(adj_f1s) / len(adj_f1s)) if adj_f1s else None
            ah_f1_avg = (sum(ah_f1s) / len(ah_f1s)) if ah_f1s else None

            n_cpdags = int(len(cp_seen)) if cp_seen else 0
            cp_shd_avg = (sum(cp_shds) / len(cp_shds)) if cp_shds else None
            cp_f1_avg = (sum(cp_f1s) / len(cp_f1s)) if cp_f1s else None
            cp_adj_f1_avg = (sum(cp_adj_f1s) / len(cp_adj_f1s)) if cp_adj_f1s else None
            cp_ah_f1_avg = (sum(cp_ah_f1s) / len(cp_ah_f1s)) if cp_ah_f1s else None

            base = (
                int(len(seen)),
                shd_avg,
                f1_avg,
                adj_f1_avg,
                ah_f1_avg,
                int(true_dag_in_compat),
                n_cpdags,
                (int(true_cpdag_in_compat) if true_cpdag_in_compat is not None else None),
                cp_shd_avg,
                cp_f1_avg,
                cp_adj_f1_avg,
                cp_ah_f1_avg,
            )
            if not include_extrema:
                return (*base, timed_out_flag)

            shd_best, shd_worst = _best_worst(shds, higher_is_better=False)
            f1_best, f1_worst = _best_worst(f1s, higher_is_better=True)
            adj_best, adj_worst = _best_worst(adj_f1s, higher_is_better=True)
            ah_best, ah_worst = _best_worst(ah_f1s, higher_is_better=True)
            cp_shd_best, cp_shd_worst = _best_worst(cp_shds, higher_is_better=False)
            cp_f1_best, cp_f1_worst = _best_worst(cp_f1s, higher_is_better=True)
            cp_adj_best, cp_adj_worst = _best_worst(cp_adj_f1s, higher_is_better=True)
            cp_ah_best, cp_ah_worst = _best_worst(cp_ah_f1s, higher_is_better=True)

            return (
                *base,
                shd_best,
                shd_worst,
                f1_best,
                f1_worst,
                adj_best,
                adj_worst,
                ah_best,
                ah_worst,
                cp_shd_best,
                cp_shd_worst,
                cp_f1_best,
                cp_f1_worst,
                cp_adj_best,
                cp_adj_worst,
                cp_ah_best,
                cp_ah_worst,
                timed_out_flag,
            )

        def _on_model(m: "clingo.Model") -> None:
            nonlocal true_dag_in_compat

            try:
                m_syms = m.symbols(shown=True)
            except Exception:
                return
            arrows = model_to_set_of_arrows(m_syms)
            dag_key = frozenset((int(a), int(b)) for (a, b) in arrows)
            if dag_key in seen:
                return
            seen.add(dag_key)

            if dag_key == true_dag_key:
                true_dag_in_compat = 1

            B_est = np.zeros((n_nodes, n_nodes), dtype=int)
            for (a, b) in dag_key:
                if 0 <= a < n_nodes and 0 <= b < n_nodes:
                    B_est[a, b] = 1
            try:
                mt = DAGMetrics(B_est, B_true, sid=False).metrics
            except Exception:
                return
            shd = mt.get("shd", None)
            p = mt.get("precision", None)
            r = mt.get("recall", None)
            try:
                shd_f = float(shd)
            except Exception:
                shd_f = float("nan")
            try:
                p_f = float(p)
            except Exception:
                p_f = float("nan")
            try:
                r_f = float(r)
            except Exception:
                r_f = float("nan")
            f1_f = _f1_from_pr(p_f, r_f) if (math.isfinite(p_f) and math.isfinite(r_f)) else float("nan")
            if math.isfinite(shd_f):
                shds.append(shd_f)
            if math.isfinite(f1_f):
                f1s.append(f1_f)

            # Skeleton (adjacency) F1
            pred_arrows = set((int(a), int(b)) for (a, b) in dag_key if a != b)
            pred_skel: set[tuple[int, int]] = set()
            for (a, b) in pred_arrows:
                x, y = (a, b) if a < b else (b, a)
                pred_skel.add((x, y))
            adj_f1s.append(_f1_from_sets(pred_skel, true_skel))

            # Orientation (arrowhead) F1
            ah_f1s.append(_f1_from_sets(pred_arrows, true_arrows))

            # CPDAG-level metrics
            if B_true_cpdag is not None:
                try:
                    C_est = dag2cpdag(B_est.copy())
                    cp_key = _canonical_cpdag_key(C_est)
                except Exception:
                    C_est = None
                    cp_key = None
                if C_est is not None and cp_key is not None and cp_key not in cp_seen:
                    cp_seen.add(cp_key)
                    try:
                        mt_cp = DAGMetrics(C_est, B_true, sid=False).metrics
                    except Exception:
                        mt_cp = {}
                    try:
                        cp_shd_f = float(mt_cp.get("shd", float("nan")))
                    except Exception:
                        cp_shd_f = float("nan")
                    try:
                        cp_f1_f = float(mt_cp.get("F1", float("nan")))
                    except Exception:
                        cp_f1_f = float("nan")
                    if math.isfinite(cp_shd_f):
                        cp_shds.append(cp_shd_f)
                    if math.isfinite(cp_f1_f):
                        cp_f1s.append(cp_f1_f)

                    pred_cp_skel = _cpdag_skeleton(C_est)
                    pred_cp_arrows = _cpdag_directed_arrows(C_est)
                    # Always compute these (even when the true set is empty), so
                    # we don't emit missing values for valid cases.
                    cp_adj_f1s.append(_f1_from_sets(pred_cp_skel, true_cp_skel))
                    cp_ah_f1s.append(_f1_from_sets(pred_cp_arrows, true_cp_arrows))

            _emit(
                "graph_eval_dag_done",
                n_dags=int(len(seen)),
                n_cpdags=int(len(cp_seen)),
            )
            if int(len(seen)) % int(snapshot_every) == 0:
                _emit(
                    "graph_eval_partial",
                    n_dags=int(len(seen)),
                    n_cpdags=int(len(cp_seen)),
                    out=_build_out(False),
                )

        remaining = max(0.0, deadline - time.perf_counter())
        if remaining <= 0.0:
            timed_out = True
        else:
            handle = ctl.solve(async_=True, on_model=_on_model)
            finished = False
            try:
                finished = bool(handle.wait(timeout=float(remaining)))
            except TypeError:
                finished = bool(handle.wait())
            if not finished:
                timed_out = True
                try:
                    handle.cancel()
                except Exception:
                    pass
                try:
                    handle.wait(1.0)
                except Exception:
                    pass
            try:
                handle.get()
            except Exception:
                pass

        if true_cpdag_in_compat is not None and true_cp_key is not None and true_cp_key in cp_seen:
            true_cpdag_in_compat = 1

        if timed_out:
            out_t = _build_out(True)
            _emit(
                "graph_eval_partial",
                n_dags=int(len(seen)),
                n_cpdags=int(len(cp_seen)),
                out=out_t,
            )
            return out_t

        out = _build_out(False)
        _emit(
            "graph_eval_partial",
            n_dags=int(len(seen)),
            n_cpdags=int(len(cp_seen)),
            out=out,
        )
        cache[key] = out
        return out
    finally:
        try:
            os.remove(eval_facts_file)
        except Exception:
            pass


def graph_eval_from_accepted_wall(
    *,
    n_nodes: int,
    G_true1: Any,
    keys_in_file_order: list[str],
    accepted_keys: set[str],
    timeout_sec: float,
    threads: int,
    cache: dict[frozenset[str], tuple[Any, ...]],
    progress_cb: Callable[[str, dict[str, Any]], None] | None = None,
    include_extrema: bool = False,
) -> tuple[Any, ...]:
    """Graph-eval with a hard wall-clock timeout (sweep-compatible).

    Motivation: clingo's in-process deadline checks control solving, but grounding
    can still exceed budgets because it runs in native code.

    - Reuses the provided cache when available.
    - Does not cache wall-timeout/error sentinel results.
    """

    key = frozenset(accepted_keys or set())
    if key in cache:
        out_cached = cache[key]
        if (not include_extrema and len(out_cached) == _GE_BASE_LEN) or (
            include_extrema and len(out_cached) == _GE_EXT_LEN
        ):
            return out_cached

    # Fast-path: disabled or no budget.
    try:
        if float(timeout_sec) <= 0.0:
            return _empty_graph_eval(timed_out=True, include_extrema=include_extrema)
    except Exception:
        return _empty_graph_eval(timed_out=True, include_extrema=include_extrema)

    # Hard wall-time budget: keep grace small so timeouts are tight.
    wall_timeout = max(0.1, float(timeout_sec) + 0.2)

    # Try to use fork when available (faster, and allows nested targets on Linux).
    try:
        ctx = multiprocessing.get_context("fork")
    except Exception:
        ctx = multiprocessing.get_context()

    q: Any = ctx.Queue()

    def _emit(event: str, **payload: Any) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(event, payload)
        except Exception:
            pass

    def _worker() -> None:
        try:
            out = graph_eval_from_accepted(
                n_nodes=n_nodes,
                G_true1=G_true1,
                keys_in_file_order=keys_in_file_order,
                accepted_keys=accepted_keys,
                timeout_sec=float(timeout_sec),
                threads=int(threads),
                cache={},
                include_extrema=bool(include_extrema),
            )
            q.put(("ok", out))
        except BaseException as e:
            q.put(("err", (repr(e), traceback.format_exc())))

    _emit(
        "graph_eval_start",
        accepted=int(len(accepted_keys or [])),
        timeout_sec=round(float(timeout_sec), 3),
        wall_timeout_sec=round(float(wall_timeout), 3),
    )
    t0_ge = time.perf_counter()
    proc = ctx.Process(target=_worker)
    proc.daemon = True
    proc.start()
    proc.join(timeout=float(wall_timeout))
    wall_ge = time.perf_counter() - t0_ge

    if proc.is_alive():
        _emit(
            "graph_eval_wall_timeout",
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
                proc.kill()  # type: ignore[attr-defined]
                proc.join(timeout=0.5)
        except Exception:
            pass
        return _empty_graph_eval(timed_out=True, include_extrema=include_extrema)

    try:
        st, payload = q.get(timeout=0.5)
    except Exception:
        st, payload = ("err", ("No graph-eval payload", ""))

    if st != "ok":
        _emit(
            "graph_eval_error",
            wall_elapsed_s=round(float(wall_ge), 3),
            err=str(payload[0]) if isinstance(payload, (list, tuple)) and payload else str(payload),
        )
        return _empty_graph_eval(timed_out=True, include_extrema=include_extrema)

    out = payload
    expected_len = _GE_EXT_LEN if include_extrema else _GE_BASE_LEN
    # Cache only successful runs (non-timeout + structurally valid).
    if (
        isinstance(out, (list, tuple))
        and len(out) == expected_len
        and (not bool(out[-1]))
        and (out[0] is not None)
    ):
        cache[key] = out  # type: ignore[assignment]

    _emit(
        "graph_eval_done",
        accepted=int(len(accepted_keys or [])),
        wall_elapsed_s=round(float(wall_ge), 3),
        timed_out=bool(out[-1]) if isinstance(out, (list, tuple)) and len(out) == expected_len else "?",
    )
    return out
