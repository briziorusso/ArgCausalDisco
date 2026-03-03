"""CausalABA variant: Bayes-ball grounding + soft observational constraints.

This solver is for performance attribution.

- Ground once (like the incremental solver): Bayes-ball encoding derives `ap3(X,Y,S)`.
- Do *not* run a remove-until-SAT loop.
- Instead, treat each observational statement as a soft constraint (weak constraint):
  - dep(X,Y,S) prefers `ap3(X,Y,S)` to hold
  - indep(X,Y,S) prefers `ap3(X,Y,S)` to *not* hold

This answers: how fast/cheap is "one optimize solve" compared to binary-search
SAT checks and compared to baseline path-enumeration grounding.

Note: this changes the objective semantics compared to baseline/incremental
removal-minimization; it is intentionally a different strategy.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from clingo import Function, Number
from clingo.control import Control

logger = logging.getLogger(__name__)

try:
    from .causalaba import extract_test_elements_from_symbol, PriorKnowledge
except ImportError:  # pragma: no cover
    from causalaba import extract_test_elements_from_symbol, PriorKnowledge


def _cond_to_term(S) -> str:
    if isinstance(S, str):
        return S
    try:
        S = tuple(sorted(S))
    except Exception:
        S = ()
    if not S:
        return "empty"
    return "s" + "y".join(str(i) for i in S)


def _iter_powerset(nodes: range):
    items = list(nodes)
    n = len(items)
    for mask in range(1 << n):
        subset = tuple(items[i] for i in range(n) if (mask >> i) & 1)
        yield subset


def _solve_with_timeout(ctl, *, solve_timeout: float | None, on_model):
    if solve_timeout is None:
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                on_model(model)
        return True

    handle = ctl.solve(async_=True, on_model=on_model)
    finished = handle.wait(solve_timeout)
    if not finished:
        try:
            handle.cancel()
        except Exception:
            pass
        cancelled_finished = False
        try:
            cancelled_finished = bool(handle.wait(1.0))
        except TypeError:
            pass
        if cancelled_finished:
            try:
                handle.get()
            except Exception:
                pass
        return False
    handle.get()
    return True


def CausalABA(
    n_nodes: int,
    facts_location: str = "",
    print_models: bool = True,
    skeleton_rules_reduction: bool = False,
    weak_constraints: bool = True,
    fact_pct: float = 1.0,
    set_indep_facts: bool = False,
    opt_mode: str = "optN",
    opt_strategy: str | None = None,
    out_n: int = 0,
    search_for_models: str = "first",
    show: list[str] | None = None,
    pre_grounding: bool = False,
    disable_reground: bool = False,
    prior_knowledge: PriorKnowledge | None = None,
    return_statistics: bool = False,
    max_path_length: int | None = None,
    max_conditioning_size: int | None = None,
    collider_tree_depth: int | None = None,
    cycle_length: int | None = None,
    threads: int | None = None,
    solve_timeout: float | None = None,
) -> list:
    show = show or ["arrow"]
    if pre_grounding:
        raise ValueError("causalaba_weakc does not support pre_grounding")
    if disable_reground:
        # Not meaningful here; keep arg for signature compatibility.
        pass
    if set_indep_facts:
        # Not meaningful here; keep arg for signature compatibility.
        pass
    if search_for_models not in ("first", "No"):
        raise ValueError("causalaba_weakc only supports search_for_models='first' or 'No'")

    logger.info("Running CausalABA (weakc-only soft facts)")

    t_compile0 = time.perf_counter()
    profile: dict = {
        "compile_sec_total": 0.0,
        "compile_sec_last": 0.0,
        "ground_sec_total": 0.0,
        "ground_sec_last": 0.0,
        "solve_sec_total": 0.0,
        "solve_sec_last": 0.0,
        "compile_calls": 0,
        "ground_calls_internal": 0,
        "solve_calls_internal": 0,
        "final_opt_sec": 0.0,
        "peak_after_compile_bytes": None,
        "peak_after_ground_bytes": None,
        "peak_after_solve_bytes_last": None,
        "peak_after_solve_bytes_max": None,
    }

    # Read observed statements + optional weights.
    facts: list[tuple[int, tuple[int, ...], int, str, float]] = []  # (X,S,Y,kind,I)
    facts_loc = facts_location.replace(".lp", "_I.lp") if (weak_constraints and facts_location) else facts_location
    if facts_loc:
        with open(facts_loc, "r") as f:
            for line in f:
                if "dep" not in line or line.startswith("%"):
                    continue
                line_clean = line.replace("#external ", "").strip()
                I = 1.0
                if " I=" in line_clean:
                    stmt, rest = line_clean.split(" I=")
                    try:
                        I = float(rest.split(",")[0])
                    except Exception:
                        I = 1.0
                    X, S, Y, dep_type = extract_test_elements_from_symbol(stmt)
                else:
                    X, S, Y, dep_type = extract_test_elements_from_symbol(line_clean)
                kind = "dep" if "dep" in dep_type and "indep" not in dep_type else "indep"
                facts.append((int(X), tuple(sorted(S)), int(Y), kind, float(I)))

    facts.sort(key=lambda t: t[4], reverse=True)
    if 0 < fact_pct < 1.0 and facts:
        keep = max(1, int(round(fact_pct * len(facts))))
        facts = facts[:keep]

    # Conditioning sets to emit `in/2`.
    if skeleton_rules_reduction:
        cond_sets = sorted({S for (_x, S, _y, _k, _I) in facts})
    else:
        cond_sets = list(_iter_powerset(range(n_nodes)))
    if max_conditioning_size is not None:
        cond_sets = [S for S in cond_sets if len(S) <= int(max_conditioning_size)]

    cpu_count = min(os.cpu_count() or 1, 64)
    threads = threads or cpu_count
    control_args = [f"-t {threads}", "--warn=none"]
    if cycle_length is not None:
        control_args += [f"-c l_cyc={int(cycle_length)}"]
    if collider_tree_depth is not None:
        control_args += [f"-c l_b={int(collider_tree_depth)}"]
    ctl = Control(control_args)
    ctl.configuration.solve.parallel_mode = threads
    ctl.configuration.solve.models = out_n
    ctl.configuration.solver.seed = "2024"
    ctl.configuration.solve.opt_mode = opt_mode
    if opt_strategy:
        try:
            ctl.configuration.solver.opt_strategy = str(opt_strategy)
        except Exception:
            # Some clingo builds expose opt_strategy only via CLI; ignore if unavailable.
            pass

    bounded_encoding_active = (cycle_length is not None and cycle_length > 0) or (
        collider_tree_depth is not None and collider_tree_depth > 0
    )
    enc = "causalaba_bounded.lp" if bounded_encoding_active else "causalaba.lp"
    ctl.load(str(Path(__file__).resolve().parent / "encodings" / enc))

    # Generated facts: in/2 + qpair/3 + obs_* + weak constraints.
    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False) as gen_f:
        gen_path = Path(gen_f.name)
        gen_f.write("#program specific.\n")
        for S in cond_sets:
            s_term = _cond_to_term(S)
            for s in S:
                gen_f.write(f"in({s},{s_term}).\n")
        for X, S, Y, _kind, _I in facts:
            a, b = (X, Y) if X < Y else (Y, X)
            gen_f.write(f"qpair({a},{b},{_cond_to_term(S)}).\n")
        for idx, (X, S, Y, kind, I) in enumerate(facts):
            a, b = (X, Y) if X < Y else (Y, X)
            s_term = _cond_to_term(S)
            w = int(round(max(0.0, float(I)) * 1000.0)) or 1
            if kind == "dep":
                gen_f.write(f"obs_dep({a},{b},{s_term}).\n")
                gen_f.write(f":~ obs_dep({a},{b},{s_term}), not ap3({a},{b},{s_term}). [{w}@1,{idx}]\n")
            else:
                gen_f.write(f"obs_indep({a},{b},{s_term}).\n")
                gen_f.write(f":~ obs_indep({a},{b},{s_term}), ap3({a},{b},{s_term}). [{w}@1,{idx}]\n")

    try:
        ctl.load(str(gen_path))
    finally:
        try:
            gen_path.unlink(missing_ok=True)
        except Exception:
            pass

    if prior_knowledge is not None:
        for (Xf, Yf) in prior_knowledge.forbidden:
            ctl.add("specific", [], f":- arrow({Xf},{Yf}).")
        for (Xr, Yr) in prior_knowledge.required:
            ctl.add("specific", [], f"arrow({Xr},{Yr}).")

    # Bayes-ball rules to derive ap3/3.
    bb: list[str] = []
    bb.append("bb_start(X,S) :- qpair(X,_,S), set(S), var(X).")
    bb.append("anc_obs(N,S) :- in(N,S), set(S), var(N).")
    if collider_tree_depth is not None and collider_tree_depth > 0:
        bb.append("anc_obs(N,S) :- in(Z,S), dpath_k(N,Z,K), set(S), var(N), var(Z), N!=Z, K=1..l_b.")
    else:
        bb.append("anc_obs(N,S) :- in(Z,S), dpath(N,Z), set(S), var(N), var(Z), N!=Z.")
    bb.append("is_collider(N) :- arrow(P,N), arrow(Q,N), P<Q, P!=Q, var(P), var(Q), var(N).")

    if max_path_length is not None:
        l_ap = int(max_path_length)
        bb.append(f"#const l_ap={l_ap}.")
        bb.append("bb_u(X,X,S,0) :- bb_start(X,S).")
        bb.append("bb_d(X,X,S,0) :- bb_start(X,S).")
        bb.append("bb_u(X,P,S,T1) :- bb_u(X,N,S,T), T < l_ap, T1 = T+1, not in(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb.append("bb_u(X,P,S,T1) :- bb_d(X,N,S,T), T < l_ap, T1 = T+1, is_collider(N), anc_obs(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb.append("bb_d(X,C,S,T1) :- bb_u(X,N,S,T), T < l_ap, T1 = T+1, not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb.append("bb_d(X,C,S,T1) :- bb_d(X,N,S,T), T < l_ap, T1 = T+1, not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_u(X,Y,S,_).")
        bb.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_d(X,Y,S,_).")
    else:
        bb.append("bb_u(X,X,S) :- bb_start(X,S).")
        bb.append("bb_d(X,X,S) :- bb_start(X,S).")
        bb.append("bb_u(X,P,S) :- bb_u(X,N,S), not in(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb.append("bb_u(X,P,S) :- bb_d(X,N,S), is_collider(N), anc_obs(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb.append("bb_d(X,C,S) :- bb_u(X,N,S), not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb.append("bb_d(X,C,S) :- bb_d(X,N,S), not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_u(X,Y,S).")
        bb.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_d(X,Y,S).")
    bb.append("ap(X,Y,bb,S) :- ap3(X,Y,S), qpair(X,Y,S).")
    ctl.add("specific", [], "\n".join(bb))

    if "arrow" in show:
        ctl.add("base", [], "#show arrow/2.")
    if "ap" in show:
        ctl.add("base", [], "#show ap/4.")

    logger.info("   Grounding full base program...")
    profile["compile_calls"] += 1
    profile["compile_sec_last"] = max(0.0, time.perf_counter() - t_compile0)
    profile["compile_sec_total"] += profile["compile_sec_last"]

    try:
        import tracemalloc as _tracemalloc

        if _tracemalloc.is_tracing():
            _cur, _peak = _tracemalloc.get_traced_memory()
            profile["peak_after_compile_bytes"] = int(_peak)
    except Exception:
        pass

    t_g0 = time.perf_counter()
    ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes - 1)])])
    t_g1 = time.perf_counter()
    profile["ground_calls_internal"] += 1
    profile["ground_sec_last"] = max(0.0, t_g1 - t_g0)
    profile["ground_sec_total"] += profile["ground_sec_last"]
    try:
        import tracemalloc as _tracemalloc

        if _tracemalloc.is_tracing():
            _cur, _peak = _tracemalloc.get_traced_memory()
            profile["peak_after_ground_bytes"] = int(_peak)
    except Exception:
        pass

    models: list[list] = []

    def _on_model(model):
        if getattr(model, "optimality_proven", False) or opt_mode not in ("opt", "optN"):
            models.append(model.symbols(shown=True))
        if print_models:
            logger.info("Answer: %s", model)

    logger.info("   Solving...")
    t_s0 = time.perf_counter()
    _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_on_model)
    t_s1 = time.perf_counter()
    profile["solve_calls_internal"] += 1
    profile["solve_sec_last"] = max(0.0, t_s1 - t_s0)
    profile["solve_sec_total"] += profile["solve_sec_last"]
    profile["final_opt_sec"] = profile["solve_sec_last"]
    try:
        import tracemalloc as _tracemalloc

        if _tracemalloc.is_tracing():
            _cur, _peak = _tracemalloc.get_traced_memory()
            profile["peak_after_solve_bytes_last"] = int(_peak)
            prev = profile.get("peak_after_solve_bytes_max")
            profile["peak_after_solve_bytes_max"] = int(_peak) if prev is None else max(int(prev), int(_peak))
    except Exception:
        pass

    if return_statistics:
        return [models, False, ctl.statistics, 0, profile]
    return [models, False]
