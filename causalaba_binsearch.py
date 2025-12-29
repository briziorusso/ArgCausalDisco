"""Baseline CausalABA grounding + binary-search removal.

This variant is meant for benchmarking: it keeps the *baseline* grounding strategy
(explicit active-path enumeration via compile_and_ground) but replaces the
baseline linear remove-until-SAT loop with the incremental solver's monotone
binary search using SAT-only checks.

Goal: isolate the impact of the *search strategy* from the impact of the
*grounding / Bayes-ball encoding*.
"""

from __future__ import annotations

import logging
import time

from clingo import Function, Number

try:
    from .causalaba import (
        compile_and_ground,
        extract_test_elements_from_symbol,
        PriorKnowledge,
        _solve_with_timeout,
        CausalABA as _baseline_causalaba,
    )
except ImportError:  # pragma: no cover
    from causalaba import (
        compile_and_ground,
        extract_test_elements_from_symbol,
        PriorKnowledge,
        _solve_with_timeout,
        CausalABA as _baseline_causalaba,
    )

logger = logging.getLogger(__name__)


def _cond_to_symbol(S) -> Function:
    try:
        S = tuple(sorted(S))
    except Exception:
        S = ()
    if not S:
        return Function("empty")
    return Function("s" + "y".join(str(i) for i in S))


def CausalABA(
    n_nodes: int,
    facts_location: str = "",
    print_models: bool = True,
    skeleton_rules_reduction: bool = False,
    weak_constraints: bool = False,
    fact_pct: float = 1.0,
    set_indep_facts: bool = False,
    opt_mode: str = "optN",
    out_n: int = 0,
    search_for_models: str = "No",
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
    verbosity: int = 0,
) -> list:
    """CausalABA with baseline grounding but binary-search removal.

    For compatibility, this follows the baseline API and return shape.
    """

    show = show or ["arrow"]
    v = int(verbosity or 0)

    logger.info("Running CausalABA (baseline-grounding + binsearch)")

    deadline = (time.perf_counter() + float(solve_timeout)) if solve_timeout is not None else None

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
        "paths_added_total": 0,
        "paths_added_last": 0,
        "sat_check_sec_total": 0.0,
        "sat_check_sec_last": 0.0,
        "final_opt_sec": 0.0,
        "peak_after_compile_bytes": None,
        "peak_after_ground_bytes": None,
        "peak_after_solve_bytes_last": None,
        "peak_after_solve_bytes_max": None,
    }

    indep_facts: dict[tuple, set[tuple]] = {}
    dep_facts: dict[tuple, set[tuple]] = {}
    facts: list[tuple] = []
    ext_flag = False

    if facts_location:
        facts_loc = facts_location.replace(".lp", "_I.lp") if weak_constraints else facts_location
        with open(facts_loc, "r") as file:
            for line in file:
                if "dep" not in line or line.startswith("%"):
                    continue
                line_clean = line.replace("#external ", "").replace("\n", "")
                if "ext_" in line_clean:
                    ext_flag = True
                if weak_constraints:
                    statement, Is = line_clean.split(" I=")
                    I, truth = Is.split(",")
                    X, S, Y, dep_type = extract_test_elements_from_symbol(statement)
                    facts.append((X, S, Y, dep_type, statement, float(I), truth))
                else:
                    X, S, Y, dep_type = extract_test_elements_from_symbol(line_clean)
                    facts.append((X, S, Y, dep_type, line_clean, float("nan"), "unknown"))

                condition_set = tuple(sorted(S))
                facts_group = indep_facts if "indep" in line_clean else dep_facts
                if (X, Y) not in facts_group:
                    facts_group[(X, Y)] = set()
                facts_group[(X, Y)].add(condition_set)

    facts = sorted(facts, key=lambda x: x[5], reverse=True)
    fact_syms = [Function(f[3], [Number(f[0]), Number(f[2]), _cond_to_symbol(f[1])]) for f in facts]

    ctl = compile_and_ground(
        n_nodes,
        facts_location,
        skeleton_rules_reduction,
        weak_constraints,
        indep_facts,
        dep_facts,
        opt_mode,
        out_n,
        show,
        pre_grounding,
        ext_flag,
        prior_knowledge,
        max_path_length=max_path_length,
        max_conditioning_size=max_conditioning_size,
        collider_tree_depth=collider_tree_depth,
        cycle_length=cycle_length,
        threads=threads,
        deadline=deadline,
    )

    try:
        p = getattr(ctl, "_causalaba_profile", None)
        if isinstance(p, dict):
            profile["compile_calls"] += 1
            profile["ground_calls_internal"] += 1
            profile["compile_sec_last"] = float(p.get("compile_sec", 0.0))
            profile["ground_sec_last"] = float(p.get("ground_sec", 0.0))
            profile["compile_sec_total"] += profile["compile_sec_last"]
            profile["ground_sec_total"] += profile["ground_sec_last"]
            profile["paths_added_last"] = int(p.get("paths_added", 0))
            profile["paths_added_total"] += profile["paths_added_last"]
            profile["peak_after_compile_bytes"] = p.get("peak_after_compile_bytes")
            profile["peak_after_ground_bytes"] = p.get("peak_after_ground_bytes")
    except Exception:
        pass

    # Normalize mode names: 'first_dep' is an alias for 'dep_first'
    if search_for_models == "first_dep":
        search_for_models = "dep_first"

    # Default behavior mirrors baseline unless special binsearch modes are requested.
    if search_for_models not in ("first", "dep_first"):
        return _baseline_causalaba(
            n_nodes,
            facts_location,
            print_models=print_models,
            skeleton_rules_reduction=skeleton_rules_reduction,
            weak_constraints=weak_constraints,
            fact_pct=fact_pct,
            set_indep_facts=set_indep_facts,
            opt_mode=opt_mode,
            out_n=out_n,
            search_for_models=search_for_models,
            show=show,
            pre_grounding=pre_grounding,
            disable_reground=disable_reground,
            prior_knowledge=prior_knowledge,
            return_statistics=return_statistics,
            max_path_length=max_path_length,
            max_conditioning_size=max_conditioning_size,
            collider_tree_depth=collider_tree_depth,
            cycle_length=cycle_length,
            threads=threads,
            solve_timeout=solve_timeout,
        )

    # Mode dispatch: 'first' vs 'dep_first'
    if search_for_models == 'first':
        # --- `search_for_models == 'first'` ---
        # Start with all facts active (baseline behavior).
        for fact, sym in zip(facts, fact_syms):
            ctl.assign_external(sym, True)

        models: list[list] = []

        def _collect_opt_models(model):
            if getattr(model, "optimality_proven", False):
                models.append(model.symbols(shown=True))
            if print_models:
                logger.info("Answer: %s", model)

        logger.info("   Solving...")
        t_s0 = time.perf_counter()
        finished = _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_collect_opt_models)
        t_s1 = time.perf_counter()
        profile["solve_calls_internal"] += 1
        profile["solve_sec_last"] = max(0.0, t_s1 - t_s0)
        profile["solve_sec_total"] += profile["solve_sec_last"]
        try:
            import tracemalloc as _tracemalloc

            if _tracemalloc.is_tracing():
                _cur, _peak = _tracemalloc.get_traced_memory()
                profile["peak_after_solve_bytes_last"] = int(_peak)
                prev = profile.get("peak_after_solve_bytes_max")
                profile["peak_after_solve_bytes_max"] = int(_peak) if prev is None else max(int(prev), int(_peak))
        except Exception:
            pass
        n_models = int(ctl.statistics["summary"]["models"]["enumerated"])
        logger.info("[first] found n_models=%s", n_models)

        if n_models > 0 and models:
            if return_statistics:
                return [models, False, ctl.statistics, 0, profile]
            return [models, False]

        # Fall through to binary search if first solve found nothing
        logger.info("[first] no models found, proceeding to binary search...")

    elif search_for_models == 'dep_first':
        # This strategy avoids regrounding until an indep must be dropped. We line-search
        # contiguous dep blocks separated by indeps; only when removing an indep is
        # unavoidable do we reground with that indep excised.

        def _sat_only() -> bool:
            cfg = None
            prev_opt_mode = None
            prev_models = None
            try:
                try:
                    cfg = ctl.configuration.solve
                    prev_opt_mode = getattr(cfg, "opt_mode", None)
                    prev_models = getattr(cfg, "models", None)
                    try:
                        cfg.opt_mode = "ignore"
                    except Exception:
                        pass
                    try:
                        cfg.models = 1
                    except Exception:
                        pass
                except Exception:
                    cfg = None

                found = {"sat": False}

                def _on_model_sat(_m):
                    found["sat"] = True

                t0 = time.perf_counter()
                done = _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_on_model_sat)
                t1 = time.perf_counter()
                profile["solve_calls_internal"] += 1
                profile["solve_sec_last"] = max(0.0, t1 - t0)
                profile["solve_sec_total"] += profile["solve_sec_last"]
                profile["sat_check_sec_last"] = profile["solve_sec_last"]
                profile["sat_check_sec_total"] += profile["sat_check_sec_last"]
                if found["sat"]:
                    return True
                if not done:
                    return False
                try:
                    res = ctl.solve()
                    return bool(getattr(res, "satisfiable", False))
                except Exception:
                    return False
            finally:
                if cfg is not None:
                    try:
                        if prev_opt_mode is not None:
                            cfg.opt_mode = prev_opt_mode
                    except Exception:
                        pass
                    try:
                        if prev_models is not None:
                            cfg.models = prev_models
                    except Exception:
                        pass

        def _segments() -> list[tuple[list[int], int | None]]:
            segments: list[tuple[list[int], int | None]] = []
            idx = len(facts) - 1
            while idx >= 0:
                block: list[int] = []
                while idx >= 0 and facts[idx][3] in ("dep", "ext_dep"):
                    block.append(idx)
                    idx -= 1
                indep_idx: int | None = None
                if idx >= 0 and facts[idx][3] in ("indep", "ext_indep"):
                    indep_idx = idx
                    idx -= 1
                segments.append((list(reversed(block)), indep_idx))
            return segments

        def _apply_block(block: list[int], remove_last: int) -> None:
            cutoff = max(0, len(block) - int(remove_last or 0))
            for j, idx in enumerate(block):
                val = True if j < cutoff else None
                ctl.assign_external(fact_syms[idx], val)

        def _reset_all_true() -> None:
            for sym in fact_syms:
                ctl.assign_external(sym, True)

        while True:
            _reset_all_true()
            segments = _segments()
            progressed = False

            for block, indep_idx in segments:
                # If there is a trailing dep block, binary search minimal removal
                # within the block without touching indeps.
                if block:
                    _apply_block(block, len(block))
                    if _sat_only():
                        lo, hi = 0, len(block)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            _apply_block(block, mid)
                            if _sat_only():
                                hi = mid
                            else:
                                lo = mid + 1
                        _apply_block(block, lo)

                        # Final optimize solve with current assignments.
                        models: list[list] = []

                        def _on_model_final(model):
                            if getattr(model, "optimality_proven", False):
                                models.append(model.symbols(shown=True))
                            if print_models:
                                logger.info("Answer: %s", model)

                        _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_on_model_final)
                        if return_statistics:
                            try:
                                stats = getattr(ctl, "statistics", None)
                            except Exception:
                                stats = None
                            return [models, False, stats, 0, profile]
                        return [models, False]

                # No satisfiable model even after dropping this dep block; drop the
                # preceding indep (if any) and reground, then continue.
                if indep_idx is None:
                    if return_statistics:
                        try:
                            stats = getattr(ctl, "statistics", None)
                        except Exception:
                            stats = None
                        return [[], False, stats, 0, profile]
                    return [[], False]

                X, S, Y, dep_type, *_rest = facts[indep_idx]
                key = (X, Y)
                tupS = tuple(sorted(S))
                if dep_type in ("indep", "ext_indep") and key in indep_facts:
                    indep_facts[key].discard(tupS)
                    if not indep_facts[key]:
                        del indep_facts[key]

                facts.pop(indep_idx)
                if v >= 2:
                    logger.debug("[dep_first] Recompiling after removing indep %s", (X, S, Y))
                elif v >= 1:
                    logger.info("[dep_first] reground after indep removal")
                ctl = compile_and_ground(
                    n_nodes,
                    facts_location,
                    skeleton_rules_reduction,
                    weak_constraints,
                    indep_facts,
                    dep_facts,
                    opt_mode,
                    out_n,
                    show,
                    pre_grounding,
                    ext_flag,
                    prior_knowledge,
                    max_path_length=max_path_length,
                    max_conditioning_size=max_conditioning_size,
                    collider_tree_depth=collider_tree_depth,
                    cycle_length=cycle_length,
                    threads=threads,
                    deadline=deadline,
                )
                try:
                    p = getattr(ctl, "_causalaba_profile", None)
                    if isinstance(p, dict):
                        profile["compile_calls"] += 1
                        profile["ground_calls_internal"] += 1
                        profile["compile_sec_last"] = float(p.get("compile_sec", 0.0))
                        profile["ground_sec_last"] = float(p.get("ground_sec", 0.0))
                        profile["compile_sec_total"] += profile["compile_sec_last"]
                        profile["ground_sec_total"] += profile["ground_sec_last"]
                        profile["paths_added_last"] = int(p.get("paths_added", 0))
                        profile["paths_added_total"] += profile["paths_added_last"]
                        profile["peak_after_compile_bytes"] = p.get("peak_after_compile_bytes")
                        profile["peak_after_ground_bytes"] = p.get("peak_after_ground_bytes")
                except Exception:
                    pass

                fact_syms = [Function(f[3], [Number(f[0]), Number(f[2]), _cond_to_symbol(f[1])]) for f in facts]
                progressed = True
                break  # restart with updated ctl/facts after reground

            if not progressed:
                if return_statistics:
                    try:
                        stats = getattr(ctl, "statistics", None)
                    except Exception:
                        stats = None
                    return [[], False, stats, 0, profile]
                return [[], False]

    else:
        # Unsupported mode; shouldn't reach here
        logger.error("Unsupported search_for_models: %s", search_for_models)
        return [[], False]

    # --- Fallback: binary search (only reached if 'first' mode found no models) ---
    # SAT-only check helper (avoid optimizing during search)
    def _apply_removed(removed: int) -> None:
        cutoff = max(0, len(facts) - int(removed or 0))
        for idx, sym in enumerate(fact_syms):
            val = True if idx < cutoff else None
            ctl.assign_external(sym, val)

    def _is_satisfiable() -> bool:
        cfg = None
        prev_opt_mode = None
        prev_models = None
        try:
            try:
                cfg = ctl.configuration.solve
                prev_opt_mode = getattr(cfg, "opt_mode", None)
                prev_models = getattr(cfg, "models", None)
                try:
                    cfg.opt_mode = "ignore"
                except Exception:
                    pass
                try:
                    cfg.models = 1
                except Exception:
                    pass
            except Exception:
                cfg = None

            found = {"sat": False}

            def _on_model_sat(_m):
                found["sat"] = True

            t0 = time.perf_counter()
            done = _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_on_model_sat)
            t1 = time.perf_counter()
            profile["solve_calls_internal"] += 1
            profile["solve_sec_last"] = max(0.0, t1 - t0)
            profile["solve_sec_total"] += profile["solve_sec_last"]
            profile["sat_check_sec_last"] = profile["solve_sec_last"]
            profile["sat_check_sec_total"] += profile["sat_check_sec_last"]
            try:
                import tracemalloc as _tracemalloc

                if _tracemalloc.is_tracing():
                    _cur, _peak = _tracemalloc.get_traced_memory()
                    profile["peak_after_solve_bytes_last"] = int(_peak)
                    prev = profile.get("peak_after_solve_bytes_max")
                    profile["peak_after_solve_bytes_max"] = int(_peak) if prev is None else max(int(prev), int(_peak))
            except Exception:
                pass
            if found["sat"]:
                return True
            if not done:
                return False
            try:
                res = ctl.solve()
                return bool(getattr(res, "satisfiable", False))
            except Exception:
                return False
        finally:
            if cfg is not None:
                try:
                    if prev_opt_mode is not None:
                        cfg.opt_mode = prev_opt_mode
                except Exception:
                    pass
                try:
                    if prev_models is not None:
                        cfg.models = prev_models
                except Exception:
                    pass

    # Monotone binary search over removed count.
    lo, hi = 0, len(facts)
    while lo < hi:
        mid = (lo + hi) // 2
        _apply_removed(mid)
        if _is_satisfiable():
            hi = mid
        else:
            lo = mid + 1

    remove_n = lo
    logger.info("[remove] minimal satisfiable removal is %s", remove_n)

    # Optionally reground once (baseline strategy) after we know the final removed set.
    if skeleton_rules_reduction and not disable_reground and remove_n > 0:
        # Update groups to reflect removed facts.
        removed_facts = facts[-remove_n:]
        reground = False
        for X, S, Y, dep_type, _stmt, *_rest in removed_facts:
            group = indep_facts if dep_type == "ext_indep" else dep_facts
            key = (X, Y)
            if key in group:
                group[key].discard(tuple(sorted(S)))
                if not group[key]:
                    del group[key]
                    if ext_flag is False or dep_type == "ext_indep":
                        reground = True
        if reground:
            logger.info("Recompiling and regrounding...")
            ctl = compile_and_ground(
                n_nodes,
                facts_location,
                skeleton_rules_reduction,
                weak_constraints,
                indep_facts,
                dep_facts,
                opt_mode,
                out_n,
                show,
                pre_grounding,
                ext_flag,
                prior_knowledge,
                max_path_length=max_path_length,
                max_conditioning_size=max_conditioning_size,
                collider_tree_depth=collider_tree_depth,
                cycle_length=cycle_length,
                threads=threads,
                deadline=deadline,
            )
            try:
                p = getattr(ctl, "_causalaba_profile", None)
                if isinstance(p, dict):
                    profile["compile_calls"] += 1
                    profile["ground_calls_internal"] += 1
                    profile["compile_sec_last"] = float(p.get("compile_sec", 0.0))
                    profile["ground_sec_last"] = float(p.get("ground_sec", 0.0))
                    profile["compile_sec_total"] += profile["compile_sec_last"]
                    profile["ground_sec_total"] += profile["ground_sec_last"]
                    profile["paths_added_last"] = int(p.get("paths_added", 0))
                    profile["paths_added_total"] += profile["paths_added_last"]
                    profile["peak_after_compile_bytes"] = p.get("peak_after_compile_bytes")
                    profile["peak_after_ground_bytes"] = p.get("peak_after_ground_bytes")
            except Exception:
                pass
            # Rebuild syms to match same order.
            fact_syms = [Function(f[3], [Number(f[0]), Number(f[2]), _cond_to_symbol(f[1])]) for f in facts]

    _apply_removed(remove_n)

    # Final optimize solve
    models = []
    logger.info("   Solving...")

    def _on_model_post(model):
        if getattr(model, "optimality_proven", False):
            models.append(model.symbols(shown=True))
        if print_models:
            logger.info("Answer: %s", model)

    if solve_timeout is not None and not finished:
        # If the initial solve already timed out, don't keep grinding.
        if return_statistics:
            return [[], False, ctl.statistics, remove_n, profile]
        return [[], False]

    t_f0 = time.perf_counter()
    _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_on_model_post)
    t_f1 = time.perf_counter()
    profile["solve_calls_internal"] += 1
    profile["solve_sec_last"] = max(0.0, t_f1 - t_f0)
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
        return [models, False, ctl.statistics, remove_n, profile]
    return [models, False]
