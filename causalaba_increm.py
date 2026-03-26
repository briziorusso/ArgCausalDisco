import logging
import math
import rustworkx as rx
from clingo import Function, Number, Symbol
import os
from pathlib import Path
import re
import tracemalloc
import time
from typing import Any, cast

logger = logging.getLogger(__name__)

try:
    from .utils.progress import start_heartbeat as _start_heartbeat, fmt_hhmmss as _fmt_hhmmss
except ImportError:  # pragma: no cover
    from utils.progress import start_heartbeat as _start_heartbeat, fmt_hhmmss as _fmt_hhmmss
try:
    from .utils import mem as _mem
    from .utils.satcheck import (
        SolveStatus as _SolveStatus,
        SatcheckSearchCheckpoint as _SatcheckSearchCheckpoint,
        SatcheckThreadTuner as _SatcheckThreadTuner,
        build_search_fingerprint as _build_search_fingerprint,
        classify_solve_result as _classify_solve_result,
        run_remove_search as _run_remove_search,
        satcheck_peak_kib as _satcheck_peak_kib,
        solve_with_timeout as _solve_with_timeout,
    )
except ImportError:  # pragma: no cover
    from utils import mem as _mem
    from utils.satcheck import (
        SolveStatus as _SolveStatus,
        SatcheckSearchCheckpoint as _SatcheckSearchCheckpoint,
        SatcheckThreadTuner as _SatcheckThreadTuner,
        build_search_fingerprint as _build_search_fingerprint,
        classify_solve_result as _classify_solve_result,
        run_remove_search as _run_remove_search,
        satcheck_peak_kib as _satcheck_peak_kib,
        solve_with_timeout as _solve_with_timeout,
    )


def _tm_stage(msg: str) -> None:
    """Log current/peak tracemalloc stats when enabled via env var.

    This is intended for diagnosing benchmark memory spikes. It is a no-op
    unless tracemalloc is already tracing and CAUSALABA_TMEM is set.
    """
    if not os.environ.get("CAUSALABA_TMEM"):
        return
    try:
        if not tracemalloc.is_tracing():
            return
        current, peak = tracemalloc.get_traced_memory()
        logger.info("[tm] %s current=%d peak=%d", msg, current, peak)
    except Exception:
        return


def _should_debug(verbose: bool) -> bool:
    return verbose and logger.isEnabledFor(logging.DEBUG)


def _fact_sort_key(fact: tuple[Any, ...]) -> tuple[float, str]:
    try:
        strength = float(fact[5])
    except Exception:
        strength = float("-inf")
    if math.isnan(strength):
        strength = float("-inf")
    return (-strength, str(fact[4]).strip())

try:
    from .causalaba import (
        compile_and_ground,
        extract_test_elements_from_symbol,
        PriorKnowledge,
        CausalABA as _baseline_causalaba,
    )
    from .causalaba_binsearch import (
        CausalABA as _binsearch_causalaba,
    )
except ImportError:  # pragma: no cover
    from causalaba import (
        compile_and_ground,
        extract_test_elements_from_symbol,
        PriorKnowledge,
        CausalABA as _baseline_causalaba,
    )
    from causalaba_binsearch import (
        CausalABA as _binsearch_causalaba,
    )

def _iter_paths_with_cutoff(graph, src, dst, cutoff):
    try:
        return (
            rx.all_simple_paths(graph, src, dst, cutoff=cutoff)
            if cutoff is not None else rx.all_simple_paths(graph, src, dst)
        )
    except TypeError:
        paths = rx.all_simple_paths(graph, src, dst)
        if cutoff is None:
            return paths

        def _gen():
            for p in paths:
                if len(p) - 1 <= cutoff:
                    yield p

        return _gen()


def _cond_to_symbol(S) -> Symbol:
    """
    Convert a conditioning set into the clingo symbol used in facts.
    """
    # clingo.Symbol is the actual runtime type; clingo.Function is a factory.
    if isinstance(S, Symbol):
        return S
    if isinstance(S, str):
        return Function(S)
    try:
        S = tuple(sorted(S))
    except Exception:
        # Best-effort fallback; treat unknown as empty.
        S = ()
    if not S:
        return Function("empty")
    return Function("s" + "y".join(str(i) for i in S))


def _cond_to_term(S) -> str:
    """Convert a conditioning set into the ASP term string used in facts."""
    if isinstance(S, Symbol):
        return str(S)
    if isinstance(S, str):
        return S
    try:
        S = tuple(sorted(S))
    except Exception:
        S = ()
    if not S:
        return "empty"
    return "s" + "y".join(str(i) for i in S)


def _pair_key(x: int, y: int) -> tuple[int, int]:
    """Return a consistent ordering for an unordered node pair."""
    return (x, y) if x <= y else (y, x)


def _block_sym(x: int, y: int) -> Symbol:
    """Symbol for the edge-blocking external."""
    a, b = _pair_key(x, y)
    return Function("block_edge", [Number(a), Number(b)])


_INCR_COUNTER = 0
# For each ordered pair, keep track of which concrete node paths have already been grounded.
_PAIR_PATHS_ADDED: dict[tuple[int, int], set[tuple[int, ...]]] = {}


def _reset_incremental_state() -> None:
    """Reset module-level incremental caches between solver runs."""
    global _INCR_COUNTER, _PAIR_PATHS_ADDED
    _INCR_COUNTER = 0
    _PAIR_PATHS_ADDED = {}


class _DebugObserver:
    """
    Program observer to capture grounded rules for debugging.
    """

    def __init__(self):
        self.ctl = None
        self.rules: list[str] = []
        self.minimize_rules: list[str] = []
        self.externals: list[str] = []
        self._lit_map: dict[int, str] = {}

    def set_ctl(self, ctl):
        self.ctl = ctl
        self._lit_map = {}

    def _ensure_map(self):
        if self.ctl is None:
            return
        # IMPORTANT: `output_atom(...)` only provides mappings for *shown* atoms.
        # We still need a full literal->symbol map for all symbolic atoms so the
        # dumped program is valid ASP (i.e., no bare numeric atoms like `2.`).
        try:
            for atom in self.ctl.symbolic_atoms:
                try:
                    lit = int(atom.literal)
                except Exception:
                    continue
                # Preserve any pre-existing mapping set by output_atom().
                self._lit_map.setdefault(lit, str(atom.symbol))
        except Exception:
            pass

    def output_atom(self, symbol, atom):
        """Capture literal-to-symbol mapping from #show atoms."""
        try:
            self._lit_map[atom] = str(symbol)
        except Exception:
            pass

    def _lit(self, lit: int) -> str:
        self._ensure_map()
        if lit == 0 or self.ctl is None:
            return str(lit)
        sign = "" if lit > 0 else "not "
        sym = self._lit_map.get(abs(lit))
        # If a literal is not part of `symbolic_atoms`, clingo has no symbolic name for it.
        # Emit a fresh, safe predicate so the dump remains valid ASP.
        atom = sym if sym is not None else f"__lit__({abs(lit)})"
        return f"{sign}{atom}"

    def rule(self, choice, head, body) -> None:
        h_str = ("; " if choice else " | ").join(self._lit(h) for h in head)
        b_str = ", ".join(self._lit(b) for b in body)
        if h_str and b_str:
            self.rules.append(f"{h_str} :- {b_str}.")
        elif h_str:
            self.rules.append(f"{h_str}.")
        elif b_str:
            self.rules.append(f":- {b_str}.")

    def weight_rule(self, choice, head, lower_bound, body) -> None:
        hb = "; ".join(self._lit(h) for h in head) if choice else " | ".join(self._lit(h) for h in head)
        body_terms = "; ".join(f"{self._lit(l)}={w}" for l, w in body)
        self.rules.append(f"{hb} :- {lower_bound}{{{body_terms}}}.")

    def minimize(self, priority, literals) -> None:
        self.minimize_rules.append(
            f":~ {', '.join(f'{self._lit(l)}={w}' for l, w in literals)}. [{priority}]"
        )

    def external(self, atom, value) -> None:
        self.externals.append(f"#external {self._lit(atom)}.")

    def dump_to(self, path: str, ext_values: dict[Symbol, bool | None], note: str = ""):
        try:
            # Build literal map before dumping so we can print symbolic names.
            self._ensure_map()
            try:
                from pathlib import Path

                p = Path(path)
                if p.parent and str(p.parent) not in (".", ""):
                    p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            with open(path, "w") as f:
                if note:
                    f.write(f"% {note}\n")
                if self._lit_map:
                    f.write("% literal mapping\n")
                    for lit, sym in sorted(self._lit_map.items()):
                        f.write(f"% {lit} -> {sym}\n")
                if self.ctl is not None:
                    try:
                        f.write("% symbolic atoms\n")
                        for atom in self.ctl.symbolic_atoms:
                            f.write(f"% {atom.literal}: {atom.symbol}\n")
                    except Exception:
                        pass
                for r in self.rules:
                    f.write(r + "\n")
                for m in self.minimize_rules:
                    f.write(m + "\n")
                if ext_values:
                    f.write("% external assignments\n")
                    for sym, val in ext_values.items():
                        f.write(f"% {sym} = {val}\n")
        except Exception:
            logger.exception("Failed to write debug dump to %s", path)


def _set_opt_mode_for_first(ctl, _opt_mode: str):
    """Configure clingo for a fast SAT witness (ignore optimization).

    This matches the baseline behavior: use opt_mode=ignore + models=1 to
    determine satisfiability/removal without paying optimization costs.
    """
    try:
        cfg = ctl.configuration.solve
    except Exception:
        return None

    prev_opt_mode = None
    prev_models = None
    try:
        prev_opt_mode = getattr(cfg, "opt_mode", None)
    except Exception:
        prev_opt_mode = None
    try:
        prev_models = getattr(cfg, "models", None)
    except Exception:
        prev_models = None

    try:
        cfg.opt_mode = "ignore"
    except Exception:
        pass
    try:
        cfg.models = 1
    except Exception:
        pass
    return (prev_opt_mode, prev_models)


def _restore_opt_mode(ctl, prev_opt_state):
    if prev_opt_state is None:
        return
    prev_opt_mode, prev_models = prev_opt_state
    try:
        cast(Any, ctl.configuration).solve.opt_mode = prev_opt_mode
    except Exception:
        pass
    try:
        if prev_models is not None:
            cast(Any, ctl.configuration).solve.models = prev_models
    except Exception:
        pass


def _add_pair_rules_incremental(
    ctl,
    n_nodes: int,
    X: int,
    Y: int,
    *,
    indep_facts: dict[tuple, set[tuple]],
    dep_facts: dict[tuple, set[tuple]],
    ext_flag: bool,
    prior_knowledge: PriorKnowledge | None,
    max_path_length: int | None,
    max_conditioning_size: int | None,
    collider_tree_depth: int | None,
    pre_grounding: bool,
    ext_values: dict[Symbol, bool | None] | None = None,
    debug_traces: bool = False,
):
    # Option A uses a Bayes-ball ASP encoding to derive ap3/3 for all required
    # (X,Y,S) triples in one shot. Incremental path grounding is no longer used.
    return None


def _count_atoms(ctl, name: str, arity: int | None = None) -> int:
    cnt = 0
    for it in ctl.symbolic_atoms:
        try:
            sym = it.symbol
        except Exception:
            continue
        if sym.name != name:
            continue
        if arity is not None and len(sym.arguments) != arity:
            continue
        cnt += 1
    return cnt


def _count_ap_for_pair(ctl, X: int, Y: int) -> int:
    cnt = 0
    for it in ctl.symbolic_atoms:
        try:
            sym = it.symbol
        except Exception:
            continue
        if sym.name != 'ap' or len(sym.arguments) < 2:
            continue
        try:
            if int(str(sym.arguments[0])) == X and int(str(sym.arguments[1])) == Y:
                cnt += 1
        except Exception:
            continue
    return cnt


def _ap_sets_for_pair(ctl, X: int, Y: int) -> set[str]:
    """Collect the conditioning set terms that appear in ap/4 for the given pair."""
    sets: set[str] = set()
    for it in ctl.symbolic_atoms.by_signature('ap', 4):
        sym = it.symbol
        try:
            if int(str(sym.arguments[0])) != X or int(str(sym.arguments[1])) != Y:
                continue
            sets.add(str(sym.arguments[3]))
        except Exception:
            continue
    return sets


def CausalABA(
    n_nodes:int, facts_location:str="", print_models:bool=True,
    skeleton_rules_reduction:bool=False,
    weak_constraints:bool=False,
    fact_pct:float=1.0,
    set_indep_facts:bool=False,
    opt_mode:str='optN',
    out_n:int=0,
    search_for_models:str='No', 
    show:list=['arrow'],
    pre_grounding: bool=False,
    disable_reground: bool=False,
    prior_knowledge: PriorKnowledge | None = None,
    return_statistics: bool = False,
    max_path_length: int | None = None,
    max_conditioning_size: int | None = None,
    collider_tree_depth: int | None = None,
    cycle_length: int | None = None,
    threads: int | None = None,
    debug_dump_path: str | None = None,
    debug_dump_always: bool = False,
    debug_dump_include_facts: bool = True,
    debug_dump_materialize_block_edges: bool = False,
    debug_traces: bool = False,
    solve_timeout: float | None = None,
    final_solve_timeout: float | None = None,
    final_solve_opt_mode: str | None = None,
    final_solve_n_models: int | None = None,
    satcheck_timeout: float | None = None,
    satcheck_threads: int | None = None,
    satcheck_probe_limit: int = 8,
    satcheck_frontier_crawl: bool = True,
    satcheck_promoted_retry: bool = True,
    satcheck_promoted_retry_timeout_scale: float = 2.0,
    satcheck_promoted_retry_max_retries: int = 1,
    satcheck_retry_frontier_sat: bool = False,
    satcheck_portfolio: bool = True,
    satcheck_portfolio_size: int = 3,
    satcheck_portfolio_timeout_scale: float = 0.5,
    satcheck_portfolio_min_timeout: float = 180.0,
    satcheck_portfolio_throttle: bool = True,
    satcheck_portfolio_min_size: int = 2,
    satcheck_plateau_stop: bool = True,
    satcheck_plateau_stop_width_ratio: float = 0.07,
    satcheck_plateau_stop_min_calls: int = 40,
    satcheck_plateau_stop_unknown_ratio: float = 0.80,
    adaptive_satcheck_threads: bool = False,
    satcheck_min_threads: int = 1,
    satcheck_increase_step: int = 2,
    verbosity: int = 0,
    )->list:
    # For pre-grounding workloads, fall back to the baseline grounding
    # with binary-search removal to improve solve-time over linear removal.
    if pre_grounding:
        return _binsearch_causalaba(
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
            verbosity=verbosity,
        )
    _verb = int(verbosity or 0)
    logger.info("Running CausalABA")
    # Reset per-run incremental bookkeeping to avoid cross-run contamination
    _reset_incremental_state()

    t_compile0 = time.perf_counter()
    profile: dict = {
        "compile_sec_total": 0.0,
        "compile_sec_last": 0.0,
        "ground_sec_total": 0.0,
        "ground_sec_last": 0.0,
        # Fine-grained compile/ground breakdown
        "write_gen_lp_sec": 0.0,
        "load_encoding_sec": 0.0,
        "load_facts_sec": 0.0,
        "load_wc_sec": 0.0,
        "add_bayesball_sec": 0.0,
        "solve_sec_total": 0.0,
        "solve_sec_last": 0.0,
        "post_solve_sec_total": 0.0,
        "compile_calls": 0,
        "ground_calls_internal": 0,
        "solve_calls_internal": 0,
        "sat_check_sec_total": 0.0,
        "sat_check_sec_last": 0.0,
        "final_opt_sec": 0.0,
        "peak_after_compile_bytes": None,
        "peak_after_ground_bytes": None,
        "peak_after_solve_bytes_last": None,
        "peak_after_solve_bytes_max": None,
        "timed_out": False,
    }

    last_solve_end: float | None = None

    def _record_solve(duration_sec: float, *, kind: str | None = None, timed_out: bool = False) -> None:
        try:
            d = float(duration_sec)
        except Exception:
            d = 0.0
        if d < 0:
            d = 0.0
        profile["solve_calls_internal"] += 1
        profile["solve_sec_last"] = d
        profile["solve_sec_total"] += d
        if kind == "satcheck":
            profile["sat_check_sec_last"] = d
            profile["sat_check_sec_total"] += d
        elif kind == "final":
            profile["final_opt_sec"] = d

        if timed_out:
            profile["timed_out"] = True

        try:
            import tracemalloc as _tracemalloc

            if _tracemalloc.is_tracing():
                _cur, _peak = _tracemalloc.get_traced_memory()
                profile["peak_after_solve_bytes_last"] = int(_peak)
                prev = profile.get("peak_after_solve_bytes_max")
                profile["peak_after_solve_bytes_max"] = int(_peak) if prev is None else max(int(prev), int(_peak))
        except Exception:
            pass

    def _ret(models_out, multiple: bool, remove_n: int = 0):
        # Record time after the last clingo solve call until returning.
        try:
            if last_solve_end is not None:
                profile["post_solve_sec_total"] = max(0.0, time.perf_counter() - float(last_solve_end))
            else:
                profile["post_solve_sec_total"] = 0.0
        except Exception:
            pass
        if return_statistics:
            try:
                stats = ctl.statistics  # type: ignore[name-defined]
            except Exception:
                stats = {}
            return [models_out, multiple, stats, int(remove_n or 0), profile]
        return [models_out, multiple]

    indep_facts: dict[tuple, set[tuple]] = {}
    dep_facts: dict[tuple, set[tuple]] = {}
    facts = []
    ext_flag = False
    block_pairs: set[tuple[int, int]] = set()
    ap_guard_syms: list[Symbol] = []
    debug_dump_path = debug_dump_path or os.environ.get("CAUSALABA_DUMP")
    ext_values: dict[Symbol, bool | None] = {}
    # IMPORTANT: registering a clingo observer that records *all grounded rules*
    # is extremely expensive and can dominate runtime (especially grounding).
    # For normal runs that only want an emitted LP for reuse, we dump the *source*
    # program instead and do NOT enable the observer.
    want_source_dump = bool(debug_dump_path) and not bool(debug_traces)
    want_grounded_dump = bool(debug_dump_path) and bool(debug_traces)
    enable_observer = bool(debug_traces)
    observer = _DebugObserver() if enable_observer else None
    debug_enabled = _should_debug(debug_traces)

    def _maybe_dump_grounded(*, note: str) -> None:
        if observer is None or not debug_dump_path:
            return
        if not want_grounded_dump:
            return
        # Allow callers to request a reusable grounded dump regardless of SAT/UNSAT.
        if not debug_dump_always:
            return
        if _verb >= 1 or debug_enabled:
            logger.info("[debug] dumping GROUNDED program to %s", debug_dump_path)
        try:
            observer.dump_to(debug_dump_path, ext_values, note=note)
        except Exception:
            logger.exception("Failed to dump grounded program")

    def _write_source_dump(*, encoding_path: str, facts_path: str | None, include_facts: bool, gen_facts_text: str | None, extra_specific: list[str], extra_base: list[str], materialize_block_edges: bool) -> None:
        if not debug_dump_path or not want_source_dump:
            return
        if not debug_dump_always:
            return
        try:
            from pathlib import Path

            p = Path(debug_dump_path)
            if p.parent and str(p.parent) not in (".", ""):
                p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            with open(encoding_path, "r") as f:
                enc_text = f.read()
        except Exception as e:
            logger.warning("Failed to read encoding for source dump: %s", e)
            enc_text = "% (failed to read encoding)\n"

        facts_text = ""
        wc_text = ""
        if include_facts and facts_path:
            try:
                with open(facts_path, "r") as f:
                    facts_text = f.read()
            except Exception as e:
                logger.warning("Failed to read facts for source dump: %s", e)
            # If weak_constraints are enabled, the caller convention is facts_path.replace('.lp','_wc.lp')
            try:
                wc_path = facts_path.replace(".lp", "_wc.lp")
                if os.path.exists(wc_path):
                    with open(wc_path, "r") as f:
                        wc_text = f.read()
            except Exception:
                wc_text = ""

        try:
            materialized_block_edge_facts: list[str] = []
            if materialize_block_edges:
                for sym, val in (ext_values or {}).items():
                    if val is not True:
                        continue
                    try:
                        if getattr(sym, 'name', None) != 'block_edge':
                            continue
                        args = getattr(sym, 'arguments', None) or []
                        if len(args) != 2:
                            continue
                        a = int(getattr(args[0], 'number', str(args[0])))
                        b = int(getattr(args[1], 'number', str(args[1])))
                        materialized_block_edge_facts.append(f"block_edge({a},{b}).")
                    except Exception:
                        continue

                # Source dumps are written before the first solve, so `ext_values` may not yet
                # contain any block_edge assignments. For the initial UNSAT check in ABAPC_INC,
                # all ext_indep facts are active; block_edge(a,b) is True iff the pair (a,b)
                # appears in any independence fact. This is exactly what `block_pairs` tracks.
                if not materialized_block_edge_facts and block_pairs:
                    for (a, b) in sorted(block_pairs):
                        materialized_block_edge_facts.append(f"block_edge({a},{b}).")

            # When materializing block edges for offline reuse, also flatten program parts.
            # clingo's CLI grounds `base` by default; custom parts like `specific` are only
            # grounded in the Python API via explicit `ctl.ground([...])` calls.
            # Flattening makes the dump one-shot executable.
            flat_one_shot = bool(materialize_block_edges)

            if flat_one_shot and enc_text:
                enc_text = "\n".join(
                    [ln for ln in enc_text.splitlines() if not (ln or "").strip().startswith("#program ")]
                ) + "\n"
            if flat_one_shot and gen_facts_text:
                gen_facts_text = "\n".join(
                    [ln for ln in gen_facts_text.splitlines() if not (ln or "").strip().startswith("#program ")]
                ) + "\n"

            with open(debug_dump_path, "w") as out:
                out.write(f"% source dump (pre-ground) from causalaba_increm.CausalABA\n")
                out.write(f"% note: this is NOT a grounded observer dump\n")
                if not include_facts:
                    out.write("% note: facts are intentionally omitted (base-only dump).\n")
                    out.write("%       Load facts separately (e.g. add your facts .lp on the clingo command line).\n")
                if flat_one_shot:
                    out.write("% note: flattened one-shot dump (no #program parts).\n")
                out.write(enc_text)
                out.write("\n")
                if facts_text:
                    out.write("% facts (as loaded)\n")
                    out.write(facts_text)
                    out.write("\n")
                if wc_text:
                    out.write("% weak constraints (as loaded)\n")
                    out.write(wc_text)
                    out.write("\n")
                if gen_facts_text:
                    out.write("% generated facts (in/2, qpair/3, etc)\n")
                    out.write(gen_facts_text)
                    out.write("\n")
                if extra_specific:
                    if not flat_one_shot:
                        out.write("#program specific.\n")
                    if materialize_block_edges:
                        filtered: list[str] = []
                        for ln in extra_specific:
                            s = (ln or "").strip()
                            if s.startswith("#external block_edge("):
                                m = re.match(r"^#external\s+block_edge\((\d+)\s*,\s*(\d+)\)\.?$", s)
                                if m:
                                    materialized_block_edge_facts.append(
                                        f"block_edge({int(m.group(1))},{int(m.group(2))})."
                                    )
                                continue
                            filtered.append(ln)
                        out.write("\n".join(filtered))
                    else:
                        out.write("\n".join(extra_specific))
                    if materialized_block_edge_facts:
                        out.write("\n% materialized external assignments (block_edge/2)\n")
                        out.write("\n".join(sorted(set(materialized_block_edge_facts))))
                    out.write("\n\n")
                if extra_base:
                    if not flat_one_shot:
                        out.write("#program base.\n")
                    out.write("\n".join(extra_base))
                    out.write("\n")
            if _verb >= 1 or debug_enabled:
                logger.info("[debug] wrote SOURCE program dump to %s", debug_dump_path)
        except Exception:
            logger.exception("Failed to write source dump to %s", debug_dump_path)
    
    _t_facts_read_start = time.perf_counter()
    if facts_location:
        facts_loc = facts_location.replace(".lp","_I.lp") if weak_constraints else facts_location
        with open(facts_loc, 'r') as file:
            for line in file:
                if "dep" not in line or line.startswith("%"):
                    continue
                line_clean = line.replace("#external ","").replace("\n","")
                if "ext_" in line_clean:
                    ext_flag = True
                if weak_constraints:
                    statement, Is = line_clean.split(" I=")
                    I,truth = Is.split(",")
                    X, S, Y, dep_type = extract_test_elements_from_symbol(statement)
                    facts.append((X,S,Y, dep_type, statement, float(I), truth))
                else:
                    X, S, Y, dep_type = extract_test_elements_from_symbol(line_clean)
                    facts.append((X,S,Y, dep_type, line_clean, float('nan'), "unknown"))
                # Debug: print raw fact and parsed tuple
                if _verb >= 2 or (debug_enabled and _verb >= 1):
                    try:
                        logger.debug(
                            "[facts] read raw: '%s' -> dep_type=%s, X=%s, Y=%s, S=%s",
                            line.strip(),
                            dep_type,
                            X,
                            Y,
                            sorted(list(S)),
                        )
                    except Exception:
                        pass
                condition_set = tuple(sorted(S))
                facts_group = indep_facts if "indep" in line_clean else dep_facts
                if (X,Y) not in facts_group:
                    facts_group[(X,Y)] = set()
                facts_group[(X,Y)].add(condition_set)
                if "indep" in line_clean:
                    block_pairs.add(_pair_key(X, Y))
    
    try:
        _t_facts_read_end = time.perf_counter()
        logger.info("[facts] read %d facts in %.3fs", len(facts), _t_facts_read_end - _t_facts_read_start)
    except Exception:
        pass

    _tm_stage("after reading facts")

    # Dynamic bound for conditioning size: if not provided, restrict to the
    # largest conditioning set actually observed in the input facts. This
    # avoids emitting unnecessary in(N,S) membership facts and keeps grounding
    # compact on pre-grounding style workloads.
    _t_bounds0 = time.perf_counter()
    try:
        _max_sz_seen = 0
        for _v in list(indep_facts.values()) + list(dep_facts.values()):
            for _S in _v:
                if len(_S) > _max_sz_seen:
                    _max_sz_seen = len(_S)
        if max_conditioning_size is None and _max_sz_seen > 0:
            max_conditioning_size = _max_sz_seen
        logger.info("[bounds] inferred max_conditioning_size=%s (computed in %.3fs)", max_conditioning_size, time.perf_counter() - _t_bounds0)
    except Exception:
        pass

    # Match baseline ordering: remove lowest-I facts first.
    # Note: strength is often unknown (nan) for many callers; Python's sort is
    # stable so order is preserved when strengths are equal.
    facts = sorted(facts, key=_fact_sort_key)
    block_pairs |= {_pair_key(x, y) for (x, y) in indep_facts}
    if _verb >= 2 or (debug_enabled and _verb >= 1):
        logger.debug("Ordered facts for processing:\n%s", "\n".join([f[4] for f in facts]))

    # Precompute clingo symbols for all external facts once.
    # The remove-until-SAT loop can call solve hundreds of times; recreating
    # Symbol objects (and conditioning-set symbols) per iteration is expensive
    # and shows up in tracemalloc peaks.
    fact_syms = [
        Function(f[3], [Number(f[0]), Number(f[2]), _cond_to_symbol(f[1])])
        for f in facts
    ]
    _tm_stage("after building fact_syms")
    # Important: build base without skeleton_rules_reduction so we don't add
    # permanent edge-forbidding constraints that cannot be retracted.
    # Build a pruned base without permanent edge-forbidding constraints.
    # We copy the relevant parts of compile_and_ground here to keep pruning
    # benefits without adding ":- edge(X,Y)." constraints that cannot be
    # retracted later during incremental removal.
    from clingo.control import Control
    from pathlib import Path
    import numpy as np
    import rustworkx as rx
    from itertools import combinations

    cpu_count = min(os.cpu_count() or 1, 64)
    threads = threads or cpu_count
    satcheck_timeout = solve_timeout if satcheck_timeout is None else satcheck_timeout
    search_fingerprint = _build_search_fingerprint(
        n_nodes=n_nodes,
        fact_keys=[str(f[4]).strip() for f in facts],
        max_path_length=max_path_length,
        max_conditioning_size=max_conditioning_size,
        collider_tree_depth=collider_tree_depth,
        cycle_length=cycle_length,
        prior_forbidden=sorted(list(getattr(prior_knowledge, "forbidden", []) or [])),
        prior_required=sorted(list(getattr(prior_knowledge, "required", []) or [])),
    )
    satcheck_tuner = _SatcheckThreadTuner(
        facts_location=facts_location,
        max_threads=int(threads),
        initial_threads=satcheck_threads,
        adaptive=bool(adaptive_satcheck_threads),
        min_threads=satcheck_min_threads,
        increase_step=satcheck_increase_step,
        logger=logger,
    )
    satcheck_checkpoint = _SatcheckSearchCheckpoint(
        facts_location=facts_location,
        fingerprint=search_fingerprint,
        satcheck_timeout=satcheck_timeout,
        logger=logger,
    )
    effective_final_solve_timeout = solve_timeout if final_solve_timeout is None else (
        None if float(final_solve_timeout) <= 0.0 else float(final_solve_timeout)
    )
    effective_final_solve_opt_mode = str(final_solve_opt_mode) if final_solve_opt_mode is not None else str(opt_mode)
    effective_final_solve_n_models = max(0, int(out_n) if final_solve_n_models is None else int(final_solve_n_models))
    satcheck_probe_limit = max(0, int(satcheck_probe_limit or 0))
    profile["satcheck_timeout"] = satcheck_timeout
    profile["satcheck_threads"] = satcheck_tuner.current_threads
    profile["satcheck_probe_limit"] = satcheck_probe_limit
    profile["final_solve_timeout"] = effective_final_solve_timeout
    profile["final_solve_opt_mode"] = effective_final_solve_opt_mode
    profile["final_solve_n_models"] = effective_final_solve_n_models
    profile["satcheck_frontier_crawl"] = bool(satcheck_frontier_crawl)
    profile["satcheck_promoted_retry"] = bool(satcheck_promoted_retry)
    profile["satcheck_promoted_retry_timeout_scale"] = float(satcheck_promoted_retry_timeout_scale)
    profile["satcheck_promoted_retry_max_retries"] = int(satcheck_promoted_retry_max_retries)
    profile["satcheck_retry_frontier_sat"] = bool(satcheck_retry_frontier_sat)
    profile["satcheck_portfolio"] = bool(satcheck_portfolio)
    profile["satcheck_portfolio_size"] = int(satcheck_portfolio_size)
    profile["satcheck_portfolio_timeout_scale"] = float(satcheck_portfolio_timeout_scale)
    profile["satcheck_portfolio_min_timeout"] = float(satcheck_portfolio_min_timeout)
    profile["satcheck_portfolio_throttle"] = bool(satcheck_portfolio_throttle)
    profile["satcheck_portfolio_min_size"] = int(satcheck_portfolio_min_size)
    profile["satcheck_plateau_stop"] = bool(satcheck_plateau_stop)
    profile["satcheck_plateau_stop_width_ratio"] = float(satcheck_plateau_stop_width_ratio)
    profile["satcheck_plateau_stop_min_calls"] = int(satcheck_plateau_stop_min_calls)
    profile["satcheck_plateau_stop_unknown_ratio"] = float(satcheck_plateau_stop_unknown_ratio)
    profile["adaptive_satcheck_threads"] = satcheck_tuner.adaptive
    profile["satcheck_min_threads"] = satcheck_tuner.min_threads
    profile["satcheck_increase_step"] = satcheck_tuner.increase_step
    profile["satcheck_memtotal_kib"] = satcheck_tuner.total_mem_kib
    profile["satcheck_autotune_path"] = satcheck_tuner.state_path_str
    profile["satcheck_search_checkpoint_path"] = satcheck_checkpoint.summary()["path"]
    control_args = [f"-t {threads}", "--warn=none"]
    if cycle_length is not None:
        control_args += [f"-c l_cyc={int(cycle_length)}"]
    if collider_tree_depth is not None:
        control_args += [f"-c l_b={int(collider_tree_depth)}"]
    ctl = Control(control_args)
    if observer is not None:
        try:
            ctl.register_observer(cast(Any, observer))
            observer.set_ctl(ctl)
        except Exception:
            logger.exception("Failed to register debug observer")

    # clingo's configuration object is dynamically-typed; cast to Any to satisfy type-checkers.
    cfg = cast(Any, ctl.configuration)
    cfg.solve.parallel_mode = threads
    cfg.solve.models = out_n
    cfg.solver.seed = "2024"
    cfg.solve.opt_mode = opt_mode

    # Add set membership facts (bounded by max_conditioning_size)
    try:
        from .utils.graph_utils import powerset as _powerset
    except ImportError:  # pragma: no cover
        from utils.graph_utils import powerset as _powerset
    # Emitting in/2 and qpair/3 via thousands of ctl.add() calls creates a lot
    # of Python-side allocation pressure (shows up in tracemalloc). Stream these
    # generated facts into a temp .lp and load it instead.
    import tempfile
    gen_facts_path: Path | None = None
    _t_gen0 = time.perf_counter()
    _gen_in_count = 0
    _gen_qpair_count = 0
    with tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False) as gen_f:
        gen_facts_path = Path(gen_f.name)
        gen_f.write("#program specific.\n")

        if skeleton_rules_reduction:
            base_condition_sets: set[tuple[int, ...]] = set()
            for v in indep_facts.values():
                base_condition_sets |= v
            for v in dep_facts.values():
                base_condition_sets |= v
            base_condition_iter = base_condition_sets
        else:
            base_condition_iter = _powerset(range(n_nodes))

        for S in base_condition_iter:
            if max_conditioning_size is not None and len(S) > max_conditioning_size:
                continue
            set_term = _cond_to_term(S)
            for s in S:
                gen_f.write(f"in({s},{set_term}).\n")
                _gen_in_count += 1

        # Seed Bayes-ball query triples explicitly to keep grounding compact.
        # We include all conditioning triples that appear in the input facts, even
        # if some are initially inactive (fact_pct) or later removed (externals).
        # Correctness is preserved because the hard constraints are driven by
        # dep/3 and indep/3; qpair/3 only decides which ap3/3 atoms get computed.
        for (X, Y), cond_sets in dep_facts.items():
            for S in cond_sets:
                gen_f.write(f"qpair({X},{Y},{_cond_to_term(S)}).\n")
                _gen_qpair_count += 1
        for (X, Y), cond_sets in indep_facts.items():
            for S in cond_sets:
                gen_f.write(f"qpair({X},{Y},{_cond_to_term(S)}).\n")
                _gen_qpair_count += 1

    _tm_stage("after writing generated facts lp")
    try:
        _t_gen_end = time.perf_counter()
        profile["write_gen_lp_sec"] = max(0.0, _t_gen_end - _t_gen0)
        logger.info("[gen-facts] wrote %d in/2 + %d qpair/3 atoms in %.3fs", _gen_in_count, _gen_qpair_count, profile["write_gen_lp_sec"])
    except Exception:
        pass

    # Load encoding and (optional) facts
    bounded_encoding_active = (cycle_length is not None and cycle_length > 0) or (
        collider_tree_depth is not None and collider_tree_depth > 0
    )
    encoding_file = 'causalaba_bounded.lp' if bounded_encoding_active else 'causalaba.lp'
    encoding_path = str(Path(__file__).resolve().parent / 'encodings' / encoding_file)
    extra_specific_src: list[str] = []
    extra_base_src: list[str] = []
    _t_load_enc0 = time.perf_counter()
    ctl.load(encoding_path)
    try:
        profile["load_encoding_sec"] = max(0.0, time.perf_counter() - _t_load_enc0)
        logger.info("[load] encoding (%s) in %.3fs", encoding_file, profile["load_encoding_sec"])
    except Exception:
        pass
    if facts_location:
        _t_load_f0 = time.perf_counter()
        ctl.load(facts_location)
        try:
            profile["load_facts_sec"] = max(0.0, time.perf_counter() - _t_load_f0)
            logger.info("[load] facts in %.3fs", profile["load_facts_sec"])
        except Exception:
            pass
        if weak_constraints:
            _t_load_wc0 = time.perf_counter()
            ctl.load(facts_location.replace('.lp','_wc.lp'))
            try:
                profile["load_wc_sec"] = max(0.0, time.perf_counter() - _t_load_wc0)
                logger.info("[load] weak constraints in %.3fs", profile["load_wc_sec"])
            except Exception:
                pass

    if ext_flag:
        line1 = "indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y."
        line2 = "dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y."
        ctl.add("specific", [], line1)
        ctl.add("specific", [], line2)
        extra_specific_src.extend([line1, line2])

    gen_facts_text: str | None = None
    if gen_facts_path is not None:
        try:
            try:
                if want_source_dump:
                    with open(gen_facts_path, "r") as gf:
                        gen_facts_text = gf.read()
            except Exception:
                gen_facts_text = None
            ctl.load(str(gen_facts_path))
        finally:
            try:
                gen_facts_path.unlink(missing_ok=True)
            except Exception:
                pass

    _tm_stage("after ctl.load inputs")

    # Enforce observational facts against active-path existence.
    # This works for both:
    # - plain facts loaded as dep/3, indep/3, and
    # - externalized facts mapped into dep/3, indep/3.
    #
    # Use ap3/3 (existential over ap/4) for reactivity and to avoid wildcard issues.
    ctl.add(
        "specific",
        [],
        ":- dep(X,Y,S), not ap3(X,Y,S), set(S), var(X), var(Y), X<Y, not in(X,S), not in(Y,S).",
    )
    ctl.add(
        "specific",
        [],
        ":- indep(X,Y,S), ap3(X,Y,S), set(S), var(X), var(Y), X<Y, not in(X,S), not in(Y,S).",
    )
    extra_specific_src.append(":- dep(X,Y,S), not ap3(X,Y,S), set(S), var(X), var(Y), X<Y, not in(X,S), not in(Y,S).")
    extra_specific_src.append(":- indep(X,Y,S), ap3(X,Y,S), set(S), var(X), var(Y), X<Y, not in(X,S), not in(Y,S).")

    # Provide explicit var/1 facts up-front (redundant with the encoding) so that
    # guard rules that quantify over all variable pairs can be grounded safely.
    # We keep these in a dedicated program part that we can ground early.
    for i in range(n_nodes):
        # Keep for source dump even though encoding already provides var/1.
        extra_base_src.append(f"var({i}).")
        ctl.add("guards_use", [], f"var({i}).")

    # Compute d-connection (active trails) using a Bayes-ball style encoding.
    #
    # This replaces explicit enumeration of all simple paths (which is exponential
    # and dominated initial grounding time/memory). Bayes-ball explores a finite
    # state space per (start node, conditioning set), which is polynomial.
    #
    # We only compute ap3/3 for (X,Y,S) triples that occur in the observational
    # facts (dep/3 or indep/3), which keeps grounding compact under
    # skeleton_rules_reduction.
    #
    # Compatibility notes:
    # - Some legacy tests and call-sites expect `ap/4` atoms to exist. We expose
    #   `ap/4` as an alias of `ap3/3` (with a dummy third argument) so callers can
    #   continue using `show=["ap"]`.
    # - `max_path_length` is implemented as a bound on the number of Bayes-ball
    #   edge traversals (steps).
    # - `collider_tree_depth` bounds how far a collider-descendant in S can be to
    #   open colliders; when > 0 we use the bounded encoding's `dpath_k/3`.

    bb_lines: list[str] = []
    bb_lines.append("% Query triples to evaluate (restrict to one orientation to match constraints).")
    if 'ap' in show:
        # Legacy/debug mode: allow users/tests to ask for active-path atoms even
        # without providing dep/indep facts.
        bb_lines.append("want_ap.")
        bb_lines.append("qpair(X,Y,S) :- want_ap, set(S), var(X), var(Y), X<Y, X!=Y.")
    bb_lines.append("")
    bb_lines.append("bb_start(X,S) :- qpair(X,_,S), set(S), var(X).")
    bb_lines.append("")

    # Ancestor-of-observed (incl. observed itself): used to open colliders.
    bb_lines.append("% Ancestor-of-observed (incl. observed itself): used to open colliders.")
    bb_lines.append("anc_obs(N,S) :- in(N,S), set(S), var(N).")
    if collider_tree_depth is not None and collider_tree_depth > 0:
        # Depth-bounded collider activation: only observed descendants within l_b
        # directed steps can open a collider.
        #
        # This matches the legacy bounded semantics (see causalaba_bounded.lp's
        # collider_desc_b/4).
        bb_lines.append("anc_obs(N,S) :- in(Z,S), dpath_k(N,Z,K), set(S), var(N), var(Z), N!=Z, K=1..l_b.")
    else:
        bb_lines.append("anc_obs(N,S) :- in(Z,S), dpath(N,Z), set(S), var(N), var(Z), N!=Z.")
    bb_lines.append("")

    # Collider detection: a node with at least two distinct parents.
    bb_lines.append("is_collider(N) :- arrow(P,N), arrow(Q,N), P<Q, P!=Q, var(P), var(Q), var(N).")
    bb_lines.append("")

    if max_path_length is not None:
        l_ap = int(max_path_length)
        bb_lines.append(f"#const l_ap={l_ap}.")
        bb_lines.append("")
        bb_lines.append("% Bayes-ball states with step counter T (0..l_ap).")
        bb_lines.append("bb_u(X,X,S,0) :- bb_start(X,S).")
        bb_lines.append("bb_d(X,X,S,0) :- bb_start(X,S).")
        bb_lines.append("")
        bb_lines.append("% Move to parents (consume one step).")
        bb_lines.append("% - From bb_u (arrive from child): propagate to parents if current node is not observed.")
        bb_lines.append("bb_u(X,P,S,T1) :- bb_u(X,N,S,T), T < l_ap, T1 = T+1, not in(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb_lines.append("% - From bb_d (arrive from parent): propagate to parents only through opened colliders.")
        bb_lines.append("bb_u(X,P,S,T1) :- bb_d(X,N,S,T), T < l_ap, T1 = T+1, is_collider(N), anc_obs(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb_lines.append("")
        bb_lines.append("% Move to children (consume one step; only if current node is not observed).")
        bb_lines.append("bb_d(X,C,S,T1) :- bb_u(X,N,S,T), T < l_ap, T1 = T+1, not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb_lines.append("bb_d(X,C,S,T1) :- bb_d(X,N,S,T), T < l_ap, T1 = T+1, not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb_lines.append("")
        bb_lines.append("% Active connection exists between X and Y given S iff Bayes-ball can reach Y.")
        bb_lines.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_u(X,Y,S,_).")
        bb_lines.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_d(X,Y,S,_).")
    else:
        bb_lines.append("% Bayes-ball states: bb_u(start,node,set) = reached node with ball arriving from a child.")
        bb_lines.append("%                  bb_d(start,node,set) = reached node with ball arriving from a parent.")
        bb_lines.append("bb_u(X,X,S) :- bb_start(X,S).")
        bb_lines.append("bb_d(X,X,S) :- bb_start(X,S).")
        bb_lines.append("")
        bb_lines.append("% Move to parents.")
        bb_lines.append("% - If the ball arrives from a child (bb_u), only propagate further if the")
        bb_lines.append("%   current node is NOT observed.")
        bb_lines.append("% - If the ball arrives from a parent (bb_d), propagate to parents iff the")
        bb_lines.append("%   current node is an ancestor of an observed node (incl. observed itself).")
        bb_lines.append("bb_u(X,P,S) :- bb_u(X,N,S), not in(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb_lines.append("bb_u(X,P,S) :- bb_d(X,N,S), is_collider(N), anc_obs(N,S), arrow(P,N), var(P), var(N), set(S).")
        bb_lines.append("")
        bb_lines.append("% Move to children (only if current node is not observed).")
        bb_lines.append("bb_d(X,C,S) :- bb_u(X,N,S), not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb_lines.append("bb_d(X,C,S) :- bb_d(X,N,S), not in(N,S), arrow(N,C), var(N), var(C), set(S).")
        bb_lines.append("")
        bb_lines.append("% Active connection exists between X and Y given S iff Bayes-ball can reach Y.")
        bb_lines.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_u(X,Y,S).")
        bb_lines.append("ap3(X,Y,S) :- qpair(X,Y,S), not in(X,S), not in(Y,S), bb_d(X,Y,S).")

    bb_lines.append("")
    bb_lines.append("% Backwards-compatible alias for callers/tests that expect ap/4 atoms.")
    bb_lines.append("ap(X,Y,bb,S) :- ap3(X,Y,S), qpair(X,Y,S).")

    _t_bb0 = time.perf_counter()
    ctl.add("specific", [], "\n".join(bb_lines))
    extra_specific_src.extend(bb_lines)
    try:
        profile["add_bayesball_sec"] = max(0.0, time.perf_counter() - _t_bb0)
        logger.info("[bayes-ball] added %d rule lines in %.3fs", len(bb_lines), profile["add_bayesball_sec"])
    except Exception:
        pass

    _tm_stage("after ctl.add bayesball rules")

    # NOTE: We no longer rely on explicit ap/4 path atoms nor on `not ap(...)`, so
    # the earlier ap_guard_set/1 machinery is not needed.
    # Dynamic skeleton blocking is only needed/valid when using externals.
    if skeleton_rules_reduction and ext_flag and block_pairs:
        sk_line = ":- block_edge(X,Y), edge(X,Y), X<Y, var(X), var(Y)."
        ctl.add("specific", [], sk_line)
        extra_specific_src.append(sk_line)
        for (a, b) in sorted(block_pairs):
            ext_line = f"#external block_edge({a},{b})."
            ctl.add("specific", [], ext_line)
            extra_specific_src.append(ext_line)

    frozen_block_values: dict[tuple[int, int], bool] | None = None

    def _compute_block_edge_value(a: int, b: int) -> bool:
        val = False
        # indep_facts stores conditioning sets for ordered pairs as they
        # appear in the facts; check both orientations.
        for (x, y) in ((a, b), (b, a)):
            for cond in indep_facts.get((x, y), set()):
                ext_sym = Function("ext_indep", [Number(x), Number(y), _cond_to_symbol(cond)])
                if ext_values.get(ext_sym) is True:
                    val = True
                    break
            if val:
                break
        return val

    def _assign_block_edges():
        nonlocal frozen_block_values
        if not skeleton_rules_reduction or not ext_flag:
            return
        # In incremental mode, disable_reground emulates the old "no-reground"
        # approximation by freezing the initial skeleton blocks induced by the
        # currently active indep externals. Once assigned the first time, these
        # blocks are not recomputed after later fact releases.
        if disable_reground:
            if frozen_block_values is None:
                frozen_block_values = {
                    (a, b): _compute_block_edge_value(a, b)
                    for (a, b) in block_pairs
                }
                frozen_true = sum(1 for val in frozen_block_values.values() if val)
                profile["block_edge_initial_true_count"] = int(frozen_true)
                profile["block_edge_frozen_pair_count"] = int(len(frozen_block_values))
                logger.info(
                    "[block-edge] disable_reground=true freezing %d/%d pair blocks from initial active indeps",
                    frozen_true,
                    len(frozen_block_values),
                )
            for (a, b), val in frozen_block_values.items():
                sym = _block_sym(a, b)
                try:
                    ctl.assign_external(sym, val)
                    ext_values[sym] = val
                except Exception:
                    if debug_enabled:
                        logger.debug("Failed to assign frozen block_edge(%s,%s)", a, b)
            return

        # Dynamic mode: block an undirected pair (a,b) iff there exists any
        # *currently active* ext_indep(a,b,S). This allows unblocking purely
        # by releasing ext_indep externals, without regrounding.
        for (a, b) in block_pairs:
            sym = _block_sym(a, b)
            val = _compute_block_edge_value(a, b)
            try:
                ctl.assign_external(sym, val)
                ext_values[sym] = val
            except Exception:
                if debug_enabled:
                    logger.debug("Failed to assign block_edge(%s,%s)", a, b)



    def _assign_fact_externals(removed: int = 0) -> None:
        """
        Ensure fact externals match the current removal prefix.

        This mirrors the baseline behavior where all remaining facts are kept
        true and removed ones are released (undefined).
        """
        if not facts_location:
            return
        if not facts:
            return
        cutoff = max(0, len(facts) - int(removed or 0))
        for idx, sym in enumerate(fact_syms):
            # Keep prefix facts true; release removed facts (None) to match baseline behavior.
            val = True if idx < cutoff else None
            try:
                ctl.assign_external(sym, val)
                ext_values[sym] = val
                if debug_enabled:
                    logger.debug("[assign] %s = %s (removed=%s)", sym, val, removed)
            except Exception:
                # Non-external facts (e.g., dep/indep without ext_) are loaded
                # as hard constraints; assign_external is a no-op in that case.
                continue



    def _assumptions_from_exts() -> list[tuple[Symbol, bool]]:
        """
        Build solver assumptions from the current external assignments so that
        they are enforced even if assign_external was a no-op (e.g., on
        simplified programs).
        """
        assumptions: list[tuple[Symbol, bool]] = []
        for sym, val in ext_values.items():
            if val is None:
                continue
            if sym.name in ("block_edge",):
                assumptions.append((sym, bool(val)))
        return assumptions

    # No explicit path enumeration: Bayes-ball encoding above derives ap3/3.
    logger.info("   Adding Specific Rules...")

    if prior_knowledge is not None:
        # Enforce prior knowledge on arrows as in original compile_and_ground
        for (Xf, Yf) in prior_knowledge.forbidden:
            line = f":- arrow({Xf},{Yf})."
            ctl.add("specific", [], line)
            extra_specific_src.append(line)
        for (Xr, Yr) in prior_knowledge.required:
            line = f"arrow({Xr},{Yr})."
            ctl.add("specific", [], line)
            extra_specific_src.append(line)



    # Show directives
    if 'arrow' in show:
        line = "#show arrow/2."
        ctl.add("base", [], line)
        extra_base_src.append(line)
    if 'indep' in show:
        line = "#show indep/3."
        ctl.add("base", [], line)
        extra_base_src.append(line)
    if 'dep' in show:
        line = "#show dep/3."
        ctl.add("base", [], line)
        extra_base_src.append(line)
    if 'ap' in show:
        line = "#show ap/4."
        ctl.add("base", [], line)
        extra_base_src.append(line)
    if 'ext' in show:
        for line in ("#show ext_indep/3.", "#show ext_dep/3.", "#show block_edge/2.", "#show active_pair/2."):
            ctl.add("base", [], line)
            extra_base_src.append(line)

    # If requested, dump the *source* program without enabling the expensive grounded observer.
    _write_source_dump(
        encoding_path=encoding_path,
        facts_path=facts_location if facts_location else None,
        include_facts=bool(debug_dump_include_facts),
        gen_facts_text=gen_facts_text,
        extra_specific=extra_specific_src,
        extra_base=extra_base_src,
        materialize_block_edges=bool(debug_dump_materialize_block_edges),
    )

    # Ground the full base
    logger.info("   Grounding full base program...")
    # Treat everything up to this point as "compile" for profiling purposes.
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

    profile["ground_calls_internal"] += 1
    # Ground all parts together in a single call (required for correct program composition)
    _t_ground_start = time.perf_counter()
    t_ground0 = time.perf_counter()
    ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes-1)])])
    t_ground1 = time.perf_counter()
    profile["ground_sec_last"] = max(0.0, t_ground1 - t_ground0)
    profile["ground_sec_total"] += profile["ground_sec_last"]
    try:
        logger.info("[ground] total=%.3fs", profile["ground_sec_last"])
    except Exception:
        pass
    try:
        import tracemalloc as _tracemalloc

        if _tracemalloc.is_tracing():
            _cur, _peak = _tracemalloc.get_traced_memory()
            profile["peak_after_ground_bytes"] = int(_peak)
    except Exception:
        pass
    _tm_stage("after ctl.ground")
    t_post_ground = time.perf_counter()
    logger.info("[timing] post-ground start")
    if _verb >= 2 or (debug_enabled and _verb >= 1):
        try:
            ext_deps = list(ctl.symbolic_atoms.by_signature("ext_dep", 3))
            ext_indeps = list(ctl.symbolic_atoms.by_signature("ext_indep", 3))
            logger.debug(
                "[externals] ext_dep=%s ext_indep=%s",
                [(str(a.symbol), bool(getattr(a, "is_external", False))) for a in ext_deps],
                [(str(a.symbol), bool(getattr(a, "is_external", False))) for a in ext_indeps],
            )
        except Exception:
            logger.debug("[externals] failed to inspect ext_* atoms", exc_info=True)
    # Activate all pair guards by default
    models = []
    n_models = 0

    if search_for_models == 'No':
        for n, (fact, ext_sym) in enumerate(zip(facts, fact_syms)):
            if fact[3] == "ext_indep" and set_indep_facts:
                ctl.assign_external(ext_sym, True)
            elif n/len(facts) <= fact_pct:
                ctl.assign_external(ext_sym, True)
            else:
                ctl.assign_external(ext_sym, False)
            ext_values[ext_sym] = not (n/len(facts) > fact_pct and fact[3] != "ext_indep")
        _assign_block_edges()
        def _on_model_no(model):
            models.append(model.symbols(shown=True))

        t_s0 = time.perf_counter()
        finished, _solve_result = _solve_with_timeout(
            ctl,
            solve_timeout=solve_timeout,
            on_model=_on_model_no,
            assumptions=_assumptions_from_exts(),
        )
        t_s1 = time.perf_counter()
        last_solve_end = float(t_s1)
        _record_solve(t_s1 - t_s0, timed_out=(not finished))
        try:
            logger.info("[timing] mode=No solve=%.3fs since-ground=%.3fs", t_s1 - t_s0, t_s1 - t_post_ground)
        except Exception:
            pass
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        if n_models > 0:
            return _ret(models, False, 0)

    elif search_for_models == 'first':
        _t_first_setup = time.perf_counter()
        for fact, ext_sym in zip(facts, fact_syms):
            ctl.assign_external(ext_sym, True)
            ext_values[ext_sym] = True
            if _verb >= 2 or (debug_enabled and _verb >= 1):
                logger.debug("[assign] %s = %s", ext_sym, True)
        _assign_block_edges()
        try:
            logger.info("[timing] first-setup=%.3fs", time.perf_counter() - _t_first_setup)
        except Exception:
            pass
        resume_remove_search = satcheck_checkpoint.has_resume_state()
        profile["satcheck_resume"] = resume_remove_search
        if resume_remove_search:
            try:
                logger.info(
                    "[first-solve] skipping solve; resuming remove-search from checkpoint cache=%d best_unsat=%s best_sat=%s complete=%s",
                    len(satcheck_checkpoint.cache),
                    satcheck_checkpoint.best_unsat_removed,
                    satcheck_checkpoint.best_sat_removed,
                    satcheck_checkpoint.search_complete,
                )
            except Exception:
                pass
            n_models = 0
        else:
            last_syms = None
            logger.info("[first-solve] activated %d facts, solving...", len(facts))
            _t_first_solve = time.perf_counter()
            prev_opt_state = _set_opt_mode_for_first(ctl, opt_mode)
            try:
                def _on_model_first(model):
                    nonlocal last_syms
                    syms = model.symbols(shown=True)
                    last_syms = syms
                    if not models:
                        models.append(syms)

                t_s0 = time.perf_counter()
                finished, _solve_result = _solve_with_timeout(
                    ctl,
                    solve_timeout=solve_timeout,
                    on_model=_on_model_first,
                    assumptions=_assumptions_from_exts(),
                )
                t_s1 = time.perf_counter()
                last_solve_end = float(t_s1)
                _record_solve(t_s1 - t_s0, timed_out=(not finished))
                try:
                    logger.info("[timing] first-solve=%.3fs since-ground=%.3fs", t_s1 - t_s0, t_s1 - t_post_ground)
                except Exception:
                    pass
            finally:
                _restore_opt_mode(ctl, prev_opt_state)
            if not models and last_syms is not None:
                models.append(last_syms)
            n_models = int(ctl.statistics['summary']['models']['enumerated'])
            try:
                _t_first_end = time.perf_counter()
                logger.info("[first-solve] found n_models=%s in %.3fs", n_models, _t_first_end - _t_first_solve)
            except Exception:
                pass
            _tm_stage("after first solve")
            if debug_enabled:
                logger.debug("[first] models: %s", models)
            if n_models > 0 and models:
                # In increm-only mode, downstream MUS/OptMCS expects a base program file.
                # If the instance is already SAT (remove_n=0), still emit the dump when requested.
                _maybe_dump_grounded(note=f"remove_n=0 models={n_models} timed_out={bool(profile.get('timed_out', False))}")
        
    elif 'subsets' in search_for_models:
        # Explore all subsets of facts by toggling externals; collect solutions.
        try:
            from .utils.graph_utils import powerset as _powerset
        except ImportError:  # pragma: no cover
            from utils.graph_utils import powerset as _powerset
        set_of_models = []
        for f_to_remove in _powerset(facts):
            # Set all facts True first
            for ext_sym in fact_syms:
                ctl.assign_external(ext_sym, True)
            # Then disable the chosen subset (unless ext_indep with set_indep_facts flag)
            for fact in f_to_remove:
                fact_t = cast(tuple[Any, Any, Any, Any], fact)
                if fact_t[3] == "ext_indep" and set_indep_facts:
                    continue
                # subsets mode is inherently exponential; keep this simple and correct.
                ext_sym = Function(fact_t[3], [Number(fact_t[0]), Number(fact_t[2]), _cond_to_symbol(fact_t[1])])
                ctl.assign_external(ext_sym, False)
            # Avoid extra per-iteration overhead: block-edge assignment is not
            # required for correctness in subsets enumeration and adds
            # significant cost. Baseline subsets mode does not perform it.
            curr = []
            def _on_model_sub(model):
                curr.append(model.symbols(shown=True))

            t_s0 = time.perf_counter()
            finished, _solve_result = _solve_with_timeout(
                ctl,
                solve_timeout=solve_timeout,
                on_model=_on_model_sub,
                # Assumptions are unnecessary here; toggled externals suffice.
                # Dropping them aligns with baseline behavior and reduces solve overhead.
                assumptions=None,
            )
            t_s1 = time.perf_counter()
            last_solve_end = float(t_s1)
            _record_solve(t_s1 - t_s0, timed_out=(not finished))
            n_curr = int(ctl.statistics['summary']['models']['enumerated'])
            if n_curr > 0:
                if search_for_models == 'first_subsets':
                    return _ret(curr, False, 0)
                set_of_models.append(curr)

        if len(set_of_models) > 0:
            _maybe_dump_grounded(note=f"remove_n=0 models={len(set_of_models)} timed_out={bool(profile.get('timed_out', False))}")
            return _ret(set_of_models, True, 0)

    if (search_for_models != 'first'):
        _maybe_dump_grounded(note=f"remove_n=0 models={n_models} timed_out={bool(profile.get('timed_out', False))}")
        return _ret(models, False, 0)

    # --- Remove-until-SAT (monotonic search) ---
    # In this mode, we only toggle externals and re-solve; we do NOT add/ground
    # new rules.
    #
    # The constraint set is monotonic in `removed`: releasing more externals can
    # only relax the program, so satisfiable(rem) is monotone non-decreasing.
    # Baseline removes facts one-by-one until SAT; that returns the *minimal*
    # number of removed facts. We can find the same point with O(log n) solves.

    def _apply_removed(removed: int) -> None:
        _assign_fact_externals(int(removed or 0))
        _assign_block_edges()

    satcheck_stats = {"calls": 0, "sec": 0.0, "timeouts": 0, "unknown": 0}

    def _is_satisfiable(*, removed: int, timeout_override: float | None = None) -> _SolveStatus:
        """Check satisfiability *without* spending time optimizing.

        Weak constraints never change satisfiable/unsatisfiable, but clingo's
        optimization can make repeated checks (during binary search) extremely
        slow. We temporarily set opt_mode=ignore and models=1.
        """
        nonlocal last_solve_end
        satcheck_stats["calls"] += 1
        _t_satcheck0 = time.perf_counter()
        cfg_solve: Any = None
        prev_opt_mode = None
        prev_models = None
        prev_parallel_mode = None
        current_satcheck_threads = int(satcheck_tuner.current_threads)
        try:
            satcheck_tuner.mark_start(removed=removed)
            try:
                cfg_solve = ctl.configuration.solve
                prev_opt_mode = getattr(cfg_solve, "opt_mode", None)
                prev_models = getattr(cfg_solve, "models", None)
                prev_parallel_mode = getattr(cfg_solve, "parallel_mode", None)
                try:
                    cfg_solve.opt_mode = "ignore"
                except Exception:
                    pass
                try:
                    cfg_solve.models = 1
                except Exception:
                    pass
                try:
                    cfg_solve.parallel_mode = current_satcheck_threads
                except Exception:
                    pass
            except Exception:
                cfg_solve = None

            # Hard-timeout satisfiability checks too; treat timeout as "unknown"
            # and let the search probe nearby removal counts instead of
            # collapsing unknown into UNSAT.
            found = {"sat": False}

            def _on_model_sat(_model):
                found["sat"] = True

            rss0 = _mem.rss_kib()
            hwm0 = _mem.maxrss_kib()
            t0 = time.perf_counter()
            finished, solve_result = _solve_with_timeout(
                ctl,
                solve_timeout=satcheck_timeout if timeout_override is None else timeout_override,
                on_model=_on_model_sat,
                assumptions=_assumptions_from_exts(),
            )
            t1 = time.perf_counter()
            rss1 = _mem.rss_kib()
            hwm1 = _mem.maxrss_kib()
            available_mem_kib = _mem.memavailable_kib()
            peak_kib = _satcheck_peak_kib(
                rss_before_kib=rss0,
                rss_after_kib=rss1,
                hwm_before_kib=hwm0,
                hwm_after_kib=hwm1,
            )
            last_solve_end = float(t1)
            _record_solve(t1 - t0, kind="satcheck", timed_out=(not finished))
            try:
                logger.info(
                    "[mem] satcheck rss=%s->%s hwm=%s->%s",
                    _mem.fmt_kib(rss0),
                    _mem.fmt_kib(rss1),
                    _mem.fmt_kib(hwm0),
                    _mem.fmt_kib(hwm1),
                )
            except Exception:
                pass
            status = _classify_solve_result(
                found_sat=bool(found["sat"]),
                finished=bool(finished),
                result=solve_result,
            )
            if status == "unknown":
                satcheck_stats["unknown"] += 1
            if not finished:
                satcheck_stats["timeouts"] += 1
            tune_result = satcheck_tuner.finish(
                removed=removed,
                status=status,
                timed_out=(not finished),
                rss_kib=rss1,
                peak_kib=peak_kib,
                available_mem_kib=available_mem_kib,
            )
            if bool(tune_result.get("changed")):
                logger.info(
                    "[autotune] satcheck_threads %d->%d event=%s peak=%s avail=%s total=%s removed=%d status=%s",
                    int(tune_result["previous_threads"]),
                    int(tune_result["current_threads"]),
                    tune_result["event"],
                    _mem.fmt_kib(peak_kib),
                    _mem.fmt_kib(available_mem_kib),
                    _mem.fmt_kib(satcheck_tuner.total_mem_kib),
                    int(removed),
                    status,
                )
            profile["satcheck_threads"] = satcheck_tuner.current_threads
            satcheck_stats["sec"] += time.perf_counter() - _t_satcheck0
            return status
        finally:
            if cfg_solve is not None:
                try:
                    if prev_opt_mode is not None:
                        cfg_solve.opt_mode = prev_opt_mode
                except Exception:
                    pass
                try:
                    if prev_models is not None:
                        cfg_solve.models = prev_models
                except Exception:
                    pass
                try:
                    if prev_parallel_mode is not None:
                        cfg_solve.parallel_mode = prev_parallel_mode
                except Exception:
                    pass

    # Binary search the minimal removed count that yields SAT.
    if n_models > 0:
        remove_n = 0
        best_sat_removed: int | None = 0
        best_unsat_removed: int | None = None
        approximate_remove_search = False
        satcheck_records: list[dict[str, Any]] = []
    else:
        _t_binsearch_start = time.perf_counter()
        logger.info(
            "[remove-search] satcheck_timeout=%s satcheck_threads=%s probe_limit=%s frontier=%s retry=%s retry_scale=%s retry_max=%s retry_sat=%s portfolio=%s portfolio_size=%s portfolio_scale=%s portfolio_min=%s portfolio_throttle=%s portfolio_min_size=%s plateau_stop=%s plateau_width_ratio=%s plateau_min_calls=%s plateau_unknown_ratio=%s adaptive=%s min_threads=%s step=%s total_mem=%s",
            satcheck_timeout,
            satcheck_tuner.current_threads,
            satcheck_probe_limit,
            bool(satcheck_frontier_crawl),
            bool(satcheck_promoted_retry),
            satcheck_promoted_retry_timeout_scale,
            satcheck_promoted_retry_max_retries,
            bool(satcheck_retry_frontier_sat),
            bool(satcheck_portfolio),
            satcheck_portfolio_size,
            satcheck_portfolio_timeout_scale,
            satcheck_portfolio_min_timeout,
            bool(satcheck_portfolio_throttle),
            satcheck_portfolio_min_size,
            bool(satcheck_plateau_stop),
            satcheck_plateau_stop_width_ratio,
            satcheck_plateau_stop_min_calls,
            satcheck_plateau_stop_unknown_ratio,
            satcheck_tuner.adaptive,
            satcheck_tuner.min_threads,
            satcheck_tuner.increase_step,
            _mem.fmt_kib(satcheck_tuner.total_mem_kib),
        )
        def _solve_removed(removed: int, timeout_override: float | None = None) -> _SolveStatus:
            _apply_removed(removed)
            return _is_satisfiable(removed=removed, timeout_override=timeout_override)

        search_result = _run_remove_search(
            total_facts=len(facts),
            checkpoint=satcheck_checkpoint,
            satcheck_probe_limit=satcheck_probe_limit,
            logger=logger,
            solve_removed=_solve_removed,
            base_satcheck_timeout=satcheck_timeout,
            enable_frontier_crawl=bool(satcheck_frontier_crawl),
            enable_promoted_retry=bool(satcheck_promoted_retry),
            promoted_retry_timeout_scale=float(satcheck_promoted_retry_timeout_scale),
            promoted_retry_max_retries=int(satcheck_promoted_retry_max_retries),
            enable_retry_frontier_sat=bool(satcheck_retry_frontier_sat),
            enable_portfolio=bool(satcheck_portfolio),
            portfolio_size=int(satcheck_portfolio_size),
            portfolio_timeout_scale=float(satcheck_portfolio_timeout_scale),
            portfolio_min_timeout=float(satcheck_portfolio_min_timeout),
            enable_portfolio_throttle=bool(satcheck_portfolio_throttle),
            portfolio_min_size=int(satcheck_portfolio_min_size),
            enable_plateau_stop=bool(satcheck_plateau_stop),
            plateau_stop_width_ratio=float(satcheck_plateau_stop_width_ratio),
            plateau_stop_min_calls=int(satcheck_plateau_stop_min_calls),
            plateau_stop_unknown_ratio=float(satcheck_plateau_stop_unknown_ratio),
        )
        remove_n = int(search_result.remove_n)
        satcheck_records = search_result.satcheck_records
        best_sat_removed = search_result.best_sat_removed
        best_unsat_removed = search_result.best_unsat_removed
        approximate_remove_search = bool(search_result.approximate_remove_search)
    profile["remove_search_approximate"] = bool(approximate_remove_search)
    profile["satcheck_records"] = satcheck_records
    profile["best_sat_removed"] = best_sat_removed
    profile["best_unsat_removed"] = best_unsat_removed
    # Record exactly which facts are considered removed under our ordering.
    # Facts are sorted in descending I; removals correspond to releasing the tail
    # (lowest-I) facts to mirror baseline behavior.
    try:
        profile["remove_n"] = int(remove_n or 0)
        if remove_n > 0:
            profile["removed_fact_keys"] = [str(f[4]).strip() for f in facts[-int(remove_n):] if str(f[4]).strip()]
        else:
            profile["removed_fact_keys"] = []
    except Exception:
        pass
    try:
        if n_models == 0:
            _t_binsearch_end = time.perf_counter()
            logger.info("[remove-search] completed in %.3fs, minimal removal=%s facts", _t_binsearch_end - _t_binsearch_start, remove_n)
            logger.info(
                "[satcheck] calls=%d time=%.3fs time_since_ground=%.3fs timeouts=%d unknown=%d approximate=%s",
                satcheck_stats["calls"],
                satcheck_stats["sec"],
                _t_binsearch_end - t_post_ground,
                satcheck_stats["timeouts"],
                satcheck_stats["unknown"],
                bool(profile.get("remove_search_approximate", False)),
            )
    except Exception:
        pass
    if remove_n > 0:
        # Facts are released as a suffix of the sorted fact order. Log the
        # release count and the boundary facts rather than a single symbol,
        # which is easy to misread as "the only released fact".
        removed_syms = fact_syms[-remove_n:]
        threshold_sym = removed_syms[0]
        tail_sym = removed_syms[-1]
        logger.info(
            "[remove] releasing count=%s threshold=%s tail_end=%s",
            remove_n,
            threshold_sym,
            tail_sym,
        )

    # Apply the chosen removal and collect a witness model (or optimal models
    # when optimization is active), matching the earlier 'first' behavior.
    _apply_removed(remove_n)
    models = []
    last_syms = None
    logger.info("   Solving...")
    try:
        final_cfg = cast(Any, ctl.configuration).solve
    except Exception:
        final_cfg = None
    if final_cfg is not None:
        try:
            final_cfg.opt_mode = effective_final_solve_opt_mode
        except Exception:
            pass
        try:
            final_cfg.models = effective_final_solve_n_models
        except Exception:
            pass

    def _on_model_post(model):
        nonlocal last_syms
        syms = model.symbols(shown=True)
        last_syms = syms
        if effective_final_solve_opt_mode in ("opt", "optN"):
            if getattr(model, 'optimality_proven', False):
                models.append(syms)
            return
        if not models:
            models.append(syms)

    logger.info(
        "[final] starting solve opt_mode=%s models=%s timeout=%s removed=%s/%s",
        effective_final_solve_opt_mode,
        effective_final_solve_n_models,
        effective_final_solve_timeout,
        remove_n,
        len(facts),
    )
    try:
        heartbeat_status_path = str(Path(facts_location).resolve().parent / "heartbeat_final-solve.status")
    except Exception:
        heartbeat_status_path = None
    profile["heartbeat_final_solve_status_path"] = heartbeat_status_path
    if heartbeat_status_path is not None:
        logger.info("[final] heartbeat_status=%s", heartbeat_status_path)
    hb_stop, _hb_thread = _start_heartbeat(
        logger,
        phase="final-solve",
        interval_sec=60.0 * 60.0,
        describe=lambda: f"removed={remove_n}/{len(facts)}",
        status_path=heartbeat_status_path,
        log_every_beat=False,
    )
    t_f0 = time.perf_counter()
    finished, _solve_result = _solve_with_timeout(
        ctl,
        solve_timeout=effective_final_solve_timeout,
        on_model=_on_model_post,
        assumptions=_assumptions_from_exts(),
    )
    t_f1 = time.perf_counter()
    if hb_stop is not None:
        try:
            hb_stop.set()
        except Exception:
            pass
    last_solve_end = float(t_f1)
    _record_solve(t_f1 - t_f0, kind="final", timed_out=(not finished))
    try:
        logger.info("[timing] final-solve=%.3fs since-ground=%.3fs", t_f1 - t_f0, t_f1 - t_post_ground)
    except Exception:
        pass
    if not models and last_syms is not None:
        models.append(last_syms)
    n_models = int(ctl.statistics['summary']['models']['enumerated'])
    logger.info("[post-solve] models=%s", n_models)
    try:
        logger.info("[timing] total post-ground=%.3fs", time.perf_counter() - t_post_ground)
    except Exception:
        pass

    # Optional debug dump of the grounded program.
    # Historically this only dumped when the fully relaxed instance was UNSAT; we now allow
    # opting into an always-dump mode so callers can reuse the exact grounded program offline.
    if observer is not None and debug_dump_path and (debug_dump_always or (n_models == 0 and remove_n >= len(facts))):
        if _verb >= 1 or debug_enabled:
            logger.info("[debug] dumping grounded program to %s", debug_dump_path)
        try:
            note = f"remove_n={remove_n} models={n_models} timed_out={bool(profile.get('timed_out', False))}"
            observer.dump_to(debug_dump_path, ext_values, note=note)
        except Exception:
            logger.exception("Failed to dump grounded program")

    # If still UNSAT, mirror baseline behavior: return no models.
    if n_models == 0:
        return _ret([], False, remove_n)

    return _ret(models, False, remove_n)
