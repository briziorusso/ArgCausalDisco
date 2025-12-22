####passes all but gets stuck on grounding for ??

import logging
import rustworkx as rx
from clingo import Function, Number, Symbol
import os
import tracemalloc

logger = logging.getLogger(__name__)


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

try:
    from .causalaba import (
        compile_and_ground,
        extract_test_elements_from_symbol,
        PriorKnowledge,
        CausalABA as _baseline_causalaba,
    )
except ImportError:  # pragma: no cover
    from causalaba import (
        compile_and_ground,
        extract_test_elements_from_symbol,
        PriorKnowledge,
        CausalABA as _baseline_causalaba,
    )


def _solve_with_timeout(ctl, *, solve_timeout: float | None, on_model, assumptions=None) -> bool:
    """Run clingo solve with an optional hard wall-time limit.

    Returns True if finished normally, False if cancelled due to timeout.
    """
    if solve_timeout is None:
        with ctl.solve(yield_=True, assumptions=assumptions or []) as handle:
            for model in handle:
                on_model(model)
        return True

    handle = ctl.solve(async_=True, on_model=on_model, assumptions=assumptions or [])
    finished = handle.wait(solve_timeout)
    if not finished:
        handle.cancel()
        handle.wait()
        try:
            handle.get()
        except Exception:
            pass
        return False
    handle.get()
    return True


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


def _block_sym(x: int, y: int) -> Function:
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


def _set_opt_mode_for_first(ctl, opt_mode: str):
    """Prepare clingo config for `search_for_models='first'`.

    For correctness parity with the baseline (and ABAPC expectations), we must
    *not* change the requested optimization mode here. In particular, leaving
    `optN` untouched ensures we enumerate all optimal models when requested.
    """
    try:
        cfg = ctl.configuration.solve
    except Exception:
        return None

    prev = None
    try:
        prev = getattr(cfg, "opt_mode", None)
    except Exception:
        prev = None

    # Intentionally no changes to cfg.opt_mode.
    return prev


def _restore_opt_mode(ctl, prev_opt_mode):
    if prev_opt_mode is None:
        return
    try:
        ctl.configuration.solve.opt_mode = prev_opt_mode
    except Exception:
        return

    def set_ctl(self, ctl):
        self.ctl = ctl
        self._lit_map = {}

    def _ensure_map(self):
        if self.ctl is None or self._lit_map:
            return
        try:
            for atom in self.ctl.symbolic_atoms:
                self._lit_map[atom.literal] = str(atom.symbol)
        except Exception:
            pass

    def output_atom(self, symbol, atom):
        """
        Capture literal-to-symbol mapping from #show atoms.
        """
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
        return f"{sign}{sym if sym is not None else abs(lit)}"

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
        self.minimize_rules.append(f":~ {', '.join(f'{self._lit(l)}={w}' for l, w in literals)}. [{priority}]")

    def external(self, atom, value) -> None:
        self.externals.append(f"#external {self._lit(atom)}.")

    def dump_to(self, path: str, ext_values: dict[Function, bool | None], note: str = ""):
        try:
            # Build literal map before dumping so we can print symbolic names.
            self._ensure_map()
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
    ext_values: dict[Function, bool | None] | None = None,
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
    debug_traces: bool = False,
    solve_timeout: float | None = None,
    )->list:
    # For pre-grounding workloads, fall back to the baseline solver to retain
    # its proven correctness and grounding strategy.
    if pre_grounding:
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
    debug_enabled = _should_debug(debug_traces)
    logger.info("Running CausalABA")
    # Reset per-run incremental bookkeeping to avoid cross-run contamination
    _reset_incremental_state()

    indep_facts: dict[tuple, set[tuple]] = {}
    dep_facts: dict[tuple, set[tuple]] = {}
    facts = []
    ext_flag = False
    block_pairs: set[tuple[int, int]] = set()
    ap_guard_syms: list[Function] = []
    debug_dump_path = debug_dump_path or os.environ.get("CAUSALABA_DUMP")
    ext_values: dict[Function, bool | None] = {}
    enable_observer = debug_traces or bool(debug_dump_path)
    observer = _DebugObserver() if enable_observer else None
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
                try:
                    if debug_enabled:
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

    _tm_stage("after reading facts")

    # Match baseline ordering: remove lowest-I facts first.
    # Note: strength is often unknown (nan) for many callers; Python's sort is
    # stable so order is preserved when strengths are equal.
    facts = sorted(facts, key=lambda x: x[5], reverse=True)
    block_pairs |= {_pair_key(x, y) for (x, y) in indep_facts}
    if debug_enabled:
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
    control_args = [f"-t {threads}"]
    if cycle_length is not None:
        control_args += [f"-c l_cyc={int(cycle_length)}"]
    if collider_tree_depth is not None:
        control_args += [f"-c l_b={int(collider_tree_depth)}"]
    ctl = Control(control_args)
    if observer is not None:
        try:
            ctl.register_observer(observer)
            observer.set_ctl(ctl)
        except Exception:
            logger.exception("Failed to register debug observer")
    ctl.configuration.solve.parallel_mode = threads
    ctl.configuration.solve.models = out_n
    ctl.configuration.solver.seed = "2024"
    ctl.configuration.solve.opt_mode = opt_mode

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

        # Seed Bayes-ball query triples explicitly to keep grounding compact.
        # We include all conditioning triples that appear in the input facts, even
        # if some are initially inactive (fact_pct) or later removed (externals).
        # Correctness is preserved because the hard constraints are driven by
        # dep/3 and indep/3; qpair/3 only decides which ap3/3 atoms get computed.
        for (X, Y), cond_sets in dep_facts.items():
            for S in cond_sets:
                gen_f.write(f"qpair({X},{Y},{_cond_to_term(S)}).\n")
        for (X, Y), cond_sets in indep_facts.items():
            for S in cond_sets:
                gen_f.write(f"qpair({X},{Y},{_cond_to_term(S)}).\n")

    _tm_stage("after writing generated facts lp")

    # Load encoding and (optional) facts
    bounded_encoding_active = (cycle_length is not None and cycle_length > 0) or (
        collider_tree_depth is not None and collider_tree_depth > 0
    )
    encoding_file = 'causalaba_bounded.lp' if bounded_encoding_active else 'causalaba.lp'
    ctl.load(str(Path(__file__).resolve().parent / 'encodings' / encoding_file))
    if facts_location:
        ctl.load(facts_location)
        if weak_constraints:
            ctl.load(facts_location.replace('.lp','_wc.lp'))

    if ext_flag:
        ctl.add("specific", [], "indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
        ctl.add("specific", [], "dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.")

    if gen_facts_path is not None:
        try:
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

    # Provide explicit var/1 facts up-front (redundant with the encoding) so that
    # guard rules that quantify over all variable pairs can be grounded safely.
    # We keep these in a dedicated program part that we can ground early.
    for i in range(n_nodes):
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

    ctl.add("specific", [], "\n".join(bb_lines))

    _tm_stage("after ctl.add bayesball rules")

    # NOTE: We no longer rely on explicit ap/4 path atoms nor on `not ap(...)`, so
    # the earlier ap_guard_set/1 machinery is not needed.
    # Dynamic skeleton blocking is only needed/valid when using externals.
    if skeleton_rules_reduction and ext_flag and block_pairs:
        ctl.add("specific", [], ":- block_edge(X,Y), edge(X,Y), X<Y, var(X), var(Y).")
        for (a, b) in sorted(block_pairs):
            ctl.add("specific", [], f"#external block_edge({a},{b}).")

    def _assign_block_edges():
        if not skeleton_rules_reduction or not ext_flag:
            return
        # Block an undirected pair (a,b) iff there exists any *currently active*
        # ext_indep(a,b,S) (for any conditioning set S). This allows skeleton
        # unblocking purely by releasing ext_indep externals, without regrounding.
        for (a, b) in block_pairs:
            sym = _block_sym(a, b)
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



    def _assumptions_from_exts() -> list[tuple[int, bool]]:
        """
        Build solver assumptions from the current external assignments so that
        they are enforced even if assign_external was a no-op (e.g., on
        simplified programs).
        """
        assumptions: list[tuple[int, bool]] = []
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
            ctl.add("specific", [], f":- arrow({Xf},{Yf}).")
        for (Xr, Yr) in prior_knowledge.required:
            ctl.add("specific", [], f"arrow({Xr},{Yr}).")



    # Show directives
    if 'arrow' in show:
        ctl.add("base", [], "#show arrow/2.")
    if 'indep' in show:
        ctl.add("base", [], "#show indep/3.")
    if 'dep' in show:
        ctl.add("base", [], "#show dep/3.")
    if 'ap' in show:
        ctl.add("base", [], "#show ap/4.")
    if 'ext' in show:
        ctl.add("base", [], "#show ext_indep/3.")
        ctl.add("base", [], "#show ext_dep/3.")
        ctl.add("base", [], "#show block_edge/2.")
        ctl.add("base", [], "#show active_pair/2.")

    # Ground the full base
    logger.info("   Grounding full base program...")
    ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes-1)])])
    _tm_stage("after ctl.ground")
    if debug_enabled:
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

        _solve_with_timeout(
            ctl,
            solve_timeout=solve_timeout,
            on_model=_on_model_no,
            assumptions=_assumptions_from_exts(),
        )
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        if n_models > 0:
            return [models, False]

    elif search_for_models == 'first':
        for fact, ext_sym in zip(facts, fact_syms):
            ctl.assign_external(ext_sym, True)
            ext_values[ext_sym] = True
            if debug_enabled:
                logger.debug("[assign] %s = %s", ext_sym, True)
        _assign_block_edges()
        last_syms = None
        logger.info("   Solving...")
        prev_opt_mode = _set_opt_mode_for_first(ctl, opt_mode)
        try:
            def _on_model_first(model):
                nonlocal last_syms
                syms = model.symbols(shown=True)
                last_syms = syms
                if opt_mode in ("opt", "optN"):
                    if getattr(model, 'optimality_proven', False):
                        models.append(syms)
                    return
                # No optimization: first model is enough.
                if not models:
                    models.append(syms)

            _solve_with_timeout(
                ctl,
                solve_timeout=solve_timeout,
                on_model=_on_model_first,
                assumptions=_assumptions_from_exts(),
            )
        finally:
            _restore_opt_mode(ctl, prev_opt_mode)
        # Fallback: if optimization was requested but no optimality was proven (likely no #minimize),
        # return the last seen model so callers get a witness.
        if not models and last_syms is not None:
            models.append(last_syms)
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        logger.info("[first] found n_models=%s", n_models)
        _tm_stage("after first solve")
        if debug_enabled:
            logger.debug("[first] models: %s", models)
        if n_models > 0 and models:
            return [models, False]
        
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
                if fact[3] == "ext_indep" and set_indep_facts:
                    continue
                # subsets mode is inherently exponential; keep this simple and correct.
                ext_sym = Function(fact[3], [Number(fact[0]), Number(fact[2]), _cond_to_symbol(fact[1])])
                ctl.assign_external(ext_sym, False)

            _assign_block_edges()
            curr = []
            def _on_model_sub(model):
                curr.append(model.symbols(shown=True))

            _solve_with_timeout(
                ctl,
                solve_timeout=solve_timeout,
                on_model=_on_model_sub,
                assumptions=_assumptions_from_exts(),
            )
            n_curr = int(ctl.statistics['summary']['models']['enumerated'])
            if n_curr > 0:
                if search_for_models == 'first_subsets':
                    return [curr, False]
                set_of_models.append(curr)

        if len(set_of_models) > 0:
            return [set_of_models, True]

    if (search_for_models != 'first') or n_models > 0:
        return [models, False]

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

    def _is_satisfiable() -> bool:
        """Check satisfiability *without* spending time optimizing.

        Weak constraints never change satisfiable/unsatisfiable, but clingo's
        optimization can make repeated checks (during binary search) extremely
        slow. We temporarily set opt_mode=ignore and models=1.
        """
        cfg_solve = None
        prev_opt_mode = None
        prev_models = None
        try:
            try:
                cfg_solve = ctl.configuration.solve
                prev_opt_mode = getattr(cfg_solve, "opt_mode", None)
                prev_models = getattr(cfg_solve, "models", None)
                try:
                    cfg_solve.opt_mode = "ignore"
                except Exception:
                    pass
                try:
                    cfg_solve.models = 1
                except Exception:
                    pass
            except Exception:
                cfg_solve = None

            # Hard-timeout satisfiability checks too; treat timeout as "unknown"
            # and conservatively return False so the search continues.
            found = {"sat": False}

            def _on_model_sat(_model):
                found["sat"] = True

            finished = _solve_with_timeout(
                ctl,
                solve_timeout=solve_timeout,
                on_model=_on_model_sat,
                assumptions=_assumptions_from_exts(),
            )
            if found["sat"]:
                return True
            if not finished:
                return False
            try:
                res = ctl.solve(assumptions=_assumptions_from_exts())
                return bool(getattr(res, "satisfiable", False))
            except Exception:
                return False
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

    # Binary search the minimal removed count that yields SAT.
    lo = 0
    hi = len(facts)
    # If already satisfiable, no removals needed.
    if n_models == 0:
        _apply_removed(0)
        if _is_satisfiable():
            lo = 0
            hi = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if debug_enabled:
            logger.debug("[remove-search] trying removed=%s (lo=%s hi=%s)", mid, lo, hi)
        _apply_removed(mid)
        sat = _is_satisfiable()
        if sat:
            hi = mid
        else:
            lo = mid + 1

    remove_n = lo
    if remove_n > 0:
        # Log the last fact that gets released at the minimal point.
        ext_sym = fact_syms[-remove_n]
        logger.info("[remove] minimal satisfiable removal is %s (releasing %s)", remove_n, ext_sym)

    # Apply the chosen removal and collect a witness model (or optimal models
    # when optimization is active), matching the earlier 'first' behavior.
    _apply_removed(remove_n)
    models = []
    last_syms = None
    logger.info("   Solving...")
    prev_opt_mode = _set_opt_mode_for_first(ctl, opt_mode)
    try:
        def _on_model_post(model):
            nonlocal last_syms
            syms = model.symbols(shown=True)
            last_syms = syms
            if opt_mode in ("opt", "optN"):
                if getattr(model, 'optimality_proven', False):
                    models.append(syms)
                return
            if not models:
                models.append(syms)

        _solve_with_timeout(
            ctl,
            solve_timeout=solve_timeout,
            on_model=_on_model_post,
            assumptions=_assumptions_from_exts(),
        )
    finally:
        _restore_opt_mode(ctl, prev_opt_mode)
    if not models and last_syms is not None:
        models.append(last_syms)
    n_models = int(ctl.statistics['summary']['models']['enumerated'])
    logger.info("[post-solve] models=%s", n_models)

    # Optional debug dump when even the fully relaxed instance is UNSAT.
    if observer is not None and n_models == 0 and debug_dump_path and remove_n >= len(facts):
        logger.info("[debug] dumping grounded program to %s", debug_dump_path)
        try:
            observer.dump_to(debug_dump_path, ext_values, note="unsat after releasing all facts")
        except Exception:
            logger.exception("Failed to dump grounded program")

    # If still UNSAT, mirror baseline behavior: return no models.
    if n_models == 0:
        return [[], False]

    return [models, False]
