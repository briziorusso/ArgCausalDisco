####passes all but gets stuck on grounding for ??

import logging
import rustworkx as rx
from clingo import Function, Number
import os

logger = logging.getLogger(__name__)


def _should_debug(verbose: bool) -> bool:
    return verbose and logger.isEnabledFor(logging.DEBUG)

from causalaba import (
    compile_and_ground,
    extract_test_elements_from_symbol,
    PriorKnowledge,
    CausalABA as _baseline_causalaba,
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


def _cond_to_symbol(S) -> Function:
    """
    Convert a conditioning set into the clingo symbol used in facts.
    """
    try:
        # Already a clingo term (e.g., Function('empty'))
        if isinstance(S, Function):
            return S
        if isinstance(S, str):
            return Function(S)
        S = tuple(sorted(S))
    except Exception:
        pass
    if not S:
        return Function("empty")
    return Function("s" + "y".join(str(i) for i in S))


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
                condition_set = tuple(sorted(list(S)))
                facts_group = indep_facts if "indep" in line_clean else dep_facts
                if (X,Y) not in facts_group:
                    facts_group[(X,Y)] = set()
                facts_group[(X,Y)].add(condition_set)
                if "indep" in line_clean:
                    block_pairs.add(_pair_key(X, Y))

    # Match the baseline solver ordering: remove lowest-I facts first.
    # Match baseline ordering: sort by strength only.
    facts = sorted(facts, key=lambda x: x[5], reverse=True)
    block_pairs |= {_pair_key(x, y) for (x, y) in indep_facts}
    if debug_enabled:
        logger.debug("Ordered facts for processing:\n%s", "\n".join([f[4] for f in facts]))
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
    from utils.graph_utils import powerset as _powerset
    base_condition_sets = (
        set().union(*indep_facts.values(), *dep_facts.values())
        if skeleton_rules_reduction
        else _powerset(range(n_nodes))
    )
    for S in base_condition_sets:
        if max_conditioning_size is not None and len(S) > max_conditioning_size:
            continue
        for s in S:
            ctl.add("specific", [], f"in({s}," + ('empty' if not S else 's'+'y'.join([str(i) for i in S])) + ").")

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
    bb_lines.append("qpair(X,Y,S) :- dep(X,Y,S), set(S), var(X), var(Y), X<Y, X!=Y.")
    bb_lines.append("qpair(X,Y,S) :- indep(X,Y,S), set(S), var(X), var(Y), X<Y, X!=Y.")
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
        for idx, fact in enumerate(facts):
            sym = Function(fact[3], [Number(fact[0]), Number(fact[2]), _cond_to_symbol(fact[1])])
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
        for n, fact in enumerate(facts):
            ext_sym = Function(fact[3], [Number(fact[0]), Number(fact[2]), _cond_to_symbol(fact[1])])
            if fact[3] == "ext_indep" and set_indep_facts:
                ctl.assign_external(ext_sym, True)
            elif n/len(facts) <= fact_pct:
                ctl.assign_external(ext_sym, True)
            else:
                ctl.assign_external(ext_sym, False)
            ext_values[ext_sym] = not (n/len(facts) > fact_pct and fact[3] != "ext_indep")
        _assign_block_edges()
        with ctl.solve(yield_=True, assumptions=_assumptions_from_exts()) as handle:
            for model in handle:
                models.append(model.symbols(shown=True))
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        if n_models > 0:
            return [models, False]

    elif search_for_models == 'first':
        for fact in facts:
            ext_sym = Function(fact[3], [Number(fact[0]), Number(fact[2]), _cond_to_symbol(fact[1])])
            ctl.assign_external(ext_sym, True)
            ext_values[ext_sym] = True
            if debug_enabled:
                logger.debug("[assign] %s = %s", ext_sym, True)
        _assign_block_edges()
        last_syms = None
        logger.info("   Solving...")
        with ctl.solve(yield_=True, assumptions=_assumptions_from_exts()) as handle:
            for i, model in enumerate(handle):
                syms = model.symbols(shown=True)
                last_syms = syms
                # Prefer collecting all optimal models when optimization is active
                if getattr(model, 'optimality_proven', False):
                    models.append(syms)
                # If no optimization is present, taking the first model is fine
                if opt_mode not in ("opt", "optN"):
                    models.append(syms)
                    break
        # Fallback: if optimization was requested but no optimality was proven (likely no #minimize),
        # return the last seen model so callers get a witness.
        if not models and last_syms is not None:
            models.append(last_syms)
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        logger.info("[first] found n_models=%s", n_models)
        if debug_enabled:
            logger.debug("[first] models: %s", models)
        if n_models > 0 and models:
            return [models, False]
        
    elif 'subsets' in search_for_models:
        # Explore all subsets of facts by toggling externals; collect solutions.
        from utils.graph_utils import powerset as _powerset
        set_of_models = []
        for f_to_remove in _powerset(facts):
            # Set all facts True first
            for fact in facts:
                ext_sym = Function(fact[3], [Number(fact[0]), Number(fact[2]), _cond_to_symbol(fact[1])])
                ctl.assign_external(ext_sym, True)
            # Then disable the chosen subset (unless ext_indep with set_indep_facts flag)
            for fact in f_to_remove:
                if fact[3] == "ext_indep" and set_indep_facts:
                    continue
                ext_sym = Function(fact[3], [Number(fact[0]), Number(fact[2]), _cond_to_symbol(fact[1])])
                ctl.assign_external(ext_sym, False)

            _assign_block_edges()
            curr = []
            with ctl.solve(yield_=True, assumptions=_assumptions_from_exts()) as handle:
                for model in handle:
                    curr.append(model.symbols(shown=True))
            n_curr = int(ctl.statistics['summary']['models']['enumerated'])
            if n_curr > 0:
                if search_for_models == 'first_subsets':
                    return [curr, False]
                set_of_models.append(curr)

        if len(set_of_models) > 0:
            return [set_of_models, True]

    if (search_for_models != 'first') or n_models > 0:
        return [models, False]

    remove_n = 0

    # (no dep_guard refresh: ext_dep semantics stay reactive via ap_guard_set + not ap)
    while n_models == 0 and remove_n < len(facts):
        remove_n += 1
        fact_to_remove = facts[-remove_n]
        rem_X, rem_S, rem_Y, dep_type, fact_str = fact_to_remove[:5]
        skeleton_expanded = False
        if debug_enabled and remove_n == 1:
            indep_cnt = _count_atoms(ctl, 'indep', 3)
            dep_cnt = _count_atoms(ctl, 'dep', 3)
            ap_cnt = _count_atoms(ctl, 'ap', 4)
            logger.debug(
                "[pre-remove] atoms indep=%s, dep=%s, ap=%s; facts=%s",
                indep_cnt,
                dep_cnt,
                ap_cnt,
                len(facts),
            )
            if facts:
                last = facts[-1]
                logger.debug("[pre-remove] last-to-remove candidate: %s", last[:5])
        logger.info(
            "[remove] removing external %s(%s,%s,%s)",
            dep_type,
            rem_X,
            rem_Y,
            rem_S,
        )
        facts_group = indep_facts if dep_type == "ext_indep" else dep_facts
        if debug_enabled:
            logger.debug(
                "[in-remove] indep facts for pair (%s,%s): %s",
                rem_X,
                rem_Y,
                facts_group.get((rem_X, rem_Y), set()),
            )
        target_key = (rem_X, rem_Y)
        existing = facts_group.get(target_key, set())
        if debug_enabled:
            logger.debug("[in-remove] existing: %s", existing)

        to_remove = None
        for t in existing:
            if set(t) == set(rem_S):
                to_remove = t
                break
        if to_remove is not None:
            if debug_enabled:
                logger.debug(
                    "[in-remove] removing S for pair (%s,%s): %s",
                    rem_X,
                    rem_Y,
                    to_remove,
                )
            existing.remove(to_remove)
            if not existing:
                del facts_group[target_key]
        # Release the removed external (None) to match baseline behavior.
        ext_sym = Function(dep_type, [Number(rem_X), Number(rem_Y), _cond_to_symbol(rem_S)])
        ctl.assign_external(ext_sym, None)
        try:
            ext_values[ext_sym] = None
        except Exception:
            pass
        # Re-apply remaining fact assignments defensively; this also ensures
        # previously-assigned externals (e.g., ext_dep) stay enabled.
        _assign_fact_externals(remove_n)

        if debug_enabled:
            logger.debug("[in-remove] existing: %s", existing)
            logger.debug(
                "[in-remove] indep facts all for pair (%s,%s): %s",
                rem_X,
                rem_Y,
                indep_facts.get((rem_X, rem_Y), set()),
            )
        _assign_block_edges()

        if skeleton_rules_reduction and dep_type == "ext_indep":
            # If an indep fact removal unblocks an edge, new simple paths become possible.
            # Those new paths can affect any remaining fact-pair (dep or indep) whose
            # connectivity relies on the unblocked edge.
            still_blocked = ((rem_X, rem_Y) in indep_facts) or ((rem_Y, rem_X) in indep_facts)
            skeleton_expanded = not still_blocked
            if skeleton_expanded:
                node_pairs_facts = list(dep_facts | indep_facts)

                # Build the *previous* skeleton (before this removal) to detect whether
                # the removed edge was a bridge between components.
                forbidden_old = set(indep_facts)
                forbidden_old.add(_pair_key(rem_X, rem_Y))
                G_old = rx.generators.complete_graph(n_nodes)
                if forbidden_old:
                    existing_edges = set(G_old.edge_list())
                    G_old.remove_edges_from(existing_edges & forbidden_old)

                if prior_knowledge is not None:
                    req_dir = set(prior_knowledge.required)
                    req_undirected = {tuple(sorted((a, b))) for (a, b) in req_dir}
                    forb_dir = set(prior_knowledge.forbidden)
                    forb_counts = {}
                    for a, b in forb_dir:
                        key = tuple(sorted((a, b)))
                        forb_counts[key] = forb_counts.get(key, 0) + 1
                    to_remove = set()
                    for (u, v) in G_old.edge_list():
                        key = tuple(sorted((u, v)))
                        if key in req_undirected:
                            continue
                        if forb_counts.get(key, 0) >= 2:
                            to_remove.add((u, v))
                    if to_remove:
                        G_old.remove_edges_from(to_remove)

                # Compute components in the old skeleton. If the removed edge connects
                # two components, only pairs across those components can gain paths.
                import networkx as _nx

                H_old = _nx.Graph()
                H_old.add_nodes_from(range(n_nodes))
                H_old.add_edges_from(G_old.edge_list())
                comps = list(_nx.connected_components(H_old))
                comp_by_node = {}
                for idx, comp in enumerate(comps):
                    for node in comp:
                        comp_by_node[int(node)] = idx

                cx = comp_by_node.get(int(rem_X))
                cy = comp_by_node.get(int(rem_Y))
                if cx is not None and cy is not None and cx != cy:
                    comp_x = comps[cx]
                    comp_y = comps[cy]
                    affected_pairs = [
                        (px, py)
                        for (px, py) in node_pairs_facts
                        if (px in comp_x and py in comp_y) or (px in comp_y and py in comp_x)
                    ]
                else:
                    # Edge was not a strict bridge; conservatively update all fact-pairs.
                    affected_pairs = node_pairs_facts

                if debug_enabled:
                    logger.debug(
                        "[increm] skeleton expanded by unblocking (%s,%s); updating %s pairs",
                        rem_X,
                        rem_Y,
                        len(affected_pairs),
                    )

                # No incremental grounding here: paths were pre-grounded on the full
                # skeleton. Unblocking edges is handled by _assign_block_edges() based
                # on current ext_indep assignments.

        if debug_enabled:
            for (dx, dy) in dep_facts:
                logger.debug("[ap-sets] pair (%s,%s) has ap sets: %s", dx, dy, _ap_sets_for_pair(ctl, dx, dy))

        # Defensive: re-apply fact externals after incremental grounding. This keeps
        # behavior aligned with the baseline solver, which reassigns externals after
        # each regrounding step.
        _assign_fact_externals(remove_n)
        _assign_block_edges()
        if debug_enabled:
            try:
                # Sanity-check that ext_dep assignments are actually fixed in the solver.
                dep_syms = [
                    Function(f[3], [Number(f[0]), Number(f[2]), _cond_to_symbol(f[1])])
                    for f in facts
                    if f[3] == "ext_dep"
                ]
                if dep_syms:
                    check = ctl.solve(assumptions=[(dep_syms[0], False)])
                    logger.debug("[debug] assume not %s satisfiable=%s", dep_syms[0], check.satisfiable)
            except Exception:
                logger.debug("[debug] failed ext_dep assumption check", exc_info=True)

        models = []
        last_syms = None
        logger.info("   Solving...")
        with ctl.solve(yield_=True, assumptions=_assumptions_from_exts()) as handle:
            for i, model in enumerate(handle):
                syms = model.symbols(shown=True)
                last_syms = syms
                if getattr(model, 'optimality_proven', False):
                    models.append(syms)
                if opt_mode not in ("opt", "optN"):
                    models.append(syms)
                    break
            result = handle.get()
        if not models and last_syms is not None:
            models.append(last_syms)
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        logger.info("[post-solve] models=%s", n_models)
        if debug_enabled:
            logger.debug("[post-solve] models: %s", models)
        try:
            if debug_enabled:
                logger.debug("[post-solve] satisfiable=%s", result.satisfiable)
        except Exception:
            pass
        if observer is not None and n_models == 0 and debug_dump_path and remove_n == 1:
            logger.info("[debug] dumping grounded program to %s", debug_dump_path)
            try:
                observer.dump_to(debug_dump_path, ext_values, note=f"unsat after removal {remove_n}")
            except Exception:
                logger.exception("Failed to dump grounded program")
        if n_models == 0 and debug_enabled:
            try:
                in_atoms = [str(it.symbol) for it in ctl.symbolic_atoms.by_signature('in', 2)]
                set_atoms = [str(it.symbol) for it in ctl.symbolic_atoms.by_signature('set', 1)]
                logger.debug("[debug] in/2 atoms: %s", in_atoms)
                logger.debug("[debug] set/1 atoms: %s", set_atoms)
                assumptions = []
                def _lit(sym):
                    for atom in ctl.symbolic_atoms.by_signature(sym.name, len(sym.arguments)):
                        if atom.symbol == sym:
                            return atom.literal
                    return None
                for (ix, iy), sets in indep_facts.items():
                    for S in sets:
                        sym = Function("ext_indep", [Number(ix), Number(iy), _cond_to_symbol(S)])
                        lit = _lit(sym)
                        if lit is not None:
                            assumptions.append((lit, True))
                for (dx, dy), sets in dep_facts.items():
                    for S in sets:
                        sym = Function("ext_dep", [Number(dx), Number(dy), _cond_to_symbol(S)])
                        lit = _lit(sym)
                        if lit is not None:
                            assumptions.append((lit, True))
                for (ax, ay), state in _ACTIVE_PAIR_STATE.items():
                    sym = Function("active_pair", [Number(ax), Number(ay)])
                    lit = _lit(sym)
                    if lit is not None:
                        assumptions.append((lit, bool(state)))
                core_res = ctl.solve(assumptions=assumptions)
                core = getattr(core_res, "unsat_core", [])
                logger.debug(
                    "[debug] assumption solve satisfiable=%s core_size=%s core=%s",
                    core_res.satisfiable,
                    len(core),
                    [str(l) for l in core],
                )
            except Exception:
                pass
    if observer is not None and debug_dump_path:
        logger.info("[debug] dumping grounded program to %s", debug_dump_path)
        try:
            observer.dump_to(debug_dump_path, ext_values, note=f"final after removal {remove_n}")
        except Exception:
            logger.exception("Failed to dump grounded program")
    return [models, False]
