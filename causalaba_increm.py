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
            if cutoff is not None
            else rx.all_simple_paths(graph, src, dst)
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
_ACTIVE_PAIR_STATE: dict[tuple[int, int], bool] = {}


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
    global _INCR_COUNTER, _PAIR_PATHS_ADDED
    debug_enabled = _should_debug(debug_traces)
    G = rx.generators.complete_graph(n_nodes)
    # Prune skeleton as compile_and_ground would: remove undirected edges for
    # any pair that has (at least one) independence statement.
    forbidden_edges = set(indep_facts)
    if debug_enabled:
        logger.debug("[increm] Pruning skeleton for pairs: %s", forbidden_edges)
    if forbidden_edges:
        # rustworkx raises if asked to remove a non-existent edge; intersect first.
        existing_edges = set(G.edge_list())
        G.remove_edges_from(existing_edges & forbidden_edges)
    if prior_knowledge is not None:
        req_dir = set(prior_knowledge.required)
        req_undirected = {tuple(sorted((a, b))) for (a, b) in req_dir}
        forb_dir = set(prior_knowledge.forbidden)
        forb_counts = {}
        for a, b in forb_dir:
            key = tuple(sorted((a, b)))
            forb_counts[key] = forb_counts.get(key, 0) + 1
        to_remove = set()
        for (u, v) in G.edge_list():
            key = tuple(sorted((u, v)))
            if key in req_undirected:
                continue
            if forb_counts.get(key, 0) >= 2:
                to_remove.add((u, v))
        if to_remove:
            G.remove_edges_from(to_remove)

    use_bounded_nb = collider_tree_depth is not None and collider_tree_depth > 0
    nb_pred = 'nb_b' if use_bounded_nb else 'nb'

    rules: list[str] = []
    new_paths: list[tuple[int, ...]] = []
    part_id = None
    qid = 0
    already = _PAIR_PATHS_ADDED.setdefault((X, Y), set())
    for path in _iter_paths_with_cutoff(G, X, Y, max_path_length):
        if tuple(path) in already:
            continue
        if debug_enabled:
            logger.debug("[increm] Adding path rule for pair (%s,%s): %s", X, Y, path)
        if part_id is None:
            _INCR_COUNTER += 1
            part_id = _INCR_COUNTER
        qid += 1
        path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
        qsym = f"q{part_id}_{qid}"
        rules.append(f"{qsym} :- active_pair({X},{Y}), {','.join(path_edges)}.")
        if pre_grounding:
            condition_sets = set()
            if (X, Y) in dep_facts:
                condition_sets.update(dep_facts[(X, Y)])
            if (X, Y) in indep_facts:
                condition_sets.update(indep_facts[(X, Y)])
                for S in condition_sets:
                    if max_conditioning_size is not None and len(S) > max_conditioning_size:
                        continue
                    s_str = 'empty' if not S else 's' + 'y'.join([str(i) for i in S])
                    nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},{s_str})" for idx in range(1, len(path)-1)]
                    nbs_str = ", " + ','.join(nbs) if len(nbs) > 0 else ""
                    rules.append(f"ap({X},{Y},{qsym},{s_str}) :- {qsym}, active_pair({X},{Y}){nbs_str}.")
                    # Restore the consistency check for this concrete conditioning set so it
                    # is not simplified away when new dep/indep atoms appear incrementally.
                    rules.append(
                        f":- dep({X},{Y},{s_str}), indep({X},{Y},{s_str}), not in({X},{s_str}), not in({Y},{s_str})."
                    )
                    if S in indep_facts.get((X, Y), set()):
                        ext_premise = f"ext_indep({X},{Y},{s_str}), " if ext_flag else ""
                        rules.append(f"dep({X},{Y},{s_str}) :- {ext_premise}ap({X},{Y},{qsym},{s_str}).")
        else:
            nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1, len(path)-1)]
            nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
            rules.append(f"ap({X},{Y},{qsym},S) :- {qsym}, active_pair({X},{Y}), {nbs_str} not in({X},S), not in({Y},S), set(S).")
        new_paths.append(tuple(path))
    if pre_grounding and (X, Y) in dep_facts:
        for S in dep_facts[(X, Y)]:
            if max_conditioning_size is not None and len(S) > max_conditioning_size:
                continue
            s_str = 'empty' if not S else 's' + 'y'.join([str(i) for i in S])
            ext_premise = f"ext_dep({X},{Y},{s_str}), " if ext_flag else ""
            rules.append(f"indep({X},{Y},{s_str}) :- {ext_premise}not ap({X},{Y},_,{s_str}).")
    if rules and part_id is not None:
        part_name = f"incr_{part_id}"
        logger.info("Grounding incremental part for pair (%s,%s) with %s rules.", X, Y, len(rules))
        active_pair_sym = Function("active_pair", [Number(X), Number(Y)])
        decls: list[str] = []
        if (X, Y) not in _ACTIVE_PAIR_STATE:
            decls.append(f"#external active_pair({X},{Y}).")
        payload = "\n".join(decls + rules)
        try:
            ctl.add(part_name, [], payload)
            ctl.ground([(part_name, [])])
            ctl.assign_external(active_pair_sym, True)
            _ACTIVE_PAIR_STATE[(X, Y)] = True
            if ext_values is not None:
                ext_values[active_pair_sym] = True
        except Exception:
            try:
                sym_summary = {}
                for atom in ctl.symbolic_atoms:
                    name = str(atom.symbol)
                    sym_summary[name] = sym_summary.get(name, 0) + 1
                logger.error(
                    "[increm] symbolic atom names before failure (count=%s): %s",
                    len(sym_summary),
                    list(sym_summary.keys()),
                )
            except Exception:
                pass
            logger.exception("[increm] failed to ground part %s with rules:\n%s", part_name, payload)
            raise
        _PAIR_PATHS_ADDED[(X, Y)].update(new_paths)
        logger.info("[increm] Added %s rules for pair (%s,%s) in part %s", len(rules), X, Y, part_name)
        if debug_enabled:
            logger.debug("[increm] Rules:\n%s", "\n".join(rules))


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
    global _INCR_COUNTER
    _INCR_COUNTER = 0
    global _ACTIVE_PAIR_STATE
    _ACTIVE_PAIR_STATE = {}
    global _PAIR_PATHS_ADDED
    _PAIR_PATHS_ADDED = {}

    indep_facts: dict[tuple, set[tuple]] = {}
    dep_facts: dict[tuple, set[tuple]] = {}
    facts = []
    ext_flag = False
    block_pairs: set[tuple[int, int]] = set()
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

    ctl.add("specific", [], "indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    ctl.add("specific", [], "dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    if skeleton_rules_reduction and block_pairs:
        ctl.add("specific", [], ":- block_edge(X,Y), edge(X,Y), X<Y, var(X), var(Y).")
        for (a, b) in sorted(block_pairs):
            ctl.add("specific", [], f"#external block_edge({a},{b}).")

    def _assign_block_edges():
        if not skeleton_rules_reduction:
            return
        active_pairs = {_pair_key(x, y) for (x, y) in indep_facts}
        for (a, b) in block_pairs:
            sym = _block_sym(a, b)
            val = (a, b) in active_pairs
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
            # Keep prefix facts true; release removed facts instead of forcing false.
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
            if sym.name in ("active_pair", "block_edge"):
                assumptions.append((sym, bool(val)))
        return assumptions

    # Build skeleton and add path rules without adding ":- edge(X,Y)." constraints
    logger.info("   Adding Specific Rules...")
    n_p = 0
    G = rx.generators.complete_graph(n_nodes)
    forbidden_edges = set()
    if skeleton_rules_reduction:
        forbidden_edges = set(indep_facts)
        if debug_enabled:
            logger.debug("[skeleton] Forbidden edges: %s", forbidden_edges)
        if forbidden_edges:
            existing_edges = set(G.edge_list())
            G.remove_edges_from(existing_edges & forbidden_edges)
    if prior_knowledge is not None:
        # prune undirected skeleton using prior knowledge (symmetric with causalaba)
        req_dir = set(prior_knowledge.required)
        req_undirected = {tuple(sorted((a, b))) for (a, b) in req_dir}
        forb_dir = set(prior_knowledge.forbidden)
        forb_counts = {}
        for a, b in forb_dir:
            key = tuple(sorted((a, b)))
            forb_counts[key] = forb_counts.get(key, 0) + 1
        to_remove = set()
        for (u, v) in G.edge_list():
            key = tuple(sorted((u, v)))
            if key in req_undirected:
                continue
            if forb_counts.get(key, 0) >= 2:
                to_remove.add((u, v))
        if to_remove:
            G.remove_edges_from(to_remove)

        # Enforce prior knowledge on arrows as in original compile_and_ground
        for (Xf, Yf) in prior_knowledge.forbidden:
            ctl.add("specific", [], f":- arrow({Xf},{Yf}).")
        for (Xr, Yr) in prior_knowledge.required:
            ctl.add("specific", [], f"arrow({Xr},{Yr}).")

    node_pairs = tuple(dep_facts | indep_facts) if skeleton_rules_reduction else tuple(combinations(range(n_nodes),2))
    use_bounded_nb = bounded_encoding_active and collider_tree_depth is not None and collider_tree_depth > 0
    for (X, Y) in node_pairs:
        if debug_enabled:
            logger.debug("[skeleton] Adding path rules for pair (%s,%s)", X, Y)
        for path in _iter_paths_with_cutoff(G, X, Y, max_path_length):
            if debug_enabled:
                logger.debug("[skeleton] Found path for pair (%s,%s): %s", X, Y, path)
            n_p += 1
            path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
            ctl.add("specific", [], f"p{n_p} :- {','.join(path_edges)}.")
            if debug_enabled:
                logger.debug("[rules] added specific p%s :- %s.", n_p, ",".join(path_edges))
            nb_pred = 'nb_b' if use_bounded_nb else 'nb'
            if pre_grounding:
                condition_sets = set()
                if (X, Y) in dep_facts:
                    condition_sets.update(dep_facts[(X, Y)])
                if (X, Y) in indep_facts:
                    condition_sets.update(indep_facts[(X, Y)])
                for S in condition_sets:
                    if max_conditioning_size is not None and len(S) > max_conditioning_size:
                        continue
                    s_str = 'empty' if not S else 's'+'y'.join([str(i) for i in S])
                    nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},{s_str})" for idx in range(1,len(path)-1)]
                    nbs_str = ", " + ','.join(nbs) if len(nbs) > 0 else ""
                    ctl.add("specific", [], f"ap({X},{Y},p{n_p},{s_str}) :- p{n_p}{nbs_str}.")
                    if S in indep_facts.get((X, Y), set()):
                        ext_premise = f"ext_indep({X},{Y},{s_str}), " if ext_flag else ""
                        ctl.add("specific", [], f"dep({X},{Y},{s_str}) :- {ext_premise}ap({X},{Y},p{n_p},{s_str}).")
            else:
                nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1,len(path)-1)]
                nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
                ctl.add("specific", [], f"ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")
            _PAIR_PATHS_ADDED.setdefault((X, Y), set()).add(tuple(path))

        if (X, Y) in dep_facts:
            if pre_grounding:
                for S in dep_facts[(X, Y)]:
                    if max_conditioning_size is not None and len(S) > max_conditioning_size:
                        continue
                    s_str = 'empty' if not S else 's' + 'y'.join([str(i) for i in S])
                    ext_premise = f"ext_dep({X},{Y},{s_str}), " if ext_flag else ""
                    ctl.add("specific", [], f"indep({X},{Y},{s_str}) :- {ext_premise}not ap({X},{Y},_,{s_str}).")
            else:
                ext_premise = f"ext_dep({X},{Y},S), " if ext_flag else ""
                ctl.add("specific", [], f"indep({X},{Y},S) :- {ext_premise}not ap({X},{Y},_,S), set(S).")
        if (X, Y) in indep_facts and pre_grounding is False:
            ext_premise = f"ext_indep({X},{Y},S), " if ext_flag else ""
            ctl.add("specific", [], f"dep({X},{Y},S) :- {ext_premise}ap({X},{Y},_,S), set(S).")


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
        # Disable the external so it no longer participates in solving
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

        # If we just removed an independence fact, mimic the baseline regrounding
        # behavior (which recompiles the program when skeleton_rules_reduction is
        # enabled). This keeps the conditioning-set domain and skeleton pruning in
        # sync with the remaining facts, avoiding spurious UNSAT/extra models.
        if skeleton_rules_reduction and dep_type == "ext_indep":
            if debug_enabled:
                logger.debug("[reground] rebuilding control after removing %s", ext_sym)
            # Rebuild a fresh Control mirroring the initial compilation step.
            from clingo.control import Control
            control_args = []
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
                    logger.exception("Failed to register debug observer on reground")
            ctl.configuration.solve.parallel_mode = threads or min(os.cpu_count() or 1, 64)
            ctl.configuration.solve.models = out_n
            ctl.configuration.solver.seed = "2024"
            ctl.configuration.solve.opt_mode = opt_mode

            # Recreate set facts based on remaining facts (matching baseline).
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

            bounded_encoding_active = (cycle_length is not None and cycle_length > 0) or (
                collider_tree_depth is not None and collider_tree_depth > 0
            )
            encoding_file = 'causalaba_bounded.lp' if bounded_encoding_active else 'causalaba.lp'
            ctl.load(str(Path(__file__).resolve().parent / 'encodings' / encoding_file))
            if facts_location:
                ctl.load(facts_location)
                if weak_constraints:
                    ctl.load(facts_location.replace('.lp','_wc.lp'))

            ctl.add("specific", [], "indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
            ctl.add("specific", [], "dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.")

            # Rebuild skeleton/path rules using current facts (as compile_and_ground does)
            G_reg = rx.generators.complete_graph(n_nodes)
            if skeleton_rules_reduction:
                forbidden_edges = set(indep_facts)
                if forbidden_edges:
                    G_reg.remove_edges_from(set(G_reg.edge_list()) & forbidden_edges)
                for (Xf, Yf) in forbidden_edges:
                    ctl.add("specific", [], f":- edge({Xf},{Yf}).")
            if prior_knowledge is not None:
                req_dir = set(prior_knowledge.required)
                req_undirected = {tuple(sorted((a, b))) for (a, b) in req_dir}
                forb_dir = set(prior_knowledge.forbidden)
                forb_counts = {}
                for a, b in forb_dir:
                    key = tuple(sorted((a, b)))
                    forb_counts[key] = forb_counts.get(key, 0) + 1
                to_remove = set()
                for (u, v) in G_reg.edge_list():
                    key = tuple(sorted((u, v)))
                    if key in req_undirected:
                        continue
                    if forb_counts.get(key, 0) >= 2:
                        to_remove.add((u, v))
                if to_remove:
                    G_reg.remove_edges_from(to_remove)
                for (Xf, Yf) in prior_knowledge.forbidden:
                    if not skeleton_rules_reduction or ((Xf, Yf) not in forbidden_edges and (Yf, Xf) not in forbidden_edges):
                        ctl.add("specific", [], f":- arrow({Xf},{Yf}).")
                for (Xr, Yr) in prior_knowledge.required:
                    if not skeleton_rules_reduction or ((Xr, Yr) not in forbidden_edges and (Yr, Xr) not in forbidden_edges):
                        ctl.add("specific", [], f"arrow({Xr},{Yr}).")

            node_pairs = tuple(dep_facts | indep_facts) if skeleton_rules_reduction else tuple(combinations(range(n_nodes),2))
            use_bounded_nb = bounded_encoding_active and collider_tree_depth is not None and collider_tree_depth > 0
            n_p = 0
            for (Xr, Yr) in node_pairs:
                for path in _iter_paths_with_cutoff(G_reg, Xr, Yr, max_path_length):
                    n_p += 1
                    path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
                    ctl.add("specific", [], f"p{n_p} :- {','.join(path_edges)}.")
                    nb_pred = 'nb_b' if use_bounded_nb else 'nb'
                    nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1,len(path)-1)]
                    nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
                    ctl.add("specific", [], f"ap({Xr},{Yr},p{n_p},S) :- p{n_p}, {nbs_str} not in({Xr},S), not in({Yr},S), set(S).")
                if (Xr, Yr) in dep_facts:
                    ext_premise = f"ext_dep({Xr},{Yr},S), " if ext_flag else ""
                    ctl.add("specific", [], f"indep({Xr},{Yr},S) :- {ext_premise}not ap({Xr},{Yr},_,S), set(S).")
                if (Xr, Yr) in indep_facts:
                    ext_premise = f"ext_indep({Xr},{Yr},S), " if ext_flag else ""
                    ctl.add("specific", [], f"dep({Xr},{Yr},S) :- {ext_premise}ap({Xr},{Yr},_,S), set(S).")

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

            ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes-1)])])

            # Reset incremental bookkeeping
            _ACTIVE_PAIR_STATE = {}
            _PAIR_PATHS_ADDED = {}
            ext_values.clear()
            # Reassign externals: remaining facts true, removed prefix released
            _assign_fact_externals(remove_n)
            _assign_block_edges()
            if debug_enabled:
                logger.debug("[reground] finished; continuing with rebuilt program")
            # Skip incremental additions below for this iteration
            models = []
            last_syms = None
            logger.info("   Solving...")
            with ctl.solve(yield_=True) as handle:
                for i, model in enumerate(handle):
                    syms = model.symbols(shown=True)
                    last_syms = syms
                    if getattr(model, 'optimality_proven', False):
                        models.append(syms)
                    if opt_mode not in ("opt", "optN"):
                        models.append(syms)
                        break
            if not models and last_syms is not None:
                models.append(last_syms)
            n_models = int(ctl.statistics['summary']['models']['enumerated'])
            logger.info("[post-solve] models=%s", n_models)
            if n_models > 0:
                return [models, False]
            # continue loop if still no models
            continue
        # Recompute active pairs to mirror baseline regrounding behavior
        node_pairs = tuple(dep_facts | indep_facts) if skeleton_rules_reduction else tuple(combinations(range(n_nodes), 2))
        # If a pair no longer has any facts, deactivate its path rules
        pair_has_fact = (
            (rem_X, rem_Y) in indep_facts
            or (rem_X, rem_Y) in dep_facts
            or (rem_Y, rem_X) in indep_facts
            or (rem_Y, rem_X) in dep_facts
        )
        if debug_enabled:
            logger.debug("[active_pair] (%s,%s) remaining facts? %s", rem_X, rem_Y, pair_has_fact)
        if not pair_has_fact:
            if (rem_X, rem_Y) in _ACTIVE_PAIR_STATE:
                if debug_enabled:
                    logger.debug("[active_pair] deactivating (%s,%s)", rem_X, rem_Y)
                ctl.assign_external(Function("active_pair", [Number(rem_X), Number(rem_Y)]), False)
                ext_values[Function("active_pair", [Number(rem_X), Number(rem_Y)])] = False
                _ACTIVE_PAIR_STATE[(rem_X, rem_Y)] = False
            if (rem_Y, rem_X) in _ACTIVE_PAIR_STATE:
                if debug_enabled:
                    logger.debug("[active_pair] deactivating (%s,%s)", rem_Y, rem_X)
                ctl.assign_external(Function("active_pair", [Number(rem_Y), Number(rem_X)]), False)
                ext_values[Function("active_pair", [Number(rem_Y), Number(rem_X)])] = False
                _ACTIVE_PAIR_STATE[(rem_Y, rem_X)] = False

        if skeleton_rules_reduction:
            if dep_type == "ext_indep":
                still_blocked = ((rem_X, rem_Y) in indep_facts) or ((rem_Y, rem_X) in indep_facts)
                skeleton_expanded = not still_blocked
            if skeleton_expanded:
                # Refresh path rules and constraints only when the skeleton expands.
                for (px, py) in node_pairs:
                    pre_pair_ap = _count_ap_for_pair(ctl, px, py)
                    _add_pair_rules_incremental(
                        ctl,
                        n_nodes,
                        px,
                        py,
                        indep_facts=indep_facts,
                        dep_facts=dep_facts,
                        ext_flag=ext_flag,
                        prior_knowledge=prior_knowledge,
                        max_path_length=max_path_length,
                        max_conditioning_size=max_conditioning_size,
                        collider_tree_depth=collider_tree_depth,
                        pre_grounding=pre_grounding,
                        ext_values=ext_values,
                    )
                    post_pair_ap = _count_ap_for_pair(ctl, px, py)
                    if debug_enabled:
                        logger.debug(
                            "[increm] ap(%s,%s,_,_) count Δ=%s (pre=%s → post=%s)",
                            px,
                            py,
                            post_pair_ap - pre_pair_ap,
                            pre_pair_ap,
                            post_pair_ap,
                        )
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
