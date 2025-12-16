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
    collider_tree_depth: int | None,
    debug_traces: bool = False,
):
    global _INCR_COUNTER, _PAIR_PATHS_ADDED
    debug_enabled = _should_debug(debug_traces)
    G = rx.generators.complete_graph(n_nodes)
    # Prune skeleton as compile_and_ground would
    dep_undirected = {tuple(sorted(p)) for p in dep_facts}
    forbidden_edges = {
        pair
        for pair in indep_facts
        if tuple(sorted(pair)) not in dep_undirected
    }
    # Remove both orientations to match the undirected skeleton
    forbidden_edges |= {(b, a) for (a, b) in forbidden_edges}
    if debug_enabled:
        logger.debug("[increm] Pruning skeleton for pairs: %s", forbidden_edges)
    # Do not remove edges; keep full skeleton so paths remain available after fact removals.
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
        nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1, len(path)-1)]
        nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
        rules.append(f"ap({X},{Y},{qsym},S) :- {qsym}, active_pair({X},{Y}), {nbs_str} not in({X},S), not in({Y},S), set(S).")
        new_paths.append(tuple(path))
    if rules and part_id is not None:
        part_name = f"incr_{part_id}"
        logger.info("Grounding incremental part for pair (%s,%s) with %s rules.", X, Y, len(rules))
        try:
            ctl.add(part_name, [], "\n".join(rules))
            ctl.ground([(part_name, [])])
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
            logger.exception("[increm] failed to ground part %s with rules:\n%s", part_name, "\n".join(rules))
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
    debug_dump_path = debug_dump_path or os.environ.get("CAUSALABA_DUMP")
    ext_values: dict[Function, bool | None] = {}
    observer = _DebugObserver()
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

    # Stable order: primary by score (if any), otherwise preserve file order
    # Store original index to keep deterministic removal when scores are NaN/unknown
    facts = [(i,)+f for i, f in enumerate(facts)]
    facts.sort(key=lambda x: (not (isinstance(x[6], float) and x[6]==x[6]), x[6] if isinstance(x[6], float) else float('-inf'), x[0]), reverse=True)
    facts = [f[1:] for f in facts]
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
    # # control_args = [f'-t {threads}']
    control_args = []
    if cycle_length is not None:
        control_args += [f"-c l_cyc={int(cycle_length)}"]
    if collider_tree_depth is not None:
        control_args += [f"-c l_b={int(collider_tree_depth)}"]
    ctl = Control(control_args)
    try:
        ctl.register_observer(observer)
        observer.set_ctl(ctl)
    except Exception:
        logger.exception("Failed to register debug observer")
    ctl.configuration.solve.parallel_mode = 1
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

    ctl.add("specific", [], "indep(X,Y,S) :- active_pair(X,Y), ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    ctl.add("specific", [], "dep(X,Y,S) :- active_pair(X,Y), ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    # Make indep facts dynamically forbid edges while the external is active (retractable when external is disabled)
    if skeleton_rules_reduction:
        ctl.add("specific", [], ":- active_pair(X,Y), ext_indep(X,Y,S), edge(X,Y), var(X), var(Y), set(S), X!=Y.")

    # Build skeleton and add path rules without adding ":- edge(X,Y)." constraints
    logger.info("   Adding Specific Rules...")
    n_p = 0
    G = rx.generators.complete_graph(n_nodes)
    forbidden_edges = set()
    if skeleton_rules_reduction:
        dep_undirected = {tuple(sorted(p)) for p in dep_facts}
        forbidden_edges = {
            pair
            for pair in indep_facts
            if tuple(sorted(pair)) not in dep_undirected
        }
        forbidden_edges |= {(b, a) for (a, b) in forbidden_edges}
        if debug_enabled:
            logger.debug("[skeleton] Forbidden edges: %s", forbidden_edges)
        # Keep the full skeleton so that paths remain available if facts are removed.
        # Independence facts will still forbid edges via constraints while active.
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
    # node_pairs = tuple(combinations(range(n_nodes), 2))
    use_bounded_nb = bounded_encoding_active and collider_tree_depth is not None and collider_tree_depth > 0
    # Track active pairs as externals so we can deactivate path rules if a pair is fully removed
    for (X, Y) in node_pairs:
        ctl.add("facts", [], f"#external active_pair({X},{Y}).")
        _ACTIVE_PAIR_STATE[(X, Y)] = True
    for (X, Y) in node_pairs:
        if debug_enabled:
            logger.debug("[skeleton] Adding path rules for pair (%s,%s)", X, Y)
        path_found = False
        for path in _iter_paths_with_cutoff(G, X, Y, max_path_length):
            if debug_enabled:
                logger.debug("[skeleton] Found path for pair (%s,%s): %s", X, Y, path)
            path_found = True
            n_p += 1
            path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
            ctl.add("specific", [], f"p{n_p} :- active_pair({X},{Y}), {','.join(path_edges)}.")
            if debug_enabled:
                logger.debug("[rules] added specific p%s :- %s.", n_p, ",".join(path_edges))
            nb_pred = 'nb_b' if use_bounded_nb else 'nb'
            nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1,len(path)-1)]
            nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
            ctl.add("specific", [], f"ap({X},{Y},p{n_p},S) :- p{n_p}, active_pair({X},{Y}), {nbs_str} not in({X},S), not in({Y},S), set(S).")
            _PAIR_PATHS_ADDED.setdefault((X, Y), set()).add(tuple(path))

        # Enforce dep/indep facts exactly as in the non-incremental solver:
        # dep facts require an active path; indep facts forbid it.
        # Match the original solver semantics:
        # - ext_dep enforces dependence by forbidding absence of active paths (indep :- not ap)
        # - ext_indep enforces independence by requiring active paths (dep :- ap)
        if (X, Y) in dep_facts:
            if pre_grounding:
                for S in dep_facts[(X, Y)]:
                    if max_conditioning_size is not None and len(S) > max_conditioning_size:
                        continue
                    s_str = 'empty' if not S else 's' + 'y'.join([str(i) for i in S])
                    ext_premise = f"active_pair({X},{Y}), " + (f"ext_dep({X},{Y},{s_str}), " if ext_flag else "")
                    ctl.add("specific", [], f"indep({X},{Y},{s_str}) :- {ext_premise}not ap_exists({X},{Y},{s_str}).")
                    ctl.add("specific", [], f":- {ext_premise}not ap_exists({X},{Y},{s_str}).")
                    ctl.add("specific", [], f":- active_pair({X},{Y}), ext_dep({X},{Y},{s_str}), not edge({X},{Y}), not edge({Y},{X}).")
            else:
                ext_premise = f"active_pair({X},{Y}), " + (f"ext_dep({X},{Y},S), " if ext_flag else "")
                ctl.add("specific", [], f"indep({X},{Y},S) :- {ext_premise}not ap_exists({X},{Y},S), set(S).")
        if (X, Y) in indep_facts:
            ext_premise = f"active_pair({X},{Y}), " + (f"ext_indep({X},{Y},S), " if ext_flag else "")
            ctl.add("specific", [], f"dep({X},{Y},S) :- {ext_premise}ap_exists({X},{Y},S), set(S).")

    ctl.add("specific", [], "ap_exists(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), set(S).")

    # Show directives
    if 'arrow' in show:
        ctl.add("base", [], "#show arrow/2.")
    if 'indep' in show:
        ctl.add("base", [], "#show indep/3.")
    if 'dep' in show:
        ctl.add("base", [], "#show dep/3.")
    if 'ap' in show:
        ctl.add("base", [], "#show ap/4.")

    # Ground the full base
    logger.info("   Grounding full base program...")
    ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes-1)])])
    # Activate all pair guards by default
    for (ax, ay), state in _ACTIVE_PAIR_STATE.items():
        ctl.assign_external(Function("active_pair", [Number(ax), Number(ay)]), True)
        ext_values[Function("active_pair", [Number(ax), Number(ay)])] = True

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
        with ctl.solve(yield_=True) as handle:
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
        last_syms = None
        logger.info("   Solving...")
        with ctl.solve(yield_=True) as handle:
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

            curr = []
            with ctl.solve(yield_=True) as handle:
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
        ctl.assign_external(ext_sym, False)
        try:
            ext_values[ext_sym] = False
        except Exception:
            pass

        if debug_enabled:
            logger.debug("[in-remove] existing: %s", existing)
            logger.debug(
                "[in-remove] indep facts all for pair (%s,%s): %s",
                rem_X,
                rem_Y,
                indep_facts.get((rem_X, rem_Y), set()),
            )
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
            # Always refresh path rules and constraints for all pairs after each removal.
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
                    collider_tree_depth=collider_tree_depth,
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
        if n_models == 0 and debug_dump_path and remove_n == 1:
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
    return [models, False]
