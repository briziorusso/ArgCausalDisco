"""Copyright 2024 Fabrizio Russo, Department of Computing, Imperial College London

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__author__ = "Fabrizio Russo"
__email__ = "fabrizio@imperial.ac.uk"
__copyright__ = "Copyright (c) 2024 Fabrizio Russo"

import os, sys
import logging
import time
from typing import Any
from clingo.control import Control
from clingo import Function, Number
import rustworkx as rx
import rustworkx.generators as rx_gen
import numpy as np
from tqdm.auto import tqdm
from itertools import combinations
from datetime import datetime
from pathlib import Path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.graph_utils import powerset, extract_test_elements_from_symbol
from utils.prior_knowledge import PriorKnowledge

def _solve_with_timeout(
    ctl: Control,
    *,
    solve_timeout: float | None,
    on_model,
    assumptions=None,
) -> bool:
    """Run clingo solve and call on_model(model) for each model.

    Returns True if finished normally, False if cancelled due to timeout.
    """
    if solve_timeout is None:
        with ctl.solve(yield_=True, assumptions=assumptions or []) as handle:
            for model in handle:
                on_model(model)
        return True

    # Async solve allows us to enforce a wall-time timeout without SIGALRM or
    # killing the whole process.
    handle = ctl.solve(async_=True, on_model=on_model, assumptions=assumptions or [])
    finished = handle.wait(solve_timeout)
    if not finished:
        handle.cancel()
        # Ensure clingo releases resources.
        handle.wait()
        try:
            handle.get()
        except Exception:
            pass
        return False
    # Force completion.
    handle.get()
    return True

def compile_and_ground(n_nodes:int, facts_location:str="",
                skeleton_rules_reduction:bool=False,
                weak_constraints:bool=False,
                indep_facts:dict[tuple[int, int], set[tuple]]=dict(),
                dep_facts:dict[tuple[int, int], set[tuple]]=dict(),
                opt_mode:str='optN',
                out_n:int=0,
                show:list=['arrow'],
                pre_grounding:bool=False,
                ext_flag: bool = False,
                prior_knowledge: PriorKnowledge | None = None,
                # Bounds for rule generation (Definition 4.5)
                max_path_length: int | None = None,
                max_conditioning_size: int | None = None,
                # Additional bounds encoded in ASP
                collider_tree_depth: int | None = None,
                cycle_length: int | None = None,
                dump_specific: str | None = None,
                threads: int | None = None,
                deadline: float | None = None,
                timing_recorder: dict | None = None,
                )->Control:

    logging.debug("Entering compile_and_ground")
    _t0 = time.perf_counter()
    ### Create Control
    cpu_count = min(os.cpu_count() or 1, 64)
    if threads is None:
        threads = cpu_count
    control_args = ['-t %d' % threads]
    # Provide constants for bounded acyclicity and collider-tree depth
    if cycle_length is not None:
        control_args += [f"-c l_cyc={int(cycle_length)}"]
    if collider_tree_depth is not None:
        control_args += [f"-c l_b={int(collider_tree_depth)}"]
    ctl = Control(control_args)
    cfg: Any = ctl.configuration  # clingo config API lacks type hints
    cfg.solve.parallel_mode = str(threads)
    cfg.solve.models = out_n
    cfg.solver.seed = 2024
    cfg.solve.opt_mode = opt_mode
    # ctl.configuration.solve.time_limit = 3600.0  # 1 hour time limit

    # Collect specific rules if dumping is requested
    specific_rules: list[str] | None = [] if dump_specific is not None else None
    def add_specific(rule_str):
        """Helper to add rule and optionally track it"""
        ctl.add("specific", [], rule_str)
        if specific_rules is not None:
            specific_rules.append(rule_str)

    ### Add set definition
    # Enumerate admissible conditioning sets S. If a bound is provided,
    # we restrict to |S| <= max_conditioning_size.
    base_condition_sets = (
        set().union(*indep_facts.values(), *dep_facts.values())
        if skeleton_rules_reduction
        else powerset(range(n_nodes))
    )
    condition_sets = (
        (S for S in base_condition_sets if max_conditioning_size is None or len(S) <= max_conditioning_size)
    )
    for S in condition_sets:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("compile_and_ground exceeded wall-time budget")
        for s in S:
            set_str = f"in({s},{'s' + 'y'.join([str(i) for i in S])})."
            add_specific(set_str)
            logging.debug(f"   {set_str}")

    ### Load main program and facts
    # Decide which encoding to load: base or bounded
    bounded_encoding_active = (cycle_length is not None and cycle_length > 0) or (
        collider_tree_depth is not None and collider_tree_depth > 0
    )
    encoding_file = 'causalaba_bounded.lp' if bounded_encoding_active else 'causalaba.lp'
    ctl.load(str(Path(__file__).resolve().parent / 'encodings' / encoding_file))
    if facts_location != "":
        ctl.load(facts_location)
        if weak_constraints:
            ctl.load(facts_location.replace(".lp","_wc.lp"))

    add_specific("indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    add_specific("dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    ### add nonblocker rules
    logging.info("   Adding Specific Rules...")

    ### Active paths rules
    n_p = 0
    G = rx_gen.complete_graph(n_nodes)
    if skeleton_rules_reduction:
        forbidden_edges = indep_facts.keys()
        G.remove_edges_from(set(G.edge_list()) & forbidden_edges)
        for (X, Y) in forbidden_edges:
            add_specific(f":- edge({X},{Y}).")
    if prior_knowledge is not None:
        # Prune the path-enumeration skeleton using prior knowledge.
        # Keep an undirected edge (u,v) if either orientation is required.
        # Remove it only if BOTH orientations are forbidden and neither is required.
        req_dir = set(prior_knowledge.required)
        req_undirected = {tuple(sorted((a, b))) for (a, b) in req_dir}
        forb_dir = set(prior_knowledge.forbidden)
        # Build a map to count how many orientations are forbidden for a pair
        forb_counts = {}
        for a, b in forb_dir:
            key = tuple(sorted((a, b)))
            forb_counts[key] = forb_counts.get(key, 0) + 1
        to_remove = set()
        for (u, v) in G.edge_list():
            key = tuple(sorted((u, v)))
            if key in req_undirected:
                continue  # keep edges that are required in at least one orientation
            # Remove only if both orientations are forbidden
            if forb_counts.get(key, 0) >= 2:
                to_remove.add((u, v))
        if to_remove:
            G.remove_edges_from(to_remove)
        for (X, Y) in prior_knowledge.forbidden:
            if not skeleton_rules_reduction or ((X, Y) not in forbidden_edges and (Y, X) not in forbidden_edges):
                add_specific(f":- arrow({X},{Y}).")
        for (X, Y) in prior_knowledge.required:
            if not skeleton_rules_reduction or ((X, Y) not in forbidden_edges and (Y, X) not in forbidden_edges):
                add_specific(f"arrow({X},{Y}).")
            else:
                logging.warning(f"Required edge ({X},{Y}) is in the forbidden edges set.")

    node_pairs = tuple(dep_facts | indep_facts if skeleton_rules_reduction else combinations(range(n_nodes),2))
    logging.debug(f"{len(node_pairs) / (n_nodes*(n_nodes-1)/2):.2%} of all node pairs will be considered for active paths.")

    if skeleton_rules_reduction is False:
        pre_grounding = False
    # Helper to iterate paths with an optional cutoff, falling back if needed
    def _iter_paths_with_cutoff(graph, src, dst, cutoff):
        try:
            # rustworkx exposes an optional cutoff argument on all_simple_paths
            return rx.all_simple_paths(graph, src, dst, cutoff=cutoff) if cutoff is not None else rx.all_simple_paths(graph, src, dst)
        except TypeError:
            # Fallback for older versions without cutoff: filter manually
            paths = rx.all_simple_paths(graph, src, dst)
            if cutoff is None:
                return paths
            # Wrap generator to filter by number of edges (|p|)
            def _gen():
                for p in paths:
                    if len(p) - 1 <= cutoff:
                        yield p
            return _gen()

    use_bounded_nb = bounded_encoding_active and collider_tree_depth is not None and collider_tree_depth > 0

    for (X, Y) in node_pairs:
        if deadline is not None and time.perf_counter() > deadline:
            raise TimeoutError("compile_and_ground exceeded wall-time budget")
        for path in _iter_paths_with_cutoff(G, X, Y, max_path_length):
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError("compile_and_ground exceeded wall-time budget")
            n_p += 1
            ### add path rule
            path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
            add_specific(f"p{n_p} :- {','.join(path_edges)}.")
            logging.debug(f"   p{n_p} :- {','.join(path_edges)}.")

            ### add active path rule
            if pre_grounding:
                condition_sets = set()
                if (X,Y) in dep_facts:
                    condition_sets.update(dep_facts[(X,Y)])
                if (X,Y) in indep_facts:
                    condition_sets.update(indep_facts[(X,Y)])
                for S in condition_sets:
                    if max_conditioning_size is not None and len(S) > max_conditioning_size:
                        continue
                    s_str = 'empty' if not S else 's'+'y'.join([str(i) for i in S])
                    nb_pred = 'nb_b' if use_bounded_nb else 'nb'
                    nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},{s_str})" for idx in range(1,len(path)-1)]
                    nbs_str = ", " + ','.join(nbs) if len(nbs) > 0 else ""
                    add_specific(f"ap({X},{Y},p{n_p},{s_str}) :- p{n_p}{nbs_str}.")
                    logging.debug(f"   ap({X},{Y},p{n_p},{s_str}) :- p{n_p}{nbs_str}.")

                    if S in indep_facts.get((X,Y), set()):
                        ext_premise = f"ext_indep({X},{Y},{s_str}), " if ext_flag else ""
                        add_specific(f"dep({X},{Y},{s_str}) :- {ext_premise}ap({X},{Y},p{n_p},{s_str}).")
            else:
                nb_pred = 'nb_b' if use_bounded_nb else 'nb'
                nbs = [f"{nb_pred}({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1,len(path)-1)]
                nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
                add_specific(f"ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")
                logging.debug(f"   ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")

        if (X, Y) in dep_facts:
            if pre_grounding:
                for S in dep_facts[(X, Y)]:
                    if max_conditioning_size is not None and len(S) > max_conditioning_size:
                        continue
                    s_str = 'empty' if not S else 's'+'y'.join([str(i) for i in S])
                    ext_premise = f"ext_dep({X},{Y},{s_str}), " if ext_flag else ""
                    add_specific(f"indep({X},{Y},{s_str}) :- {ext_premise}not ap({X},{Y},_,{s_str}).")
            else:
                ext_premise = f"ext_dep({X},{Y},S), " if ext_flag else ""
                add_specific(f"indep({X},{Y},S) :- {ext_premise}not ap({X},{Y},_,S), set(S).")
        if (X, Y) in indep_facts and pre_grounding is False:
            ext_premise = f"ext_indep({X},{Y},S), " if ext_flag else ""
            add_specific(f"dep({X},{Y},S) :- {ext_premise}ap({X},{Y},_,S), set(S).")

    logging.debug(f"{n_p} active paths added.")

    ### add show statements
    if 'arrow' in show:
        ctl.add("base", [], "#show arrow/2.")
    if 'indep' in show:
        ctl.add("base", [], "#show indep/3.")
    if 'dep' in show:
        ctl.add("base", [], "#show dep/3.")
    if 'collider' in show:
        ctl.add("base", [], "#show collider/3.")
    if 'collider_desc' in show:
        ctl.add("base", [], "#show collider_desc/4.")
    if 'nb' in show:
        ctl.add("base", [], "#show nb/4.")
    if 'ap' in show:
        ctl.add("base", [], "#show ap/4.")
    if 'dpath' in show:
        ctl.add("base", [], "#show dpath/2.")

    ### Dump specific rules to file if requested
    if dump_specific is not None and specific_rules is not None:
        with open(dump_specific, 'w') as f:
            for rule in specific_rules:
                f.write(rule + '\n')
        logging.debug(f"   Dumped {len(specific_rules)} specific rules to {dump_specific}")

    ### Ground
    logging.info("   Grounding...")
    start_ground_dt = datetime.now()
    _tg0 = time.perf_counter()
    ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes-1)])])
    _tg1 = time.perf_counter()
    logging.info(f"   Grounding time: {str(datetime.now()-start_ground_dt)}")

    # Record timing breakdown if requested
    if timing_recorder is not None:
        try:
            timing_recorder['compile_sec_total'] = max(0.0, _tg0 - _t0)
            timing_recorder['ground_sec_total'] = max(0.0, _tg1 - _tg0)
            timing_recorder['total_sec'] = max(0.0, _tg1 - _t0)
        except Exception:
            pass

    return ctl


def CausalABA(n_nodes:int, facts_location:str="", print_models:bool=True,
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
                # Bounds for rule generation (Definition 4.5)
                max_path_length: int | None = None,
                max_conditioning_size: int | None = None,
                # Additional bounds encoded in ASP
                collider_tree_depth: int | None = None,
                cycle_length: int | None = None,
                threads: int | None = None,
                solve_timeout: float | None = None,
                timing_recorder: dict | None = None,
                )->list:
    """
    CausalABA, a function that takes in the number of nodes in a graph and a string of facts and returns a list of compatible causal graphs.
    
    """
    logging.info(f"Running CausalABA")

    # Treat solve_timeout as a wall-clock budget for the whole run (ground+solve).
    deadline = (time.perf_counter() + float(solve_timeout)) if solve_timeout is not None else None

    # Optional timing: track grounding/solving time spent on UNSAT instances
    # during the removal loop (search_for_models='first') before SAT is reached.
    if timing_recorder is not None:
        timing_recorder.setdefault('unsat_ground_sec_total', 0.0)
        timing_recorder.setdefault('unsat_solve_sec_total', 0.0)
        timing_recorder.setdefault('timed_out', False)
        timing_recorder.setdefault('timeout_phase', None)
    
    # (X, Y) -> their condition sets S
    indep_facts: dict[tuple, set[tuple]] = {}
    dep_facts: dict[tuple, set[tuple]] = {}
    facts = []
    ext_flag = False
    if facts_location:
        facts_loc = facts_location.replace(".lp","_I.lp") if weak_constraints else facts_location
        logging.debug(f"   Loading facts from {facts_location}")
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
                    facts.append((X,S,Y, dep_type, line_clean, np.nan, "unknown"))

                assert (X not in S) and (Y not in S), f"X or Y in S: {line_clean}"
                condition_set = tuple(S)

                facts_group = indep_facts if "indep" in line_clean else dep_facts
                if (X,Y) not in facts_group:
                    facts_group[(X,Y)] = set()
                assert condition_set not in facts_group[(X,Y)], f"Redundant external fact: {line_clean}"
                facts_group[(X,Y)].add(condition_set)

    facts = sorted(facts, key=lambda x: x[5], reverse=True)
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
        timing_recorder=timing_recorder,
    )

    if search_for_models == 'No':
        for n, fact in enumerate(facts):
            if fact[3] == "ext_indep" and set_indep_facts:
                ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
                logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
            elif n/len(facts) <= fact_pct:
                ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
                logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
            else:
                ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), False)
                logging.debug(f"   False fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
        models = []
        logging.info("   Solving...")
        def _on_model(model):
            models.append(model.symbols(shown=True))
            if print_models:
                logging.info(f"Answer {len(models)}: {model}")
        finished = False
        remaining_timeout = None
        solve_started = time.perf_counter()
        if deadline is not None:
            remaining_timeout = max(0.0, deadline - solve_started)
            logging.info(f"Initial solve budget (wall clock): {remaining_timeout:.3f}s")
            if remaining_timeout == 0:
                logging.error("Timeout: no time remaining for initial solve.")
            else:
                finished = _solve_with_timeout(
                    ctl,
                    solve_timeout=remaining_timeout,
                    on_model=_on_model,
                )
        else:
            logging.info(f"Initial solve budget (solve_timeout arg): {solve_timeout}")
            finished = _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_on_model)
        solve_ended = time.perf_counter()
        if not finished:
            elapsed = solve_ended - solve_started
            budget = remaining_timeout if remaining_timeout is not None else solve_timeout
            total_budget = solve_timeout if solve_timeout is not None else "unlimited"
            logging.error(f"Solve timed out after {elapsed:.3f}s (budget={budget:.3f}s of {total_budget} total) [phase=initial, mode=No]")
            if timing_recorder is not None:
                timing_recorder['timed_out'] = True
                timing_recorder['timeout_phase'] = 'solve'
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        logging.info(f"Number of models: {n_models}")
        times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
        logging.info(f"Times: {times}")

    elif search_for_models == 'first':
        def _configure_first_witness(_ctl: Control) -> None:
            """Configure clingo to return a quick SAT witness (no enumeration/optimization)."""
            try:
                cfg: Any = _ctl.configuration
                cfg.solve.models = 1
                # Weak constraints do not affect satisfiability; ignore optimization to get a quick witness.
                cfg.solve.opt_mode = 'ignore'
            except Exception:
                pass

        for fact in facts:
            ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
            logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")

        # Ensure the initial solve does not enumerate.
        _configure_first_witness(ctl)
        models = []
        logging.info("   Solving...")
        i_counter = {"i": 0}

        def _on_model_first(model):
            i_counter["i"] += 1
            models.append(model.symbols(shown=True))
            if print_models:
                logging.info(f"Answer {i_counter['i']}: {model}")

        finished = False
        remaining_timeout = None
        solve_started = time.perf_counter()
        if deadline is not None:
            remaining_timeout = max(0.0, deadline - solve_started)
            logging.info(f"Initial solve budget (wall clock): {remaining_timeout:.3f}s")
            if remaining_timeout == 0:
                logging.error("Timeout: no time remaining for initial solve.")
            else:
                finished = _solve_with_timeout(
                    ctl,
                    solve_timeout=remaining_timeout,
                    on_model=_on_model_first,
                )
        else:
            logging.info(f"Initial solve budget (solve_timeout arg): {solve_timeout}")
            finished = _solve_with_timeout(ctl, solve_timeout=solve_timeout, on_model=_on_model_first)
        solve_ended = time.perf_counter()
        if not finished:
            elapsed = solve_ended - solve_started
            budget = remaining_timeout if remaining_timeout is not None else solve_timeout
            total_budget = solve_timeout if solve_timeout is not None else "unlimited"
            logging.error(f"Solve timed out after {elapsed:.3f}s (budget={budget:.3f}s of {total_budget} total) [phase=initial, mode=first]")
            if timing_recorder is not None:
                timing_recorder['timed_out'] = True
                timing_recorder['timeout_phase'] = 'solve'
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        logging.info(f"Number of models: {n_models}")
        times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
        logging.info(f"Times: {times}")

        if timing_recorder is not None and n_models == 0:
            try:
                timing_recorder['unsat_ground_sec_total'] += float(timing_recorder.get('ground_sec_total', 0.0))
                timing_recorder['unsat_solve_sec_total'] += max(0.0, solve_ended - solve_started)
            except Exception:
                pass
        remove_n = 0
        logging.debug(f"Number of facts removed: {remove_n}")

        ## start removing facts if no models are found
        while n_models == 0 and remove_n < len(facts):
            # Check if we've exhausted the deadline before attempting another removal iteration
            if deadline is not None and time.perf_counter() > deadline:
                logging.error(f"Timeout: removal iteration {remove_n} exceeded overall solve_timeout budget.")
                if timing_recorder is not None:
                    timing_recorder['timed_out'] = True
                    timing_recorder['timeout_phase'] = 'solve'
                break

            remove_n += 1
            logging.debug(f"Number of facts removed: {remove_n}")

            iter_ground_sec = 0.0

            reground = False
            fact_to_remove = facts[-remove_n]
            X, S, Y, dep_type, fact_str = fact_to_remove[:5]
            logging.debug(f"Removing fact {fact_str}")

            facts_group = indep_facts if dep_type == "ext_indep" else dep_facts
            facts_group[(X, Y)].remove(tuple(S))
            if not facts_group[(X, Y)]:
                del facts_group[(X, Y)]
                ## regrground only if disable_reground is False, skeleton_rules_reduction is True, and either ext_flag is False or dep_type is "ext_indep"
                reground = (
                    disable_reground is False
                    and skeleton_rules_reduction
                    and (ext_flag is False or dep_type == "ext_indep")
                )
            else:
                logging.debug(f"   Not removing fact {fact_str} because there are multiple facts with the same X and Y")
            ctl.assign_external(Function(dep_type, [Number(X), Number(Y), Function(fact_str.replace(').','').split(",")[-1])]), None)

            if reground:
                ### Save external statements
                logging.info(f"Facts removed: {remove_n} -> Recompiling and regrounding...")
                reground_timing: dict = {}
                # Compute remaining time for the regrounding phase
                reground_deadline = None
                if deadline is not None:
                    reground_remaining = max(0.0, deadline - time.perf_counter())
                    if reground_remaining <= 0:
                        logging.error(f"Timeout: no time remaining for reground in removal iteration {remove_n}.")
                        if timing_recorder is not None:
                            timing_recorder['timed_out'] = True
                            timing_recorder['timeout_phase'] = 'ground'
                        break
                    reground_deadline = time.perf_counter() + reground_remaining

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
                    deadline=reground_deadline,
                    timing_recorder=reground_timing,
                )

                # Ensure post-reground solve does not enumerate.
                _configure_first_witness(ctl)
                iter_ground_sec = float(reground_timing.get('ground_sec_total', 0.0))
                for fact in facts[:-remove_n]:
                    ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
                    logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
                for fact in facts[-remove_n:]:
                    ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), None)
                    logging.debug(f"   False fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
            models = []
            logging.debug("   Solving...")
            i_counter = {"i": 0}

            def _on_model2(model):
                i_counter["i"] += 1
                models.append(model.symbols(shown=True))
                if print_models:
                    logging.info(f"Answer {i_counter['i']}: {model}")

            # Ensure each removal-iteration solve does not enumerate.
            _configure_first_witness(ctl)

            # If we already hit the timeout in the initial solve, stop trying
            # additional removal iterations.
            if solve_timeout is not None and not finished:
                break
            # Use remaining time from deadline instead of original solve_timeout
            remaining_timeout = None
            if deadline is not None:
                remaining_timeout = max(0.0, deadline - time.perf_counter())
                if remaining_timeout == 0:
                    logging.error(f"Timeout: no time remaining for solve in removal iteration {remove_n}.")
                    if timing_recorder is not None:
                        timing_recorder['timed_out'] = True
                        timing_recorder['timeout_phase'] = 'solve'
                    break
                logging.debug(f"Removal iteration {remove_n} solve budget (remaining from {solve_timeout}s total): {remaining_timeout:.3f}s")
            else:
                logging.debug(f"Removal iteration {remove_n} solve budget: {solve_timeout}s")
            t_s0 = time.perf_counter()
            finished = _solve_with_timeout(
                ctl,
                solve_timeout=remaining_timeout if deadline is not None else solve_timeout,
                on_model=_on_model2,
            )
            t_s1 = time.perf_counter()
            if not finished:
                elapsed = t_s1 - t_s0
                budget = remaining_timeout if remaining_timeout is not None else solve_timeout
                total_budget = solve_timeout if solve_timeout is not None else "unlimited"
                logging.error(f"Solve timed out after {elapsed:.3f}s (remaining budget={budget:.3f}s of {total_budget}s total) [phase=removal, iter={remove_n}]")
                if timing_recorder is not None:
                    timing_recorder['timed_out'] = True
                    timing_recorder['timeout_phase'] = 'solve'
            try:
                n_models = int(ctl.statistics['summary']['models']['enumerated'])
                logging.debug(f"Number of models: {n_models}")
                times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
                logging.debug(f"Times: {times}")
            except (KeyError, TypeError, AttributeError) as e:
                logging.warning(f"Could not access solver statistics: {e}")
                n_models = len(models)
                logging.info(f"Number of models: {n_models}")

            if timing_recorder is not None and n_models == 0:
                try:
                    timing_recorder['unsat_ground_sec_total'] += float(iter_ground_sec)
                    timing_recorder['unsat_solve_sec_total'] += max(0.0, t_s1 - t_s0)
                except Exception:
                    pass
        
    elif 'subsets' in search_for_models:
        set_of_models = []
        logging.info(f"Number of subsets to remove: {len(list(powerset(facts)))}")
        for f_to_remove in tqdm(powerset(facts), desc=f"Removing facts"):
            ### remove fact
            logging.debug(f"Removing fact {[f[4] for f in f_to_remove]}")
            for fact in facts:
                ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
                logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
                if fact in f_to_remove:
                    if fact[3] == "ext_indep" and set_indep_facts:
                        continue
                    ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), False)
                    logging.debug(f"   False fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
            
            models = []
            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    models.append(model.symbols(shown=True))
                    if print_models:
                        logging.info(f"Answer {len(models)}: {model}")
            try:
                n_models = int(ctl.statistics['summary']['models']['enumerated'])
            except (KeyError, TypeError, AttributeError) as e:
                logging.warning(f"Could not access solver statistics: {e}")
                n_models = len(models)
            
            if n_models > 0:
                if search_for_models == "first_subsets":
                    return [models, False]
                else:
                    set_of_models.append(models)

        if len(set_of_models) > 0:
            return [set_of_models, True]

    if return_statistics:
        return [models, False, ctl.statistics, remove_n if 'remove_n' in locals() else 0]
    else:
        return [models, False]

# CausalABA(3, "outputs/test_facts.lp", False)
