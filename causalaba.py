import os
import logging
from clingo.control import Control
from clingo import Function, Number, String
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from itertools import combinations
from datetime import datetime
from utils import powerset, extract_test_elements_from_symbol

def CausalABA(n_nodes:int, facts_location:str=None, show:list=['arrow'], search_for_models:bool=False, print_models:bool=True, verbose:bool=False)->list:     
    """
    CausalABA, a function that takes in the number of nodes in a graph and a string of facts and returns a list of compatible causal graphs.

    input:
    n_nodes: int
    facts: str
    verbose: bool

    return: list
    
    This function takes in the number of nodes in a graph and a string of facts and returns a list of compatible causal graphs.
    The CausalABA program is a logic program that encodes the causal structure of a graph.
    The program is written in the Answer Set Programming (ASP) language.

    The program is based on the following paper:
    
    """

    logging.info(f"Running CausalABA")
    ### Create Control
    ctl = Control(['-t %d' % os.cpu_count()])
    ctl.configuration.solve.parallel_mode = os.cpu_count()
    ctl.configuration.solve.models="0"
    ctl.configuration.solver.seed="2024"

    ### Add set definition
    for S in powerset(range(n_nodes)):
        for s in S:
            ctl.add("specific", [], f"in({s},{'s'+'y'.join([str(i) for i in S])}).")
    #         logging.debug(f"in({s},{'s'+'y'.join([str(i) for i in S])}).")

    ### Load main program and facts
    ctl.load("encodings/causalaba.lp")
    if facts_location:
        ctl.load(facts_location)

    ### add nonblocker rules
    logging.info("   Adding Specific Rules...")
    indep_facts = set()
    dep_facts = set()
    facts = []
    if facts_location:
        logging.debug(f"   Loading facts from {facts_location}")
        with open(facts_location, 'r') as file:
            for line in file:
                if "dep" in line:
                    ext_fact = True
                    line = line.replace("#external ","")
                    X, S, Y, dep_type = extract_test_elements_from_symbol(line.replace("\n",""))
                    facts.append((X,S,Y,dep_type, line.replace("\n","")))
                    if 'indep' in line:
                        indep_facts.add((X,Y))
                    elif 'dep' in line and 'in' not in line:
                        dep_facts.add((X,Y))
        if ext_fact:
            ctl.add("specific", [], "indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
            ctl.add("specific", [], "dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.")

    ### Active paths rules
    n_p = 0
    G = nx.complete_graph(n_nodes)
    for (X,Y) in combinations(range(n_nodes),2):
        paths = nx.all_simple_paths(G, source=X, target=Y)
        ### remove paths that contain an indep fact
        paths_mat = np.array([np.array(list(xi)+[None]*(n_nodes-len(xi))) for xi in paths])
        paths_mat_red = paths_mat[[not any([(paths_mat[i,j],paths_mat[i,j+1]) in indep_facts 
                                            for j in range(n_nodes-1) if paths_mat[i,j] is not None]) \
                                                for i in range(len(paths_mat))]]
        remaining_paths = [list(filter(lambda x: x is not None, paths_mat_red[i])) for i in range(len(paths_mat_red))]
        logging.debug(f"   Paths from {X} to {Y}: {len(paths_mat)}, removing indep: {len(remaining_paths)}")

        indep_rule_body = []
        for path in remaining_paths:
            n_p += 1
            ### build indep rule body
            indep_rule_body.append(f" not ap({X},{Y},p{n_p},S)")

            ### add path rule
            path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
            ctl.add("specific", [], f"p{n_p} :- {','.join(path_edges)}.")
            logging.debug(f"p{n_p} :- {','.join(path_edges)}.")

            ### add active path rule
            nbs = [f"nb({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1,len(path)-1)]
            nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
            ctl.add("specific", [], f"ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")
            logging.debug(f"ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")

        ### add indep rule
        if len(indep_rule_body) > 0 and (X,Y) in dep_facts:
            indep_rule = f"indep({X},{Y},S) :- {','.join(indep_rule_body)}, not in({X},S), not in({Y},S), set(S)."
            ctl.add("specific", [], indep_rule)
            logging.debug(indep_rule)

    ### add dep rule
    ctl.add("specific", [], f"dep(X,Y,S):- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).")
    logging.debug(f"dep(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).")

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


    ### Ground & Solve
    logging.info("   Grounding...")
    start_ground = datetime.now()
    ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes-1)])])
    logging.info(f"   Grounding time: {str(datetime.now()-start_ground)}")
    for fact in facts:
        ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
        # logging.debug(f"   Adding fact {fact[4]}")
    models = []
    count_models = 0
    logging.info("   Solving...")
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            models.append(model.symbols(shown=True))
            if print_models:
                count_models += 1
                logging.info(f"Answer {count_models}: {model}")

    logging.info(f"Number of models: {int(ctl.statistics['summary']['models']['enumerated'])}")
    times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
    logging.info(f"Times: {times}")

    set_of_models = []
    if count_models == 0 and search_for_models:
        facts = sorted(facts, key=lambda x: len(x[1]), reverse=True)
        logging.info(f"Number of subsets to remove: {len(list(powerset(facts)))}")
        for f_to_remove in tqdm(powerset(facts), desc=f"Removing facts"):
            ### remove fact
            logging.debug(f"Removing fact {[f[4] for f in f_to_remove]}")
            for fact in facts:
                ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
                if fact in f_to_remove:
                    ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), False)

            count_models = 0
            models = []
            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    models.append(model.symbols(shown=True))
                    count_models += 1
                    if print_models:
                        logging.info(f"Answer {count_models}: {model}")
            logging.debug(f"   Number of models: {int(ctl.statistics['summary']['models']['enumerated'])}")
            times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
            logging.debug(f"Times: {times}")
            if count_models > 0:
                set_of_models.append(models)
                    # break

    if len(set_of_models) > 0:
        return set_of_models, True

    return models, False

# CausalABA(3, "outputs/test_facts.lp", False)