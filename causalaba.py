import os
import logging
from clingo.control import Control
import networkx as nx
import numpy as np
from itertools import combinations
from datetime import datetime
from utils import powerset, extract_test_elements_from_symbol

def CausalABA(num_of_nodes:int, facts_location:str=None, show:list=['arrow'], print_models:bool=True, verbose:bool=False)->list:     
    """
    CausalABA, a function that takes in the number of nodes in a graph and a string of facts and returns a list of compatible causal graphs.

    input:
    num_of_nodes: int
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
    ### Add vars
    ctl.add("base", [], f"#const n_vars = {num_of_nodes-1}.")
    logging.debug(f"#const n_vars = {num_of_nodes-1}.")
    ### Add set definition
    for S in powerset(range(num_of_nodes)):
        for s in S:
            ctl.add("base", [], f"in({s},{'s'+'y'.join([str(i) for i in S])}).")
    #         logging.debug(f"in({s},{'s'+'y'.join([str(i) for i in S])}).")

    ### Load main program and facts
    ctl.load("encodings/causalaba.lp")
    if facts_location:
        ctl.load(facts_location)

    ### add nonblocker rules
    logging.info("   Adding Specific Rules...")
    indep_facts = set()
    dep_facts = set()
    if facts_location:
        with open(facts_location, 'r') as file:
            for line in file:
                if 'indep' in line:
                    X, _, Y, _ = extract_test_elements_from_symbol(line.replace("\n",""))
                    indep_facts.add((X,Y))
                elif 'dep' in line and 'in' not in line:
                    X, _, Y, _ = extract_test_elements_from_symbol(line.replace("\n",""))
                    dep_facts.add((X,Y))

    ### Active paths rules
    n_p = 0
    for (X,Y) in dep_facts:
        G = nx.complete_graph(num_of_nodes)
        paths = nx.all_simple_paths(G, source=X, target=Y)
        ### make the list a matrix
        logging.debug(f"Paths from {X} to {Y}: {len(list(paths))}")
        indep_rule_body = []
        for path in paths:
            broken_path = False
            for idx in range(len(path)-1):
                idx1 = path[idx]
                idx2 = path[idx+1]
                if (idx1,idx2) in indep_facts:
                    broken_path = True
                    break
            if not broken_path:
                n_p += 1
                path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
                nbs = [f"nb({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1,len(path)-1)]
                nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
                ctl.add("base", [], f"p{n_p} :- {','.join(path_edges)}.")
                logging.debug(f"p{n_p} :- {','.join(path_edges)}.")
                ctl.add("base", [], f"ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")
                logging.debug(f"ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")
                indep_rule_body.append(f" not ap({X},{Y},p{n_p},S)")
                # ctl.add("base", [], f"dep(X,Y,S) :- ap(X,Y,p{n_p},S), var(X), var(Y), X!=Y, not in({X},S), not in({Y},S), set(S).")
                # logging.debug(f"dep(X,Y,S) :- ap(X,Y,p{n_p},S), var(X), var(Y), X!=Y, not in({X},S), not in({Y},S), set(S).")
        ctl.add("base", [], f"dep(X,Y,S):- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in({X},S), not in({Y},S), set(S).")
        logging.debug(f"dep(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in({X},S), not in({Y},S), set(S).")

        indep_rule = f"indep({X},{Y},S) :- {','.join(indep_rule_body)}, not in({X},S), not in({Y},S), set(S)."
        ctl.add("base", [], indep_rule)
        logging.debug(indep_rule)

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
        ctl.add("base", [], "#show ap/3.")
        ctl.add("base", [], "#show ap/4.")
        ctl.add("base", [], "#show ap/5.")
    if 'dpath' in show:
        ctl.add("base", [], "#show dpath/2.")


    ### Ground & Solve
    logging.info("   Grounding...")
    start_ground = datetime.now()
    ctl.ground([("base", [])])
    logging.info(f"   Grounding time: {str(datetime.now()-start_ground)}")
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

    return models

# CausalABA(3, "outputs/test_facts.lp", False)