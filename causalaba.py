import os
import logging
from clingo.control import Control
import networkx as nx
from itertools import combinations
from datetime import datetime
from utils import powerset, format_time

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
    ### Add vars
    ctl.add("base", [], f"var(0..{num_of_nodes-1}).")
    logging.debug(f"var(0..{num_of_nodes-1}).")
    ### Add set definition
    for S in powerset(range(num_of_nodes)):
        for s in S:
            ctl.add("base", [], f"in({s},{'s'+'y'.join([str(i) for i in S])}).")

    ### Load main program and facts
    ctl.load("encodings/causalaba.lp")
    if facts_location:
        ctl.load(facts_location)

    ### add nonblocker rules
    logging.info("   Adding Specific Rules...")
    for Z in range(num_of_nodes):
        ctl.add("base", [], f"nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in({Z},S), collider_desc({Z},N,X,Y).")
        logging.debug(f"nb(N,X,Y,S) :- not in(N,S), var(N), var(X), var(Y), N!=X, N!=Y, X!=Y, in({Z},S), collider_desc({Z},N,X,Y).")
    ### Active paths rules
    for path_len in range(2,num_of_nodes+1):
        if path_len>2:
            Xs = [f"X{idx}" for idx in range(path_len)]
            vars = [f"var({X})" for X in Xs]
            edges = [f"edge(X{idx},X{idx+1})" for idx in range(len(Xs)-1)]
            nbs = [f"nb(X{idx},X{idx-1},X{idx+1},S)" for idx in range(1,len(Xs)-1)]
            uneq_const = []
            for comb in combinations(Xs, 2):
                uneq_const.append(f"{comb[0]}!={comb[1]}")
            if len(nbs)>0:
                body = f"{','.join(edges)}, {','.join(vars)}, {','.join(uneq_const)}, {','.join(nbs)}"
            else:
                body = f"{','.join(edges)}, {','.join(vars)}, {','.join(uneq_const)}"
            ### Active paths
            ctl.add("base", [], f"ap({','.join(Xs)},S) :- {body}, not in({Xs[0]},S), not in({Xs[-1]},S), set(S).")
            logging.debug(f"ap({','.join(Xs)},S) :- {body}, not in({Xs[0]},S), not in({Xs[-1]},S), set(S).") 
    ### add indep rules
    for (X,Y) in combinations(range(num_of_nodes),2):
        G = nx.complete_graph(num_of_nodes)
        paths = nx.all_simple_paths(G, source=X, target=Y)
        indep_rule_body = []
        for path in paths:
            indep_rule_body.append(f" not ap({','.join([str(p) for p in path])},S)")
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
    ctl.ground([("base", [])])
    ctl.configuration.solve.models="0"
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