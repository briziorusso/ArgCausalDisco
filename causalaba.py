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
from clingo.control import Control
from clingo import Function, Number, String
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from itertools import combinations
from datetime import datetime
from collections import defaultdict
from pathlib import Path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.graph_utils import powerset, extract_test_elements_from_symbol

def compile_and_ground(n_nodes:int, facts_location:str="",
                skeleton_rules_reduction:bool=False,
                weak_constraints:bool=False,
                indep_facts:set=set(),
                dep_facts:set=set(),
                opt_mode:str='optN',
                show:list=['arrow']
                )->Control:

    logging.info(f"Compiling the program")
    ### Create Control
    ctl = Control(['-t %d' % os.cpu_count()])
    ctl.configuration.solve.parallel_mode = os.cpu_count()
    ctl.configuration.solve.models="0"
    ctl.configuration.solver.seed="2024"
    ctl.configuration.solve.opt_mode = opt_mode

    ### Add set definition
    for S in powerset(range(n_nodes)):
        for s in S:
            ctl.add("specific", [], f"in({s},{'s'+'y'.join([str(i) for i in S])}).")
            # logging.debug(f"in({s},{'s'+'y'.join([str(i) for i in S])}).")

    ### Load main program and facts
    ctl.load(str(Path(__file__).resolve().parent / 'encodings' / 'causalaba.lp'))
    if facts_location != "":
        ctl.load(facts_location)
    if weak_constraints:
        ctl.load(facts_location.replace(".lp","_wc.lp"))

    ctl.add("specific", [], "indep(X,Y,S) :- ext_indep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    ctl.add("specific", [], "dep(X,Y,S) :- ext_dep(X,Y,S), var(X), var(Y), set(S), X!=Y.")
    ### add nonblocker rules
    logging.info("   Adding Specific Rules...")

    ### Active paths rules
    n_p = 0
    G = nx.complete_graph(n_nodes)
    ### remove paths that contain an indep fact
    if skeleton_rules_reduction:
        G.remove_edges_from(indep_facts)

    for (X,Y) in combinations(range(n_nodes),2):
        for path in nx.all_simple_paths(G, source=X, target=Y):
            n_p += 1
            ### add path rule
            path_edges = [f"edge({path[idx]},{path[idx+1]})" for idx in range(len(path)-1)]
            ctl.add("specific", [], f"p{n_p} :- {','.join(path_edges)}.")
            logging.debug(f"   p{n_p} :- {','.join(path_edges)}.")

            ### add active path rule
            nbs = [f"nb({path[idx]},{path[idx-1]},{path[idx+1]},S)" for idx in range(1,len(path)-1)]
            nbs_str = ','.join(nbs)+"," if len(nbs) > 0 else ""
            ctl.add("specific", [], f"ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")
            logging.debug(f"   ap({X},{Y},p{n_p},S) :- p{n_p}, {nbs_str} not in({X},S), not in({Y},S), set(S).")

        ### add indep rule
        if not skeleton_rules_reduction or (X,Y) in dep_facts or (Y,X) in dep_facts:
            indep_rule = f"indep({X},{Y},S) :- not ap({X},{Y},_,S), not in({X},S), not in({Y},S), set(S)."
            ctl.add("specific", [], indep_rule)
            logging.debug(   indep_rule)

    ### add dep rule
    ctl.add("specific", [], f"dep(X,Y,S):- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).")
    logging.debug(   f"dep(X,Y,S) :- ap(X,Y,_,S), var(X), var(Y), X!=Y, not in(X,S), not in(Y,S), set(S).")

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

    ### Ground
    logging.info("   Grounding...")
    start_ground = datetime.now()
    ctl.ground([("base", []), ("facts", []), ("specific", []), ("main", [Number(n_nodes-1)])])
    logging.info(f"   Grounding time: {str(datetime.now()-start_ground)}")

    return ctl

def CausalABA(n_nodes:int, facts_location:str="", print_models:bool=True,
                skeleton_rules_reduction:bool=False,
                weak_constraints:bool=False,
                fact_pct:float=1.0,
                set_indep_facts:bool=False,
                opt_mode:str='optN',
                search_for_models:str='No', 
                show:list=['arrow']
                )->list:     
    """
    CausalABA, a function that takes in the number of nodes in a graph and a string of facts and returns a list of compatible causal graphs.
    
    """
    logging.info(f"Running CausalABA")

    facts_location_wc = facts_location.replace(".lp","_wc.lp")
    facts_location_I = facts_location.replace(".lp","_I.lp")

    indep_facts = defaultdict(int)
    dep_facts = defaultdict(int)
    facts = []
    if facts_location:
        facts_loc = facts_location.replace(".lp","_I.lp") if weak_constraints else facts_location
        logging.debug(f"   Loading facts from {facts_location}")
        with open(facts_loc, 'r') as file:
            for line in file:
                if "dep" not in line:
                    continue
                line_clean = line.replace("#external ","").replace("\n","")
                if weak_constraints:
                    statement, Is = line_clean.split(" I=")
                    I,truth = Is.split(",")
                    X, S, Y, dep_type = extract_test_elements_from_symbol(statement)
                    facts.append((X,S,Y, dep_type, statement, float(I), truth))
                else:
                    X, S, Y, dep_type = extract_test_elements_from_symbol(line_clean)
                    facts.append((X,S,Y, dep_type, line_clean, np.nan, "unknown"))

                if 'indep' in line_clean:
                    indep_facts[(X,Y)] += 1 
                elif 'dep' in line_clean and 'in' not in line_clean:
                    dep_facts[(X,Y)] += 1

    ctl = compile_and_ground(n_nodes, facts_location, skeleton_rules_reduction,
                weak_constraints, indep_facts, dep_facts, opt_mode, show)

    facts = sorted(facts, key=lambda x: x[5], reverse=True)
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
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                models.append(model.symbols(shown=True))
                if print_models:
                    logging.info(f"Answer {len(models)}: {model}")
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        logging.info(f"Number of models: {n_models}")
        times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
        logging.info(f"Times: {times}")

    elif search_for_models == 'first':
        for fact in facts:
            ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
            logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
        models = []
        logging.info("   Solving...")
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                models.append(model.symbols(shown=True))
                if print_models:
                    logging.info(f"Answer {len(models)}: {model}")
        n_models = int(ctl.statistics['summary']['models']['enumerated'])
        logging.info(f"Number of models: {n_models}")
        times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
        logging.info(f"Times: {times}")
        remove_n = 0
        logging.info(f"Number of facts removed: {remove_n}")

        ## start removing facts if no models are found
        while n_models == 0 and remove_n < len(facts):
            remove_n += 1
            logging.info(f"Number of facts removed: {remove_n}")

            reground = False
            fact_to_remove = facts[-remove_n]
            logging.debug(f"Removing fact {fact_to_remove[4]}")
            if fact_to_remove[3] == "ext_indep":
                indep_facts[(fact_to_remove[0], fact_to_remove[2])] -= 1
                if indep_facts[(fact_to_remove[0], fact_to_remove[2])] == 0:
                    del indep_facts[(fact_to_remove[0], fact_to_remove[2])]
                else:
                    logging.debug(f"   Not removing fact {fact_to_remove[4]} because there are multiple facts with the same X and Y")                            
                # if set_indep_facts:
                #     ctl.assign_external(Function(fact_to_remove[3], [Number(fact_to_remove[0]), Number(fact_to_remove[2]), Function(fact_to_remove[4].replace(').','').split(",")[-1])]), True)
                # else:
                reground = skeleton_rules_reduction
            else:
                dep_facts[(fact_to_remove[0], fact_to_remove[2])] -= 1
                if dep_facts[(fact_to_remove[0], fact_to_remove[2])] == 0:
                    del dep_facts[(fact_to_remove[0], fact_to_remove[2])]
                else:
                    logging.debug(f"   Not removing fact {fact_to_remove[4]} because there are multiple facts with the same X and Y")
            ctl.assign_external(Function(fact_to_remove[3], [Number(fact_to_remove[0]), Number(fact_to_remove[2]), Function(fact_to_remove[4].replace(').','').split(",")[-1])]), None)

            if reground:
                ### Save external statements
                with open(facts_location, "w") as f:
                    for n, s in enumerate(facts[:-remove_n]):
                        f.write(f"#external ext_{s[4]}\n")
                ### Save weak constraints
                with open(facts_location_wc, "w") as f:
                    for n, s in enumerate(facts[:-remove_n]):
                        f.write(f":~ ext_{s[4]} [-{int(s[5]*1000)}]\n")
                ### Save inner strengths
                with open(facts_location_I, "w") as f:
                    for n, s in enumerate(facts[:-remove_n]):
                        f.write(f"ext_{s[4]} I={s[5]}, NA\n")
                logging.info("Recompiling and regrounding...")
                ctl = compile_and_ground(n_nodes, facts_location, skeleton_rules_reduction,
                                weak_constraints, indep_facts, dep_facts, opt_mode, show)
                for fact in facts[:-remove_n]:
                    ctl.assign_external(Function(fact[3], [Number(fact[0]), Number(fact[2]), Function(fact[4].replace(').','').split(",")[-1])]), True)
                    logging.debug(f"   True fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
                for fact in facts[-remove_n:]:
                    logging.debug(f"   False fact: {fact[4]} I={fact[5]}, truth={fact[6]}")
            models = []
            logging.info("   Solving...")
            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    models.append(model.symbols(shown=True))
                    if print_models:
                        logging.info(f"Answer {len(models)}: {model}")
            n_models = int(ctl.statistics['summary']['models']['enumerated'])
            logging.info(f"Number of models: {n_models}")
            times={key: ctl.statistics['summary']['times'][key] for key in ['total','cpu','solve']}
            logging.info(f"Times: {times}")
        
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
            n_models = int(ctl.statistics['summary']['models']['enumerated'])
            
            if n_models > 0:
                if search_for_models == "first_subsets":
                    return models, False
                else:
                    set_of_models.append(models)

        if len(set_of_models) > 0:
            return set_of_models, True

    return models, False

# CausalABA(3, "outputs/test_facts.lp", False)