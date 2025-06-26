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

import os,gc,sys
import logging
import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from itertools import combinations
from cd_algorithms.PC import pc
from causalaba import CausalABA
# sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from utils.helpers import logger_setup, random_stability
from utils.graph_utils import initial_strength, set_of_models_to_set_of_graphs

def ABAPC(data, 
          seed=2024, alpha=0.05, indep_test='fisherz',
          stable=True, conservative=True,
          base_fact_pct=1.0, set_indep_facts=False, 
          scenario="ABAPC", base_location="results",
          out_mode="opt" , print_models=False,
          sepsets = None, smoothing_k=0):
    """
    Args:
    data: np.array
        The dataset to be used for the PC algorithm
    seed: int
        The seed to be used for the random number generator
    alpha: float
        The significance level to be used for the PC algorithm
    indep_test: str
        The independence test to be used for the PC algorithm
    stable: bool
        Whether to use the stable version of the PC algorithm
    conservative: bool
        Whether to use the conservative version of the PC algorithm
    base_fact_pct: float
        The percentage of facts to be used for the ABAPC algorithm
    set_indep_facts: bool
        Whether to set the independence facts in the ABAPC algorithm
    scenario: str
        The scenario to be used for the ABAPC algorithm
    base_location: str
        The base location to save the results of the ABAPC algorithm
    out_mode: str
        The output mode to be used for the ABAPC algorithm
    """
    facts_location = f"{base_location}/{scenario}/facts.lp"
    facts_location_I = f"{base_location}/{scenario}/facts_I.lp"
    facts_location_wc = f"{base_location}/{scenario}/facts_wc.lp"
    ## create folder if it does not exist
    if not os.path.exists(f"{base_location}/{scenario}"):
        os.makedirs(f"{base_location}/{scenario}")
    logger_setup(f"{base_location}/{scenario}/log.log")
    logging.info(f"===============Running {scenario}===============")
    n_nodes = data.shape[1]
    random_stability(seed)
    uc_rule = 5 if conservative else 0
    if sepsets == None:
        logging.info("Running PC algorithm")
        ## Run PC algorithm
        cg = pc(data=data, alpha=alpha, indep_test=indep_test, uc_rule=uc_rule, stable=stable, show_progress=False, verbose=False)
        sepsets = cg.sepset
    else:
        logging.info("Using provided cg")
    ## Extract facts from PC
    facts = set()
    for X,Y in combinations(range(n_nodes), 2):
        test_PC = [t for t in sepsets[X,Y]]
        for S, p in test_PC:
            dep_type_PC = "indep" if p > alpha else "dep" 
            I = initial_strength(p, len(S), alpha, 0.5, n_nodes, smoothing_k=smoothing_k)
            s_str = 'empty' if len(S)==0 else 's'+'y'.join([str(i) for i in S])
            facts.add((X,S,Y,dep_type_PC, f"{dep_type_PC}({X},{Y},{s_str}).", I))

    ### Save external statements
    with open(facts_location, "w") as f:
        for n, s in enumerate(facts):
            if n/len(facts) <= base_fact_pct:
                f.write(f"#external ext_{s[4]}\n")
    ### Save weak constraints
    with open(facts_location_wc, "w") as f:
        for n, s in enumerate(facts):
            if n/len(facts) <= base_fact_pct:
                f.write(f":~ ext_{s[4]} [-{int(s[5]*1000)}]\n")
    ### Save inner strengths
    with open(facts_location_I, "w") as f:
        for n, s in enumerate(facts):
            if n/len(facts) <= base_fact_pct:
                f.write(f"ext_{s[4]} I={s[5]}, NA\n")

    set_of_model_sets = []
    model_sets, multiple_solutions = CausalABA(n_nodes, facts_location, weak_constraints=True, skeleton_rules_reduction=True,
                                                fact_pct=base_fact_pct, search_for_models='first',
                                                opt_mode='optN', print_models=print_models, set_indep_facts=set_indep_facts)

    if multiple_solutions:
        for model in model_sets:
            models, MECs = set_of_models_to_set_of_graphs(model, n_nodes, False)
            set_of_model_sets.append(models)
    else:
        models, MECs = set_of_models_to_set_of_graphs(model_sets, n_nodes, False)

    if len(set_of_model_sets) > 0:
        logging.info(f"Number of solutions found: {len(set_of_model_sets)}")
    
    model_ranking = []
    if len(models) > 50000:
        logging.info("Pick the first 50,000 models for I calculation")
        models = set(list(models)[:50000]) ## Limit the number of models to 30,000
    for n, model in tqdm(enumerate(models), desc="Models from ABAPC"):
        ## derive B_est from the model
        B_est = np.zeros((n_nodes, n_nodes))
        for edge in model:
            B_est[edge[0], edge[1]] = 1
        logging.debug("DAG from d-ABA")
        logging.debug(B_est)
        G_est = nx.DiGraph(pd.DataFrame(B_est, columns=[f"X{i+1}" for i in range(B_est.shape[1])], index=[f"X{i+1}" for i in range(B_est.shape[1])]))
        logging.debug(G_est.edges)
        est_I = 0
        for x,y in combinations(range(n_nodes), 2):
            I_from_data = list(set(sepsets[x,y]))
            for s,p in I_from_data:
                PC_dep_type = 'indep' if p > alpha else 'dep'
                s_text = [f"X{r+1}" for r in s]
                dep_type = 'indep' if nx.algorithms.d_separated(G_est, {f"X{x+1}"}, {f"X{y+1}"}, set(s_text)) else 'dep'
                I = initial_strength(p, len(s), alpha, 0.5, n_nodes, smoothing_k=smoothing_k)
                if dep_type != PC_dep_type:
                    est_I += -I
                else:
                    est_I += I

        if out_mode == "opt":
            best_model = [(n, est_I)] if n == 0 else best_model if best_model[0][1] > est_I else [(n, est_I)]
        elif out_mode == "optN":
            model_ranking.append((n, est_I))
            ## Select the best model
            best_model = sorted(model_ranking, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError("out_mode must be either 'opt' or 'optN'")

    B_est = np.zeros((n_nodes, n_nodes))
    for edge in list(models)[best_model[0][0]]:
        B_est[edge[0], edge[1]] = 1

    logging.info(f"Best model by I:")
    logging.info(B_est)

    if out_mode == "opt":
        del models, model_ranking, model_sets, MECs, facts, I_from_data
        gc.collect()
        return B_est
    elif out_mode == "optN":
        return models, best_model
    else:
        raise ValueError("out_mode must be either 'opt' or 'optN'")
    

# data = np.zeros((100, 5))
# start = datetime.now()
# ABAPC(data)
# logging.info(f"Total time={str(datetime.now()-start)}")