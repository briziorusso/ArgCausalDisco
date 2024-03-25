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

import logging
from causalaba import CausalABA
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm.auto import tqdm
from utils import *

def ABAPC(data, seed=2024, scenario="ABAPC", alpha=0.05, indep_test='fisherz', base_pct=1, base_location="results"):
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
    cg = pc(data=data, alpha=alpha, indep_test=indep_test, uc_rule=3, uc_priority=2, show_progress=False, verbose=False)
    ## Extract facts from PC
    facts = []
    for X,Y in combinations(range(n_nodes), 2):
        test_PC = [t for t in cg.sepset[X,Y]]
        for S, p in test_PC:
            dep_type_PC = "indep" if p > alpha else "dep" 
            I = initial_strength(p, len(S), alpha, 0.5, n_nodes)
            s_str = 'empty' if len(S)==0 else 's'+'y'.join([str(i) for i in S])
            facts.append((X,S,Y,dep_type_PC, f"{dep_type_PC}({X},{Y},{s_str}).", I))

    ### Save external statements
    with open(facts_location, "w") as f:
        for s in facts:
            f.write(f"#external ext_{s[4]}\n")
    ### Save weak constraints
    with open(facts_location_wc, "w") as f:
        for s in facts:
            f.write(f":~ {s[4]} [-{int(s[5]*1000)}]\n")
    ### Save inner strengths
    with open(facts_location_I, "w") as f:
        for s in facts:
            f.write(f"{s[4]} I={s[5]}, NA\n")
    
    set_of_model_sets = []
    model_sets, multiple_solutions = CausalABA(n_nodes, facts_location, weak_constraints=True, 
                                                fact_pct=base_pct, search_for_models='first',
                                                opt_mode='optN', print_models=False, show=['arrow'])

    if multiple_solutions:
        for model in model_sets:
            models, MECs = set_of_models_to_set_of_graphs(model, n_nodes)
            set_of_model_sets.append(models)
    else:
        models, MECs = set_of_models_to_set_of_graphs(model_sets, n_nodes)

    if len(set_of_model_sets) > 0:
        logging.info(f"Number of solutions found: {len(set_of_model_sets)}")
    
    model_ranking = []
    for n, model in tqdm(enumerate(models), desc="Models from ABAPC"):
        ## derive B_est from the model
        B_est = np.zeros((n_nodes, n_nodes))
        for edge in model:
            B_est[edge[0], edge[1]] = 1
        logging.debug("DAG from d-ABA")
        logging.debug(B_est)
        G_est = nx.DiGraph(pd.DataFrame(B_est, columns=[f"X{i+1}" for i in range(B_est.shape[1])], index=[f"X{i+1}" for i in range(B_est.shape[1])]))
        logging.debug(G_est.edges)

        est_seplist = find_all_d_separations_sets(G_est, verbose=False)
        est_ind_statements = []
        est_I = 0
        for test in est_seplist:
            X, S, Y, dep_type = extract_test_elements_from_symbol(test)
            test_PC = [t for t in cg.sepset[X,Y] if set(t[0])==S]
            if len(test_PC)==1:
                p = test_PC[0][1]
                dep_type_PC = "indep" if p > alpha else "dep" 
                I = initial_strength(p, len(S), alpha, 0.5, n_nodes)
                if dep_type == dep_type_PC:
                    est_ind_statements.append((X,S,Y,dep_type, test, I))
                    est_I += I
                else:
                    est_ind_statements.append((X,S,Y,dep_type, test, 1-I))
                    est_I += 1-I
        model_ranking.append((n, est_I))

    ## Select the best model
    best_model = sorted(model_ranking, key=lambda x: x[1], reverse=True)
    B_est = np.zeros((n_nodes, n_nodes))
    for edge in list(models)[best_model[0][0]]:
        B_est[edge[0], edge[1]] = 1

    logging.info(f"Best model by I:")
    logging.info(B_est)

    return B_est

# data = np.zeros((100, 5))
# start = datetime.now()
# ABAPC(data)
# logging.info(f"Total time={str(datetime.now()-start)}")