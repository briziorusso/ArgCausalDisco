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
from utils import *

def randomG_PC_facts(n_nodes, edge_per_node=2, graph_type="ER", seed=2024, mec_check=True):
    scenario = "randomG_PC_facts"
    alpha = 0.05
    base_pct = 0.5
    output_name = f"{scenario}_{n_nodes}_{edge_per_node}_{graph_type}_{seed}"
    facts_location = f"encodings/test_lps/{output_name}.lp"
    facts_location_I = f"encodings/test_lps/{output_name}_I.lp"
    facts_location_wc = f"encodings/test_lps/{output_name}_wc.lp"
    logger_setup(output_name)
    logging.info(f"===============Running {scenario}===============")
    logging.info(f"n_nodes={n_nodes}, edge_per_node={edge_per_node}, graph_type={graph_type}, seed={seed}")
    s0 = int(n_nodes*edge_per_node)
    if s0 > int(n_nodes*(n_nodes-1)/2):
        logging.info(f'{s0} is too many edges, setting s0 to the max:', int(n_nodes*(n_nodes-1)/2))
        s0 = int(n_nodes*(n_nodes-1)/2)
    random_stability(2024)
    
    B_true = simulate_dag(d=n_nodes, s0=s0, graph_type=graph_type)
    logging.debug("True DAG")
    logging.debug(B_true)
    G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))    
    logging.debug(G_true.edges)
    
    logging.debug("True CPDAG")
    logging.debug(dag2cpdag(B_true))

    inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
    G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)
    expected = frozenset(set(G_true1.edges()))

    true_seplist = find_all_d_separations_sets(G_true, verbose=False)

    cg = simulate_data_and_run_PC(G_true, alpha)
    CP_est_PC = (cg.G.graph != 0).astype(int)#from_causallearn_to_cpdag(cg)
    logging.debug("CPDAG from PC")
    logging.debug(CP_est_PC)
    ## undirected edges are not considered in DAG metrics
    B_est_PC = (cg.G.graph > 0).astype(int)
    logging.debug("DAG from PC - removing undirected edges")
    logging.debug(B_est_PC)
    ## calculate metrics for DAG
    metrics = MetricsDAG(B_est_PC, B_true).metrics
    ## calculate metrics for CPDAG
    metrics_cp = MetricsDAG(CP_est_PC, B_true).metrics
    ## Log metrics
    logging.info(f"Metrics for ShapleyPC (DAG):")
    logging.info(metrics)
    logging.info(f"Metrics for ShapleyPC (CPDAG):")
    logging.info(metrics_cp)

    ## Extract facts from PC
    facts = []
    count_wrong = 0
    for test in true_seplist:
        X, S, Y, dep_type = extract_test_elements_from_symbol(test)

        test_PC = [t for t in cg.sepset[X,Y] if set(t[0])==S]
        if len(test_PC)==1:
            p = test_PC[0][1]
            dep_type_PC = "indep" if p > alpha else "dep" 
            I = initial_strength(p, len(S), alpha, 0.5, n_nodes)
            if dep_type == dep_type_PC:
                facts.append((X,S,Y,dep_type_PC, test, I, dep_type == dep_type_PC))
            elif dep_type == "indep":
                count_wrong += 1
                facts.append((X,S,Y,dep_type_PC, test.replace("indep", "dep"), I, dep_type == dep_type_PC))
            elif dep_type == "dep":
                count_wrong += 1
                facts.append((X,S,Y,dep_type_PC, test.replace("dep", "indep"), I, dep_type == dep_type_PC))
    
    logging.info(f"Number of total independence statements: {len(true_seplist)}")
    logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
    logging.info(f"Number of wrong facts: {count_wrong} ({count_wrong/len(facts)*100:.2f}%)")

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
            f.write(f"{s[4]} I={s[5]}, {s[6]}\n")
    
    set_of_model_sets = []
    model_sets, multiple_solutions = CausalABA(n_nodes, facts_location, weak_constraints=True, 
                                                fact_pct=base_pct, search_for_models='first',
                                                opt_mode='opt', print_models=False)

    if multiple_solutions:
        for model in model_sets:
            models, MECs = set_of_models_to_set_of_graphs(model, n_nodes, mec_check)
            set_of_model_sets.append(models)
    else:
        models, MECs = set_of_models_to_set_of_graphs(model_sets, n_nodes, mec_check)

    if len(set_of_model_sets) > 0:
        logging.info(f"Number of solutions found: {len(set_of_model_sets)}")
        count_right = 0
        for model_sets in set_of_model_sets:
            if expected in model_sets:
                count_right += 1
                logging.debug(f"Models set: {model_sets}")
        logging.info(f"Number of right solutions found: {count_right}")
    
    ## derive B_est from model_sets
    for model in models:
        B_est = np.zeros((n_nodes, n_nodes))
        for edge in model:
            B_est[edge[0], edge[1]] = 1
        logging.debug("DAG from d-ABA")
        logging.debug(B_est)
        G_est = nx.DiGraph(pd.DataFrame(B_est, columns=[f"X{i+1}" for i in range(B_est.shape[1])], index=[f"X{i+1}" for i in range(B_est.shape[1])]))
        logging.debug(G_est.edges)
        ## calculate metrics
        metrics = MetricsDAG(B_est, B_true).metrics
        ## Log metrics
        logging.info(f"Metrics for DAG:")
        logging.info(metrics)

        CP_est = dag2cpdag(B_est)
        logging.debug("CPDAG from d-ABA")
        logging.debug(CP_est)
        ## calculate metrics for CPDAG
        metrics_cp = MetricsDAG(CP_est, B_true).metrics
        ## Log metrics
        logging.info(f"Metrics for CPDAG:")
        logging.info(metrics_cp)
    ## Save results
    # save_results(output_name, G_true, G_est, models, MECs, metrics, metrics_cp)

start = datetime.now()
randomG_PC_facts(4, 1, "ER", 2024)
logging.info(f"Total time={str(datetime.now()-start)}")