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

import unittest
import logging
from causalaba import CausalABA
import networkx as nx
import numpy as np
import pandas as pd
from datetime import datetime
from utils import *

class TestCausalABA(unittest.TestCase):

    def three_node_all_graphs(self):
        logger_setup()
        logging.info("===============Running three_node_all_graphs===============")
        n_nodes = 3
        expected = set([
            frozenset(),
            frozenset({(1, 0)}),
            frozenset({(0, 2)}),
            frozenset({(0, 1)}),
            frozenset({(1, 2)}),
            frozenset({(2, 0)}),
            frozenset({(2, 1)}),
            ##chains
            frozenset({(0, 2), (2, 1)}),
            frozenset({(0, 1), (2, 0)}),
            frozenset({(1, 0), (2, 1)}),
            frozenset({(1, 2), (2, 0)}),
            frozenset({(1, 0), (0, 2)}),
            frozenset({(0, 1), (1, 2)}),
            ##colliders
            frozenset({(0, 1), (2, 1)}),
            frozenset({(0, 2), (1, 2)}),
            frozenset({(1, 0), (2, 0)}),
            ##confounders
            frozenset({(2, 0), (2, 1)}),
            frozenset({(1, 0), (1, 2)}),
            frozenset({(0, 1), (0, 2)}),
            ##three arrows configurations
            frozenset({(0, 1), (0, 2), (1, 2)}),
            frozenset({(0, 1), (0, 2), (2, 1)}),
            frozenset({(1, 0), (0, 2), (1, 2)}),
            frozenset({(1, 0), (2, 0), (2, 1)}),
            frozenset({(0, 1), (2, 0), (2, 1)}),
            frozenset({(1, 0), (1, 2), (2, 0)})
        ])
        models, _ = CausalABA(n_nodes)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(model_sets, expected)

    def three_node_graph_empty(self):
        logger_setup()
        logging.info("===============Running three_node_graph_empty===============")
        n_nodes = 3
        expected = set([
            frozenset(),
        ])
        models, _ = CausalABA(n_nodes, "encodings/test_lps/three_node_empty.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(model_sets, expected)

    def collider(self):
        logger_setup()
        logging.info("===============Running collider===============")
        n_nodes = 3
        expected = set([
            frozenset({(0, 2), (1, 2)}),
        ])
        models, _ = CausalABA(n_nodes, "encodings/test_lps/collider.lp", show=["arrow", "indep"])
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))

        self.assertEqual(model_sets, expected)

    def chains_confounder(self):
        logger_setup()
        logging.info("===============Running chains_confounder===============")
        n_nodes = 3
        expected = set([
            ##chains
            frozenset({(1, 2), (2, 0)}),
            frozenset({(0, 2), (2, 1)}),
            ##confounder
            frozenset({(2, 0), (2, 1)})
        ])
        models, _ = CausalABA(n_nodes, "encodings/test_lps/chains_confounder.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))

        self.assertEqual(model_sets, expected)

    def one_edge(self):
        logger_setup()
        logging.info("===============Running one_edge===============")
        n_nodes = 3
        expected = set([
            frozenset({(0, 1)}),
            frozenset({(1, 0)}),
        ])
        models, _ = CausalABA(n_nodes, "encodings/test_lps/one_edge.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))

        self.assertEqual(model_sets, expected)

    def incompatible_Is(self):
        logger_setup()
        logging.info("===============Running incompatible_Is===============")
        n_nodes = 3
        expected = set()
        models, _ = CausalABA(n_nodes, "encodings/test_lps/incompatible_Is.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))       

        self.assertEqual(model_sets, expected)

    def four_node_all_graphs(self):
        logger_setup()
        logging.info("===============Running four_node_graph_full===============")
        n_nodes = 4
        models, _ = CausalABA(n_nodes, print_models=False)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 543)

    def four_node_shapPC_example(self):
        logger_setup()
        scenario = "four_node_shapPC_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  0],
                            [ 0,  0,  1,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)

        expected = set([
            frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        ])

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models, _ = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(set(model_sets), expected)

    def incompatible_chain(self):
        logger_setup()
        scenario = "incompatible_chain"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        logging.info(f"===============Running {scenario}===============")
        n_nodes = 4
        expected = set()

        models, _ = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(set(model_sets), expected)

    def five_node_all_graphs(self):
        logger_setup()
        logging.info("===============Running five_node_all_graphs===============")
        n_nodes = 5
        models, _ = CausalABA(n_nodes, print_models=False) 
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 29281)

    def five_node_colombo_example(self):
        scenario = "five_node_colombo_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  1,  1],
                            [ 0,  0,  1,  0,  1],
                            [ 0,  0,  0,  1,  1],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        expected = {frozenset({(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)})}

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models, _ = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(set(model_sets), expected)

    def five_node_sprinkler_example(self):
        scenario = "five_node_sprinkler_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        expected = frozenset({(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)})

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models, _ = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertIn(expected, model_sets)

    def six_node_all_graphs(self):
        logger_setup()
        logging.info("===============Running six_node_all_graphs===============")
        n_nodes = 6
        models, _ = CausalABA(n_nodes, print_models=False)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 3781503)

    def six_node_example(self):
        scenario = "six_node_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  1,  1,  1],
                            [ 0,  0,  1,  0,  0,  1],
                            [ 0,  0,  0,  1,  1,  1],
                            [ 0,  0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)

        expected = {frozenset(list(G_true1.edges()))}

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models, _ = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(model_sets, expected)

    def randomG(self, n_nodes, edge_per_node=2, graph_type="ER", seed=2024, mec_check=True):
        scenario = "randomG"
        output_name = f"{scenario}_{n_nodes}_{edge_per_node}_{graph_type}_{seed}"
        facts_location = f"encodings/test_lps/{output_name}.lp"
        logger_setup(output_name)
        logging.info(f"===============Running {scenario}===============")
        logging.info(f"n_nodes={n_nodes}, edge_per_node={edge_per_node}, graph_type={graph_type}, seed={seed}")
        s0 = int(n_nodes*edge_per_node)
        if s0 > int(n_nodes*(n_nodes-1)/2):
            logging.info(f'{s0} is too many edges, setting s0 to the max:', int(n_nodes*(n_nodes-1)/2))
            s0 = int(n_nodes*(n_nodes-1)/2)
        random_stability(seed)
        B_true = simulate_dag(d=n_nodes, s0=s0, graph_type=graph_type)
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)
        true_seplist = find_all_d_separations_sets(G_true, verbose=False)
        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)
        expected = frozenset(set(G_true1.edges()))

        models, _ = CausalABA(n_nodes, facts_location)
        model_sets, MECs = set_of_models_to_set_of_graphs(models, n_nodes, mec_check)
           
        self.assertIn(expected, model_sets)

    ############################################################################################################
    #######                                 CausalABA with PC facts                                     ########
    ############################################################################################################

    def four_node_shapPC_PC_facts(self):
        scenario = "four_node_shapPC_PC_facts"
        alpha = 0.05
        facts_location = f"encodings/test_lps/{scenario}.lp"
        facts_location_I = f"encodings/test_lps/{scenario}_I.lp"
        facts_location_wc = f"encodings/test_lps/{scenario}_wc.lp"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  0],
                            [ 0,  0,  1,  1],
                            [ 0,  0,  0,  1],
                            [ 0,  0,  0,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)

        expected = frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})

        true_seplist = find_all_d_separations_sets(G_true)

        cg = simulate_data_and_run_PC(G_true, alpha)

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
                                                   fact_pct=0.27, print_models=False)
        if multiple_solutions:
            for model in model_sets:
                models, MECs = set_of_models_to_set_of_graphs(model, n_nodes)
                set_of_model_sets.append(models)
        else:
            models, MECs = set_of_models_to_set_of_graphs(model_sets, n_nodes)

        if len(set_of_model_sets) > 0:
            logging.info(f"Number of solutions found: {len(set_of_model_sets)}")
            count_right = 0
            for model_sets in set_of_model_sets:
                if expected in model_sets:
                    count_right += 1
                    logging.debug(f"Models set: {model_sets}")
            logging.info(f"Number of right solutions found: {count_right}")
            self.assertTrue(count_right > 0)
        else:
            self.assertIn(expected, models)

    def five_node_colombo_PC_facts(self):
        scenario = "five_node_colombo_PC_facts"
        alpha = 0.05
        facts_location = f"encodings/test_lps/{scenario}.lp"
        facts_location_I = f"encodings/test_lps/{scenario}_I.lp"
        facts_location_wc = f"encodings/test_lps/{scenario}_wc.lp"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  1,  1,  1],
                            [ 0,  0,  1,  0,  1],
                            [ 0,  0,  0,  1,  1],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        expected = frozenset({(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)})

        true_seplist = find_all_d_separations_sets(G_true)

        cg = simulate_data_and_run_PC(G_true, alpha)

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
                                                   fact_pct=0.27, print_models=False)
        if multiple_solutions:
            for model in model_sets:
                models, MECs = set_of_models_to_set_of_graphs(model, n_nodes)
                set_of_model_sets.append(models)
        else:
            models, MECs = set_of_models_to_set_of_graphs(model_sets, n_nodes)

        if len(set_of_model_sets) > 0:
            logging.info(f"Number of solutions found: {len(set_of_model_sets)}")
            count_right = 0
            for model_sets in set_of_model_sets:
                if expected in model_sets:
                    count_right += 1
                    logging.debug(f"Models set: {model_sets}")
            logging.info(f"Number of right solutions found: {count_right}")
            self.assertTrue(count_right > 0)
        else:
            self.assertIn(expected, models)

    def five_node_sprinkler_PC_facts(self):
        scenario = "five_node_sprinkler_PC_facts"
        alpha = 0.05
        facts_location = f"encodings/test_lps/{scenario}.lp"
        facts_location_I = f"encodings/test_lps/{scenario}_I.lp"
        facts_location_wc = f"encodings/test_lps/{scenario}_wc.lp"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        expected = frozenset({(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)})

        true_seplist = find_all_d_separations_sets(G_true)

        cg = simulate_data_and_run_PC(G_true, alpha)

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
                                                   fact_pct=0.27, print_models=False)
        if multiple_solutions:
            for model in model_sets:
                models, MECs = set_of_models_to_set_of_graphs(model, n_nodes)
                set_of_model_sets.append(models)
        else:
            models, MECs = set_of_models_to_set_of_graphs(model_sets, n_nodes)

        if len(set_of_model_sets) > 0:
            logging.info(f"Number of solutions found: {len(set_of_model_sets)}")
            count_right = 0
            for model_sets in set_of_model_sets:
                if expected in model_sets:
                    count_right += 1
                    logging.debug(f"Models set: {model_sets}")
            logging.info(f"Number of right solutions found: {count_right}")
            self.assertTrue(count_right > 0)
        else:
            self.assertIn(expected, models)

    def randomG_PC_facts(self, n_nodes, edge_per_node=2, graph_type="ER", seed=2024, mec_check=True):
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
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)
        expected = frozenset(set(G_true1.edges()))

        true_seplist = find_all_d_separations_sets(G_true, verbose=False)

        cg = simulate_data_and_run_PC(G_true, alpha)

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
                                                   fact_pct=base_pct, search_for_models='all_subsets',
                                                   opt_mode='optN', print_models=False)

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
            self.assertTrue(count_right > 0)
        else:
            self.assertIn(expected, models)

    def test_metrics_perfect(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = MetricsDAG(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['fdr'], 0)
        self.assertEqual(metrics['tpr'], 1)
        self.assertEqual(metrics['fpr'], 0)
        self.assertEqual(metrics['shd'], 0)
        self.assertEqual(metrics['nnz'], n_edges)
        self.assertEqual(metrics['precision'], 1)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['F1'], 1)
        self.assertEqual(metrics['sid'], 0)
        
        ## calculate metrics for CPDAG
        metrics = MetricsDAG(B_est, B_true, cpdag=True).metrics
        ## test metrics
        self.assertEqual(metrics['fdr'], 0)
        self.assertEqual(metrics['tpr'], 1)
        self.assertEqual(metrics['fpr'], 0)
        self.assertEqual(metrics['shd'], 0)
        self.assertEqual(metrics['nnz'], n_edges)
        self.assertEqual(metrics['precision'], 1)
        self.assertEqual(metrics['recall'], 1)
        self.assertEqual(metrics['F1'], 1)

        self.assertEqual(metrics['SID_cpdag'][0], 0)
        self.assertEqual(metrics['SID_cpdag'][1], 12)

    def test_metrics_errors(self):
        ## true DAG
        B_true = np.array( [[ 0,  1,  1,  0,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  1,  0],
                            [ 0,  0,  0,  0,  1],
                            [ 0,  0,  0,  0,  0]])
        n_edges = B_true.sum()

        ## estimated DAG
        B_est = np.array( [[ 0,  1,  1,  0,  0],
                           [ 0,  0,  0,  0,  0],
                           [ 0,  0,  0,  1,  1],
                           [ 0,  0,  0,  0,  1],
                           [ 0,  0,  0,  0,  0]])
        
        ## calculate metrics
        metrics = MetricsDAG(B_est, B_true).metrics
        ## test metrics
        self.assertEqual(metrics['fdr'], 0.2) #1/5
        self.assertEqual(metrics['tpr'], 0.8) #4/5
        self.assertEqual(metrics['fpr'], 0.2) #1/5
        self.assertEqual(metrics['shd'], 2)
        self.assertEqual(metrics['nnz'], 5)
        self.assertEqual(metrics['precision'], 0.8)
        self.assertEqual(metrics['recall'], 0.8)
        self.assertEqual(metrics['F1'], 0.8)
        self.assertEqual(metrics['sid'], 2)
        
        ## calculate metrics for CPDAG
        metrics = MetricsDAG(B_est, B_true, cpdag=True).metrics
        ## test metrics
        self.assertEqual(metrics['fdr'], 0.2) #1/5
        self.assertEqual(metrics['tpr'], 0.8) #4/5
        self.assertEqual(metrics['fpr'], 0.2) #1/5
        self.assertEqual(metrics['shd'], 3)
        self.assertEqual(metrics['nnz'], 5)
        self.assertEqual(metrics['precision'], 0.8)
        self.assertEqual(metrics['recall'], 0.8)
        self.assertEqual(metrics['F1'], 0.8)

        self.assertEqual(metrics['SID_cpdag'][0], 2)
        self.assertEqual(metrics['SID_cpdag'][1], 15)

start = datetime.now()
TestCausalABA().three_node_all_graphs()
TestCausalABA().three_node_graph_empty()
TestCausalABA().collider()
TestCausalABA().chains_confounder()
TestCausalABA().one_edge()
TestCausalABA().incompatible_Is()
TestCausalABA().four_node_all_graphs()
TestCausalABA().four_node_shapPC_example()
TestCausalABA().incompatible_chain()
TestCausalABA().five_node_all_graphs()
TestCausalABA().five_node_colombo_example()
TestCausalABA().five_node_sprinkler_example()
## TestCausalABA().six_node_all_graphs() ## This test takes 8 minutes to run, 3.7M models
TestCausalABA().six_node_example()
TestCausalABA().randomG(7, 1, "ER", 2024)
TestCausalABA().randomG(8, 1, "ER", 2024)
TestCausalABA().randomG(9, 1, "ER", 2024) ## 13 seconds, 4 models
## TestCausalABA().randomG(10, 1, "ER", 2024) ## This test takes 2 minutes to run, 4 models

TestCausalABA().test_metrics_perfect()
TestCausalABA().test_metrics_errors()

# TestCausalABA().four_node_shapPC_PC_facts() ## Does not pass, needs accuracy evaluation
# TestCausalABA().five_node_colombo_PC_facts() ## Does not pass, needs accuracy evaluation
# TestCausalABA().five_node_sprinkler_PC_facts() ## Does not pass, needs accuracy evaluation
# TestCausalABA().randomG_PC_facts(4, 1, "ER", 2024) ## Does not pass, needs accuracy evaluation

logging.info(f"Total time={str(datetime.now()-start)}")