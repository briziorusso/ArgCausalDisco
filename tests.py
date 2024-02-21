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
        models = CausalABA(n_nodes)
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
        models = CausalABA(n_nodes, "encodings/test_lps/three_node_empty.lp")
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
        models = CausalABA(n_nodes, "encodings/test_lps/collider.lp", show=["arrow", "indep"])
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
        models = CausalABA(n_nodes, "encodings/test_lps/chains_confounder.lp")
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
        models = CausalABA(n_nodes, "encodings/test_lps/one_edge.lp")
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
        models = CausalABA(n_nodes, "encodings/test_lps/incompatible_Is.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))       

        self.assertEqual(model_sets, expected)

    def four_node_all_graphs(self):
        logger_setup()
        logging.info("===============Running four_node_graph_full===============")
        n_nodes = 4
        models = CausalABA(n_nodes, print_models=False)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 543)

    def four_node_example(self):
        logger_setup()
        scenario = "four_node_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  0,  0],
                            [ 0,  0,  0,  0],
                            [ 1,  1,  0,  0],
                            [ 0,  1,  1,  0],
                            ])
        n_nodes = B_true.shape[0]
        logging.info(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.info(G_true.edges)

        expected = set([
            frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        ])

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models = CausalABA(n_nodes, facts_location)
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

        models = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(set(model_sets), expected)

    def five_node_all_graphs(self):
        logger_setup()
        logging.info("===============Running five_node_all_graphs===============")
        n_nodes = 5
        models = CausalABA(n_nodes, print_models=False) 
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
        B_true = np.array( [[ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 1,  1,  0,  0,  0],
                            [ 1,  0,  1,  0,  0],
                            [ 1,  1,  1,  1,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        expected = {frozenset({(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)})}

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(set(model_sets), expected)

    def six_node_all_graphs(self):
        logger_setup()
        logging.info("===============Running six_node_all_graphs===============")
        n_nodes = 6
        models = CausalABA(n_nodes, print_models=False)
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
        B_true = np.array( [[ 0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0],
                            [ 1,  1,  0,  0,  0,  0],
                            [ 1,  0,  1,  0,  0,  0],
                            [ 1,  0,  1,  0,  0,  0],
                            [ 1,  1,  1,  1,  1,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)

        expected = {frozenset(list(G_true1.edges()))}

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(model_sets, expected)

    def five_node_colombo_PC_facts(self, mec_check=True):
        scenario = "five_node_colombo_PC_facts"
        alpha = 0.05
        facts_location = f"encodings/test_lps/{scenario}.lp"
        logger_setup(scenario)
        logging.info(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 1,  1,  0,  0,  0],
                            [ 1,  0,  1,  0,  0],
                            [ 1,  1,  1,  1,  0]])
        n_nodes = B_true.shape[0]
        logging.debug(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
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
                if dep_type == dep_type_PC:
                    facts.append(test)
                elif dep_type == "indep":
                    count_wrong += 1
                    facts.append(test.replace("indep", "dep"))
                elif dep_type == "dep":
                    facts.append(test.replace("dep", "indep"))
                    count_wrong += 1
        
        logging.info(f"Number of wrong facts={count_wrong}")
        with open(facts_location, "w") as f:
            for s in facts:
                f.write(s + "\n")

        models = CausalABA(n_nodes, facts_location)
        model_sets = set_of_models_to_set_of_graphs(models, n_nodes, mec_check)

        self.assertIn(expected, model_sets)

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
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        logging.debug(G_true.edges)
        true_seplist = find_all_d_separations_sets(G_true, verbose=False)
        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)
        expected = frozenset(set(G_true1.edges()))

        models = CausalABA(n_nodes, facts_location)
        model_sets = set_of_models_to_set_of_graphs(models, n_nodes, mec_check)
           
        self.assertIn(expected, model_sets)

    def randomG_PC_facts(self, n_nodes, edge_per_node=2, graph_type="ER", seed=2024, mec_check=True):
        scenario = "randomG_PC_facts"
        alpha = 0.05
        output_name = f"{scenario}_{n_nodes}_{edge_per_node}_{graph_type}_{seed}"
        facts_location = f"encodings/test_lps/{output_name}.lp"
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
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
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
                if dep_type == dep_type_PC:
                    facts.append(test)
                elif dep_type == "indep":
                    count_wrong += 1
                    facts.append(test.replace("indep", "dep"))
                elif dep_type == "dep":
                    count_wrong += 1
                    facts.append(test.replace("dep", "indep"))
        
        logging.info(f"Number of wrong facts={count_wrong}")
        with open(facts_location, "w") as f:
            for s in facts:
                f.write(s + "\n")
        
        models = CausalABA(n_nodes, facts_location)
        model_sets = set_of_models_to_set_of_graphs(models, n_nodes, mec_check)

        self.assertIn(expected, model_sets)

start = datetime.now()
# TestCausalABA().three_node_all_graphs()
# TestCausalABA().three_node_graph_empty()
# TestCausalABA().collider()
# TestCausalABA().chains_confounder()
# TestCausalABA().one_edge()
# TestCausalABA().incompatible_Is()
# TestCausalABA().four_node_all_graphs()
# TestCausalABA().four_node_example()
# TestCausalABA().incompatible_chain()
# TestCausalABA().five_node_all_graphs()
# TestCausalABA().five_node_colombo_example()
# # TestCausalABA().six_node_all_graphs() ## This test takes 8 minutes to run, 3.7M models
# TestCausalABA().six_node_example()
# TestCausalABA().randomG(10, 1, "ER", 2024)

# TestCausalABA().five_node_colombo_PC_facts()
TestCausalABA().randomG_PC_facts(9, 1, "ER", 2024)


logging.info(f"Total time={str(datetime.now()-start)}")