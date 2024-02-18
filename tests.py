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
from collections import defaultdict
from utils import *
# import cdt
# cdt.SETTINGS.rpath = '../R/R-4.1.2/bin/Rscript'
# from cdt.metrics import get_CPDAG

class TestCausalABA(unittest.TestCase):

    def three_node_all_graphs(self):
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
        logging.info("===============Running four_node_graph_full===============")
        n_nodes = 4
        models = CausalABA(n_nodes, print_models=False)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 543)

    def four_node_example(self):
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

    def five_node_colombo_PC_facts(self):
        scenario = "five_node_colombo_PC_facts"
        alpha = 0.05
        facts_location = f"encodings/test_lps/{scenario}.lp"
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
        for test in true_seplist:
            X, S, Y, dep_type = extract_test_elements_from_symbol(test)

            test_PC = [t for t in cg.sepset[X,Y] if set(t[0])==S]
            if len(test_PC)==1:
                p = test_PC[0][1]
                dep_type_PC = "indep" if p > alpha else "dep" 
                if dep_type == dep_type_PC:
                    facts.append(test)

        with open(facts_location, "w") as f:
            for s in facts:
                f.write(s + "\n")

        models = CausalABA(n_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertIn(expected, model_sets)

    def randomG(self, n_nodes, edge_per_node=2, graph_type="ER", seed=2024, mec_check=True):
        scenario = "randomG"
        facts_location = f"encodings/test_lps/{scenario}.lp"
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

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)
        expected = frozenset(set(G_true1.edges()))

        true_seplist = find_all_d_separations_sets(G_true, verbose=False)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        MECs = defaultdict(list)
        MEC_set = set()
        models = CausalABA(n_nodes, facts_location)
        model_sets = set()
        logging.info(   f"Checking MECs")
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))        
            if mec_check:
                adj = model_to_adjacency_matrix(model, n_nodes)
                cp_adj = dag2cpdag(adj)
                #cp_adj = get_CPDAG(adj)
                cp_adj_hashable = map(tuple, cp_adj)
                MECs[cp_adj_hashable] = list(adj.flatten())
                MEC_set.add(frozenset(cp_adj_hashable))
                assert len(MEC_set) == 1, f"More than one MEC found, \n MEC_set={MEC_set}"
           
        self.assertIn(expected, model_sets)

    def randomG_PC_facts(self, n_nodes, edge_per_node=2, graph_type="ER", seed=2024):
        scenario = "randomG_PC_facts"
        alpha = 0.05
        facts_location = f"encodings/test_lps/{scenario}.lp"
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
        for test in true_seplist:
            X, S, Y, dep_type = extract_test_elements_from_symbol(test)

            test_PC = [t for t in cg.sepset[X,Y] if set(t[0])==S]
            if len(test_PC)==1:
                p = test_PC[0][1]
                dep_type_PC = "indep" if p > alpha else "dep" 
                if dep_type == dep_type_PC:
                    facts.append(test)

        with open(facts_location, "w") as f:
            for s in facts:
                f.write(s + "\n")

        models = CausalABA(n_nodes, facts_location, print_models=False)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, n_nodes)
            model_sets.add(frozenset(arrows))            

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
## TestCausalABA().six_node_all_graphs() ## This test takes 8 minutes to run, 3.7M models
# TestCausalABA().six_node_example()
# TestCausalABA().five_node_colombo_PC_facts()
# TestCausalABA().randomG_PC_facts()
TestCausalABA().randomG(7, 1, "ER", 2024)
# TestCausalABA().randomG(9, 1, "ER", 2024)

logging.info(f"Total time={str(datetime.now()-start)}")