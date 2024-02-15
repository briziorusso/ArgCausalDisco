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
from causalaba import CausalABA
import networkx as nx
import numpy as np
import pandas as pd
from utils import model_to_set_of_arrows, find_all_d_separations_sets, simulate_data_and_run_PC, extract_test_elements_from_symbol


class TestCausalABA(unittest.TestCase):

    def three_node_all_graphs(self):
        print("===============Running three_node_all_graphs===============")
        num_of_nodes = 3
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
        models = CausalABA(num_of_nodes)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(model_sets, expected)

    def three_node_graph_empty(self):
        print("===============Running three_node_graph_empty===============")
        num_of_nodes = 3
        expected = set([
            frozenset(),
        ])
        models = CausalABA(num_of_nodes, "encodings/test_lps/three_node_empty.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(model_sets, expected)

    def collider(self):
        print("===============Running collider===============")
        num_of_nodes = 3
        expected = set([
            frozenset({(0, 2), (1, 2)}),
        ])
        models = CausalABA(num_of_nodes, "encodings/test_lps/collider.lp", show=["arrow", "indep"])
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))

        self.assertEqual(model_sets, expected)

    def chains_confounder(self):
        print("===============Running chains_confounder===============")
        num_of_nodes = 3
        expected = set([
            ##chains
            frozenset({(1, 2), (2, 0)}),
            frozenset({(0, 2), (2, 1)}),
            ##confounder
            frozenset({(2, 0), (2, 1)})
        ])
        models = CausalABA(num_of_nodes, "encodings/test_lps/chains_confounder.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))

        self.assertEqual(model_sets, expected)

    def one_edge(self):
        print("===============Running one_edge===============")
        num_of_nodes = 3
        expected = set([
            frozenset({(0, 1)}),
            frozenset({(1, 0)}),
        ])
        models = CausalABA(num_of_nodes, "encodings/test_lps/one_edge.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))

        self.assertEqual(model_sets, expected)

    def incompatible_Is(self):
        print("===============Running incompatible_Is===============")
        num_of_nodes = 3
        expected = set()
        models = CausalABA(num_of_nodes, "encodings/test_lps/incompatible_Is.lp")
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))       

        self.assertEqual(model_sets, expected)

    def four_node_all_graphs(self):
        print("===============Running four_node_graph_full===============")
        num_of_nodes = 4
        models = CausalABA(num_of_nodes, print_models=False)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 543)

    def four_node_example(self):
        scenario = "four_node_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        print(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  0,  0],
                            [ 0,  0,  0,  0],
                            [ 1,  1,  0,  0],
                            [ 0,  1,  1,  0],
                            ])
        num_of_nodes = B_true.shape[0]
        print(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        print(G_true.edges)

        expected = set([
            frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        ])

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models = CausalABA(num_of_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(set(model_sets), expected)

    def five_node_all_graphs(self):
        print("===============Running five_node_all_graphs===============")
        num_of_nodes = 5
        models = CausalABA(num_of_nodes, print_models=False) 
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 29281)

    def five_node_colombo_example(self):
        scenario = "five_node_colombo_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        print(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  0,  0],
                            [ 0,  0,  0,  0],
                            [ 1,  1,  0,  0],
                            [ 0,  1,  1,  0],
                            ])
        num_of_nodes = B_true.shape[0]
        print(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        print(G_true.edges)

        expected = set([
            frozenset({(0, 2), (1, 2), (1, 3), (2, 3)})
        ])

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models = CausalABA(num_of_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(set(model_sets), expected)

    def six_node_all_graphs(self):
        print("===============Running six_node_all_graphs===============")
        num_of_nodes = 6
        models = CausalABA(num_of_nodes, print_models=False)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            
        
        self.assertEqual(len(model_sets), 3781503)


    def six_node_example(self):
        scenario = "six_node_example"
        facts_location = f"encodings/test_lps/{scenario}.lp"
        print(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0],
                            [ 1,  1,  0,  0,  0,  0],
                            [ 1,  0,  1,  0,  0,  0],
                            [ 1,  0,  1,  0,  0,  0],
                            [ 1,  1,  1,  1,  1,  0]])
        num_of_nodes = B_true.shape[0]
        print(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        print(G_true.edges)

        inv_nodes_dict = {n:int(n.replace("X",""))-1 for n in G_true.nodes()}
        G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)

        expected = {frozenset(list(G_true1.edges()))}

        true_seplist = find_all_d_separations_sets(G_true)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models = CausalABA(num_of_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertEqual(model_sets, expected)

    def five_node_colombo_PC_facts(self):
        scenario = "five_node_colombo_PC_facts"
        alpha = 0.05
        facts_location = f"encodings/test_lps/{scenario}.lp"
        print(f"===============Running {scenario}===============")
        B_true = np.array( [[ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 1,  1,  0,  0,  0],
                            [ 1,  0,  1,  0,  0],
                            [ 1,  1,  1,  1,  0]])
        num_of_nodes = B_true.shape[0]
        print(B_true)
        G_true = nx.DiGraph(pd.DataFrame(B_true.T, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))
        print(G_true.edges)

        expected = frozenset({(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)})

        true_seplist = find_all_d_separations_sets(G_true)

        cg = simulate_data_and_run_PC(G_true, alpha)

        facts = []
        for test in true_seplist:
            X, S, Y, dep_type = extract_test_elements_from_symbol(test)

            test_PC = [t for t in cg.sepset[X,Y] if set(t[0])==S]
            if len(test_PC)==1:
                p = test_PC[0][1]
                dep_type_PC = "I" if p > alpha else "D" 
                if dep_type == dep_type_PC:
                    facts.append(test)

        with open(facts_location, "w") as f:
            for s in true_seplist:
                f.write(s + "\n")

        models = CausalABA(num_of_nodes, facts_location)
        model_sets = set()
        for model in models:
            arrows = model_to_set_of_arrows(model, num_of_nodes)
            model_sets.add(frozenset(arrows))            

        self.assertIn(expected, model_sets)

TestCausalABA().three_node_all_graphs()
TestCausalABA().three_node_graph_empty()
TestCausalABA().collider()
TestCausalABA().chains_confounder()
TestCausalABA().one_edge()
TestCausalABA().incompatible_Is()
TestCausalABA().four_node_all_graphs()
TestCausalABA().four_node_example()
TestCausalABA().five_node_all_graphs()
TestCausalABA().five_node_colombo_example()
## TestCausalABA().six_node_all_graphs() ## This test takes 8 minutes to run, 3.7M models
TestCausalABA().six_node_example()
TestCausalABA().five_node_colombo_PC_facts()