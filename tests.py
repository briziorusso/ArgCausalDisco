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
from utils import model_to_set_of_arrows

class TestCausalABA(unittest.TestCase):

    def three_node_graph_full(self):
        print("===============Running three_node_graph_full===============")
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

TestCausalABA().three_node_graph_full()
TestCausalABA().three_node_graph_empty()
TestCausalABA().collider()