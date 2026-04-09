import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

from ArgCausalDisco.causalaba import CausalABA
from ArgCausalDisco.utils.data_utils import simulate_dag
from ArgCausalDisco.utils.graph_utils import set_of_models_to_set_of_graphs
from ArgCausalDisco.utils.helpers import random_stability

from src.utils.fact_utils import facts_from_file
from src.abapc import get_arrow_sets_from_facts

import pandas as pd
import networkx as nx
import pytest


@pytest.mark.parametrize("n_nodes, edge_per_node", [
    (7, 1),
    (8, 1),
    (9, 1),
])
def test_randomG(n_nodes, edge_per_node):
    scenario = "randomG"
    graph_type = "ER"
    seed = 2024

    output_name = f"{scenario}_{n_nodes}_{edge_per_node}_{graph_type}_{seed}"
    facts_location = f"tests/data/{output_name}.lp"
    s0 = int(n_nodes*edge_per_node)
    if s0 > int(n_nodes*(n_nodes-1)/2):
        s0 = int(n_nodes*(n_nodes-1)/2)
    random_stability(seed)
    B_true = simulate_dag(d=n_nodes, s0=s0, graph_type=graph_type)
    G_true = nx.DiGraph(pd.DataFrame(
        B_true, columns=[f"X{i+1}" for i in range(B_true.shape[1])], index=[f"X{i+1}" for i in range(B_true.shape[1])]))

    inv_nodes_dict = {n: int(n.replace("X", ""))-1 for n in G_true.nodes()}
    G_true1 = nx.relabel_nodes(G_true, inv_nodes_dict)
    expected = frozenset(set(G_true1.edges()))

    models, _ = CausalABA(n_nodes, facts_location, skeleton_rules_reduction=True, weak_constraints=False,
                          fact_pct=1.0, search_for_models='No', opt_mode='optN', print_models=False)
    models, _ = set_of_models_to_set_of_graphs(models, n_nodes, False)

    facts = facts_from_file(facts_location)

    new_models, _ = get_arrow_sets_from_facts(facts, n_nodes)

    assert expected in models, f"Expected graph is not in old implementation models!"
    assert expected in new_models, f"Expected graph is not in new implementation models!"
    assert set(new_models) == set(models), f"New implementation models are not equal to old implementation models!"

    print(f"Test passed successfully! {output_name} with {n_nodes} nodes and {edge_per_node} edges per node.")


if __name__ == "__main__":
    test_randomG(7, 1)
    test_randomG(8, 1)
    test_randomG(9, 1)
