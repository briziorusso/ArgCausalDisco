import pytest
from src.utils.fact_utils import facts_from_file
from src.causal_aba.factory import ABASPSolverFactory
from src.utils.utils import get_arrows_from_model
from src.utils.enums import Fact, RelationEnum
from src.utils.enums import SemanticEnum


@pytest.mark.parametrize(
    "filepath, n_nodes, assert_equal, expected",
    [('tests/data/five_node_colombo_example.lp', 5, True, {(0, 2), (1, 2), (0, 4), (2, 4), (3, 4), (0, 3), (1, 4), (2, 3)}),
     ('tests/data/five_node_sprinkler_example.lp', 5, False, {(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)}),
     ('tests/data/six_node_example.lp', 6, True, {(2, 4), (1, 2), (0, 4), (1, 5), (0, 3), (2, 3), (0, 2), (4, 5), (0, 5), (2, 5), (3, 5)})],
    ids=['colombo', 'sprinkler', 'six_node'])
def test_graph_examples(filepath, n_nodes, assert_equal, expected):
    facts = facts_from_file(filepath)
    factory = ABASPSolverFactory(n_nodes)
    solver = factory.create_solver(facts=facts)
    models = solver.enumerate_extensions(SemanticEnum.ST.value)
    arrow_sets = [get_arrows_from_model(model) for model in models]

    assert expected in arrow_sets

    if assert_equal:
        assert len(arrow_sets) == 1


def test_arrows_are_removed_when_specified():
    edges_to_remove = [frozenset({0, 1}), frozenset({1, 2}), frozenset({3, 4})]
    facts = [
        Fact(
            relation=RelationEnum.indep,
            node1=X,
            node2=Y,
            node_set={},
            score=0,
        )
        for X, Y in edges_to_remove
    ]
    n_nodes = 5
    # meaning there are 4 + 3 + 2 + 1 = 10 edges
    # minus the 3 edges in edges_to_remove equals 7 edges

    factory = ABASPSolverFactory(n_nodes)
    solver = factory.create_solver(facts=facts)
    arrow_assumptions = [a for a in solver.assumptions if a.startswith("arr_")]
    noe_assumptions = [a for a in solver.assumptions if a.startswith("noe_")]

    atoms = list(solver.abaf.idx_to_atom.values())
    assert len(atoms) == len(set(atoms))  # no duplicates

    arrow_atoms = [a for a in atoms if a.startswith("arr_")]
    noe_atoms = [a for a in atoms if a.startswith("noe_")]

    assert len(arrow_assumptions) == len(set(arrow_assumptions))
    assert len(arrow_assumptions) == 7*2  # 7 edges * 2 directions

    assert len(noe_assumptions) == len(set(noe_assumptions))
    assert len(noe_assumptions) == 7  # 7 possible edges

    # no extra atoms created when declaring the rules
    assert set(arrow_assumptions) == set(arrow_atoms)
    assert set(noe_assumptions) == set(noe_atoms)
