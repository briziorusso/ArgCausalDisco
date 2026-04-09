from src.causal_aba.core_factory import CoreABASPSolverFactory
from src.utils.utils import get_arrows_from_model
from src.utils.enums import SemanticEnum


def test_all_3_node_graphs():
    expected = set([
        frozenset(),
        frozenset({(1, 0)}),
        frozenset({(0, 2)}),
        frozenset({(0, 1)}),
        frozenset({(1, 2)}),
        frozenset({(2, 0)}),
        frozenset({(2, 1)}),
        # chains
        frozenset({(0, 2), (2, 1)}),
        frozenset({(0, 1), (2, 0)}),
        frozenset({(1, 0), (2, 1)}),
        frozenset({(1, 2), (2, 0)}),
        frozenset({(1, 0), (0, 2)}),
        frozenset({(0, 1), (1, 2)}),
        # colliders
        frozenset({(0, 1), (2, 1)}),
        frozenset({(0, 2), (1, 2)}),
        frozenset({(1, 0), (2, 0)}),
        # confounders
        frozenset({(2, 0), (2, 1)}),
        frozenset({(1, 0), (1, 2)}),
        frozenset({(0, 1), (0, 2)}),
        # three arrows configurations
        frozenset({(0, 1), (0, 2), (1, 2)}),
        frozenset({(0, 1), (0, 2), (2, 1)}),
        frozenset({(1, 0), (0, 2), (1, 2)}),
        frozenset({(1, 0), (2, 0), (2, 1)}),
        frozenset({(0, 1), (2, 0), (2, 1)}),
        frozenset({(1, 0), (1, 2), (2, 0)})
    ])
    factory = CoreABASPSolverFactory(3)
    solver = factory.create_core_solver(edges_to_remove=set())
    models = solver.enumerate_extensions(SemanticEnum.ST.value)
    arrow_sets = [frozenset(get_arrows_from_model(model)) for model in models]

    assert len(arrow_sets) == len(expected)
    assert set(arrow_sets) == expected


def test_arrows_are_removed_when_specified():
    edges_to_remove = {frozenset({0, 1}), frozenset({1, 2}), frozenset({3, 4})}
    n_nodes = 5
    # meaning there are 4 + 3 + 2 + 1 = 10 edges
    # minus the 3 edges in edges_to_remove equals 7 edges

    factory = CoreABASPSolverFactory(n_nodes)
    solver = factory.create_core_solver(edges_to_remove=edges_to_remove)
    arrow_assumptions = [a for a in solver.assumptions if a.startswith("arr_")]
    noe_assumptions = [a for a in solver.assumptions if a.startswith("noe_")]

    arrow_atoms = [a for a in solver.atoms if a.startswith("arr_")]
    noe_atoms = [a for a in solver.atoms if a.startswith("noe_")]

    assert len(arrow_assumptions) == len(set(arrow_assumptions))
    assert len(arrow_assumptions) == 7*2  # 7 edges * 2 directions

    assert len(noe_assumptions) == len(set(noe_assumptions))
    assert len(noe_assumptions) == 7  # 7 possible edges

    # no extra atoms created when declaring the rules
    assert set(arrow_assumptions) == set(arrow_atoms)
    assert set(noe_assumptions) == set(noe_atoms)
