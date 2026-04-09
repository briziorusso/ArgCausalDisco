import numpy as np
from itertools import chain, combinations, product
from aspforaba.src.aspforaba.abaf import AssumptionSet
from typing import List, Dict, Tuple, Set, FrozenSet
import networkx as nx


def is_unique(ary):
    return len(ary) == len(set(ary))


def powerset(s):
    s = sorted(list(s))
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def unique_product(elements, repeat: int):
    '''
    Generate all combinations of elements without duplicates.
    '''
    for element_set in product(elements, repeat=repeat):
        if is_unique(element_set):
            yield element_set


def parse_arrow(arrow: str):
    """
    Parse an arrow string into a tuple of integers.
    :param arrow: The arrow string to parse.
    :return: A tuple of integers representing the arrow.
    """

    try:
        _, node1, node2 = arrow.split("_")
        node1 = int(node1)
        node2 = int(node2)
    except ValueError as e:
        raise ValueError(f"Arrow must be of form arr_<node1>_<node2>, got {arrow}. Here is the exception {e}")

    return node1, node2


def get_arrows_from_model(model: AssumptionSet):
    """
    Get the arrows from a model.
    :param model: The model to extract arrows from.
    :return: A set of tuples representing the arrows in the model.
    """
    arrows = set()
    for assumption in model.assumptions:
        if assumption.startswith("arr_"):
            node1, node2 = parse_arrow(assumption)
            arrows.add((node1, node2))
    return frozenset(arrows)


def get_arrows_from_assumptions(assumptions: list):
    """
    Get the arrows from a model.
    :param model: The model to extract arrows from.
    :return: A set of tuples representing the arrows in the model.
    """
    arrows = set()
    for assumption in assumptions:
        if assumption.startswith("arr_"):
            node1, node2 = parse_arrow(assumption)
            arrows.add((node1, node2))
    return frozenset(arrows)


def get_matrix_from_arrow_set(arrow_set, n_nodes):
    """
    Get the adjacency matrix from the arrow set.
    Args:
        arrow_set: set
            The arrow set to be converted to an adjacency matrix
        n_nodes: int
            The number of nodes in the graph
    Returns:
        B_est: np.array
            The adjacency matrix of the graph
    """
    B_est = np.zeros((n_nodes, n_nodes), dtype=int)
    for node1, node2 in arrow_set:
        B_est[node1, node2] = 1
    return B_est


def check_arrows_dag(arrows: Set[Tuple[int, int]]) -> bool:    
    """Check if the given set of arrows is acyclic.
    """
    G = nx.DiGraph()
    G.add_edges_from(arrows)
    return nx.is_directed_acyclic_graph(G)
