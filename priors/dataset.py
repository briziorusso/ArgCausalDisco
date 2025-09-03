import json
import re
from itertools import islice
from typing import Callable, Iterable
from pathlib import Path

import networkx as nx
import pyagrum as gum
import rustworkx as rx


with (Path(__file__).parent / "causal_relationships.json").open("r") as f:
    causal_concepts = json.load(f)
causal_net_nx = nx.from_dict_of_lists(causal_concepts, create_using=nx.DiGraph)
causal_net: rx.PyDiGraph = rx.networkx_converter(causal_net_nx)


def heuristic_by_degrees(candidates: Iterable[dict[int, int]]) -> dict[int, int] | None:
    """Heuristic for selecting the most specific concept groups."""
    return min(
        candidates,
        # Heuristic for selecting the most specific concept groups
        key=lambda x: sum(
            causal_net.in_degree(node_id) + causal_net.out_degree(node_id)
            for node_id in x.keys()
        ),
        default=None,
    )


def generate_causal_graph(
    n: int,
    ratio_arc: float = 1.2,
    domain_size: int = 2,
    seed: int = 42,
    subgraph_candidates: int | None = 3,
    heuristic_func: Callable[
        [Iterable[dict[int, int]]], dict[int, int]
    ] = lambda iterable: next(iterable, None),
) -> gum.BayesNet:
    """
    Pipeline for generating a synthetic causal graph with semantically meaningful
    variables and relationships.

    Parameters
    ----------
    n : int
        The number of variables in the causal graph. Must >= 4.
    ratio_arc : float, optional
        The ratio of arcs to variables, by default 1.2.
    domain_size : int, optional
        The number of states for each variable, by default 2.
    seed : int, optional
        The random seed for reproducibility, by default 42.
    subgraph_candidates : int | None, by default 3.
        The number of subgraph candidates to consider, by default 3. The
        algorithm will choose the best one among them based on the provided
        heuristic function.
    heuristic_func: Callable[[Iterable[dict[int, int]]], dict[int, int]]
        A function that takes an iterable of candidate subgraph mappings and
        returns the best one.

    Returns
    -------
    gum.BayesNet
        The generated causal graph.
    """
    assert n >= 4, "The number of variables must be at least 4."
    gum.initRandom(seed)
    while True:
        # Generate an unlabeled Bayesian network with random CPTs
        bn_rand = gum.randomBN(n=n, ratio_arc=ratio_arc, domain_size=domain_size)

        # Find a subgraph that's isomorphic to the random DAG
        G_rand = rx.networkx_converter(
            nx.from_numpy_array(bn_rand.adjacencyMatrix(), create_using=nx.DiGraph)
        )
        iso_mapping = heuristic_func(
            islice(
                rx.vf2_mapping(causal_net, G_rand, id_order=False, subgraph=True),
                subgraph_candidates,
            )
        )
        if iso_mapping is not None:
            break

    for concept_id, bn_id in iso_mapping.items():
        bn_rand.changeVariableName(
            bn_id, re.sub("\W+|^(?=\d)", "_", causal_net.get_node_data(concept_id))
        )

    return bn_rand
