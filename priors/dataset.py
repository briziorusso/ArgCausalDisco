import json
import re
from itertools import islice, chain, combinations
from typing import Callable, Iterable
from pathlib import Path

import networkx as nx
import numpy as np
import pyagrum as gum
import rustworkx as rx
from scipy.stats import spearmanr


ambiguous_concepts = {
    "illness",
    "illnesses",
    "grief",
    "early_death",
    "hospitalization",
    "serious_problems",
    "neglect",
}
with (Path(__file__).parent / "causal_relationships.json").open("r") as f:
    causal_concepts = json.load(f)
concepts = sorted(
    causal_concepts.keys() | set(chain.from_iterable(causal_concepts.values()))
)
concepts_indexes = {c: i for i, c in enumerate(concepts)}

embeddings = np.load(Path(__file__).parent / "concept_embeddings.npy")
assert embeddings.shape[0] == len(concepts)

causal_net_nx = nx.from_dict_of_lists(causal_concepts, create_using=nx.DiGraph)
causal_net_nx.remove_nodes_from(ambiguous_concepts)
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


def heuristic_by_semantics(
    candidates: Iterable[dict[int, int]],
    w_compact: float = 1,
    w_specificity: float = 1,
    w_correlation: float = 1,
) -> dict[int, int] | None:
    """Heuristic for selecting the most specific concept groups."""

    def _calc_semantic_score(subgraph_mapping: dict[int, int]) -> int:
        node_indices = list(subgraph_mapping.keys())
        node_names = [causal_net.get_node_data(i) for i in node_indices]
        node_vectors = np.array(
            [embeddings[concepts_indexes[name]] for name in node_names]
        )

        centroid = np.mean(node_vectors, axis=0)
        sims = np.dot(node_vectors, centroid) / (
            np.linalg.norm(node_vectors, axis=1) * np.linalg.norm(centroid)
        )
        distances = 1 - sims
        score_compactness = np.mean(distances)

        degrees = np.array(
            [causal_net.in_degree(i) + causal_net.out_degree(i) for i in node_indices]
        )
        log_degrees = np.log(degrees + 1)  # Add 1 to avoid log(0)
        score_specificity = np.mean(log_degrees)

        subgraph = nx.induced_subgraph(causal_net_nx, node_names).to_undirected()
        shortest_paths = dict(nx.all_pairs_shortest_path(subgraph))
        graph_distances = []
        semantic_distances = []
        for i1, i2 in combinations(range(len(node_names)), 2):
            graph_distances.append(len(shortest_paths[node_names[i1]][node_names[i2]]))

            vec1, vec2 = node_vectors[i1], node_vectors[i2]
            sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            semantic_distances.append(1 - sim)

        correlation, _ = spearmanr(semantic_distances, graph_distances)

        return (w_compact * score_compactness) + (w_specificity * score_specificity) + (w_correlation * (1-correlation.item()))

    return min(
        candidates,
        # Heuristic for selecting the most specific concept groups
        key=_calc_semantic_score,
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
