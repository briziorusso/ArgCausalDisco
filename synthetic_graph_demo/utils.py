import json
import os
import random
import tempfile
from itertools import chain, combinations, islice
from pathlib import Path
from typing import Callable, Iterable

import igraph as ig
import networkx as nx
import numpy as np
import pyagrum as gum
import pydot
import rustworkx as rx
from scipy.stats import spearmanr

CAUSE_NET_PATH = Path(__file__).parent / "causal_relationships.json"
EMBEDDINGS_PATH = Path(__file__).parent / "concept_embeddings.npy"
if not EMBEDDINGS_PATH.exists():
    print(
        "Warning: 'concept_embeddings.npy' not found. Please download it from the link in README.md"
    )

ambiguous_concepts = {
    "illness",
    "illnesses",
    "grief",
    "early_death",
    "hospitalization",
    "serious_problems",
    "neglect",
    "chronic_pain",
    "heart_failure",
    "despair",
    "hopeless",
    "difficulties",
    "premature_death",
    "serious_health_problems",
    "heart_problems",
    "medical_problems",
    "infertility",
    "irritability",
    "numbness",
    "sexual_dysfunction",
    "gout",
    "memory_problems",
    "eye_problems",
    "health_risks",
    "high_blood_pressure",
    "limitations",
    "frustrations",
    "loss_of_function",
}

with CAUSE_NET_PATH.open("r") as f:
    causal_concepts = json.load(f)
concepts = sorted(
    causal_concepts.keys() | set(chain.from_iterable(causal_concepts.values()))
)
concepts_indexes = {c: i for i, c in enumerate(concepts)}

embeddings = np.load(EMBEDDINGS_PATH)
assert embeddings.shape[0] == len(concepts)

cause_net_nx = nx.from_dict_of_lists(causal_concepts, create_using=nx.DiGraph)
cause_net_nx.remove_nodes_from(ambiguous_concepts)
cause_net: rx.PyDiGraph = rx.networkx_converter(cause_net_nx)


def get_random_dag(size: int, num_ones: int) -> np.ndarray:
    """
    Creates a square adjacency matrix with a specified number of 1s randomly
    placed in its lower triangle, ensuring each required row has at least one 1.

    Args:
        size (int): The order (width and height) of the square matrix.
        num_ones (int): The total number of 1s to place in the lower triangle.

    Returns:
        np.ndarray: The resulting size x size NumPy array.
    """
    # --- 1. Input Validation ---
    min_required = size - 1
    max_possible = size * (size - 1) // 2
    if not min_required <= num_ones <= max_possible:
        raise ValueError(
            f"For size {size}, 'num_ones' must be between {min_required} "
            f"and {max_possible}. You provided {num_ones}."
        )

    # --- 2. Generate Guaranteed and Additional Positions ---
    # Use a list comprehension to guarantee a '1' in each row from 1 to size-1.
    guaranteed_positions = {(r, random.randint(0, r - 1)) for r in range(1, size)}

    # Identify all possible positions in the lower triangle.
    all_rows, all_cols = np.tril_indices(size, k=-1)
    all_positions = set(zip(all_rows, all_cols))

    # Sample from the remaining available positions.
    available_positions = list(all_positions - guaranteed_positions)
    additional_positions = random.sample(
        available_positions, k=(num_ones - len(guaranteed_positions))
    )

    # --- 3. Create Matrix and Place all 1s at Once ---
    matrix = np.zeros((size, size), dtype=int)

    # Combine all positions and unpack for advanced indexing.
    final_positions = list(guaranteed_positions) + additional_positions
    if final_positions:
        rows, cols = zip(*final_positions)
        matrix[rows, cols] = 1

    return matrix


def heuristic_by_degrees(
    cause_net: rx.PyDiGraph, candidates: Iterable[dict[int, int]]
) -> dict[int, int] | None:
    """Heuristic for selecting the most specific concept groups."""
    return min(
        candidates,
        # Heuristic for selecting the most specific concept groups
        key=lambda x: sum(
            cause_net.in_degree(node_id) + cause_net.out_degree(node_id)
            for node_id in x.keys()
        ),
        default=None,
    )


def heuristic_by_semantics(
    cause_net: rx.PyDiGraph,
    candidates: Iterable[dict[int, int]],
    w_compact: float = 1,
    w_specificity: float = 1,
    w_correlation: float = 1,
) -> dict[int, int] | None:
    """Heuristic for selecting the most specific concept groups."""

    def _calc_semantic_score(subgraph_mapping: dict[int, int]) -> int:
        cost_compactness = cost_specificity = cost_correlation = 0

        node_indices = list(subgraph_mapping.keys())
        node_names = [cause_net.get_node_data(i) for i in node_indices]
        node_vectors = np.array(
            [embeddings[concepts_indexes[name]] for name in node_names]
        )
        if w_compact:
            centroid = np.mean(node_vectors, axis=0)
            sims = np.dot(node_vectors, centroid) / (
                np.linalg.norm(node_vectors, axis=1) * np.linalg.norm(centroid)
            )
            distances = 1 - sims
            cost_compactness = w_compact * np.mean(distances)

        if w_specificity:
            degrees = np.array(
                [cause_net.in_degree(i) + cause_net.out_degree(i) for i in node_indices]
            )
            log_degrees = np.log(degrees + 1)  # Add 1 to avoid log(0)
            cost_specificity = w_specificity * np.mean(log_degrees)

        if w_correlation:
            subgraph = nx.induced_subgraph(cause_net_nx, node_names).to_undirected()
            shortest_paths = dict(nx.all_pairs_shortest_path(subgraph))
            graph_distances = []
            semantic_distances = []
            for i1, i2 in combinations(range(len(node_names)), 2):
                graph_distances.append(
                    len(shortest_paths[node_names[i1]][node_names[i2]])
                )

                vec1, vec2 = node_vectors[i1], node_vectors[i2]
                sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                semantic_distances.append(1 - sim)

            correlation, _ = spearmanr(semantic_distances, graph_distances)
            cost_correlation = w_correlation * (1 - correlation.item())

        return cost_compactness + cost_specificity + cost_correlation

    return min(
        candidates,
        # Heuristic for selecting the most specific concept groups
        key=_calc_semantic_score,
        default=None,
    )


### From notears repo: https://github.com/xunzheng/notears
def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def generate_causal_graph(
    n_nodes: int,
    n_edges: int,
    subgraph_candidates: int | None = 3,
    heuristic_func: Callable[
        [Iterable[dict[int, int]]], dict[int, int]
    ] = lambda cause_net, iterable: next(iterable, None),
    graph_type: str = "random",
    excluded_concepts: set[str] | None = None,
):
    """
    Pipeline for generating a synthetic causal graph with semantically meaningful
    variables and relationships.

    Parameters
    ----------
    n_nodes : int
        The number of variables in the causal graph.
    n_edges : int
        The number of edges in the causal graph.
    subgraph_candidates : int | None, by default 3.
        The number of subgraph candidates to consider, by default 3. The
        algorithm will choose the best one among them based on the provided
        heuristic function.
    heuristic_func: Callable[[Iterable[dict[int, int]]], dict[int, int]]
        A function that takes an iterable of candidate subgraph mappings and
        returns the best one.

    Returns
    -------
    str
        The DOT representation of the generated causal graph.
    """
    local_cause_net = cause_net
    if excluded_concepts is not None:
        local_cause_net = rx.networkx_converter(
            nx.subgraph_view(
                cause_net_nx, filter_node=lambda n: n not in excluded_concepts
            )
        )
    while True:
        G = nx.from_numpy_array(
            simulate_dag(n_nodes, n_edges, graph_type)
            if graph_type in ["ER", "SF"]
            else get_random_dag(n_nodes, n_edges),
            create_using=nx.DiGraph,
        )
        G_rand = rx.networkx_converter(G)
        iso_mapping = heuristic_func(
            local_cause_net,
            islice(
                rx.vf2_mapping(local_cause_net, G_rand, id_order=False, subgraph=True),
                subgraph_candidates,
            ),
        )
        if iso_mapping is not None:
            return local_cause_net.subgraph(list(iso_mapping)).to_dot(
                node_attr=lambda n: {"label": str(n)}
            )


def get_bifxml(dot: str) -> str:
    graph = pydot.graph_from_dot_data(dot)[0]

    node_labels = {}
    for node in graph.get_nodes():
        node_name = node.get_name().strip('"')
        label = node.get("label").strip('"')
        node_labels[node_name] = label

    edge_list = []
    for edge in graph.get_edges():
        source = edge.get_source().strip('"')
        target = edge.get_destination().strip('"')
        edge_list.append(f"{node_labels[source]}->{node_labels[target]}")

    bn = gum.fastBN(";".join(edge_list))

    # Create temporary file and save BIFXML
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".bifxml", delete=False
    ) as temp_file:
        temp_path = temp_file.name

    try:
        bn.saveBIFXML(temp_path)

        # Read the content and return as string
        with open(temp_path, "r") as f:
            content = f.read()

        return content
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
