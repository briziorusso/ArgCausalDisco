import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling

from logger import logger


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def create_random_dag(n_nodes, n_edges):
    nodes = [f'X{i}' for i in range(n_nodes)]
    edges = set()
    while len(edges) < n_edges:
        a, b = random.sample(nodes, 2)
        if (a, b) not in edges and (b, a) not in edges:
            temp_dag = nx.DiGraph(list(edges) + [(a, b)])
            if nx.is_directed_acyclic_graph(temp_dag):
                edges.add((a, b))
    return list(edges), nodes


def generate_random_cpds(model, nodes, cardinality=2):
    for node in nodes:
        parents = model.get_parents(node)
        if not parents:
            values = [[random.random()] for _ in range(cardinality)]
            total = sum(v[0] for v in values)
            values = [[v[0] / total] for v in values]
        else:
            parent_card = cardinality ** len(parents)
            values = []
            for _ in range(cardinality):
                row = [random.random() for _ in range(parent_card)]
                values.append(row)
            for col in range(parent_card):
                col_sum = sum(values[row][col] for row in range(cardinality))
                for row in range(cardinality):
                    values[row][col] /= col_sum
        cpd = TabularCPD(
            variable=node, variable_card=cardinality,
            values=values,
            evidence=parents if parents else None,
            evidence_card=[cardinality] * len(parents) if parents else None
        )
        model.add_cpds(cpd)


def generate_random_bn_data(n_nodes=5, n_edges=6, n_samples=1000, seed=42, standardise=True):
    # Set global seed
    set_global_seed(seed)

    # Create DAG and model
    edges, nodes = create_random_dag(n_nodes, n_edges)
    model = DiscreteBayesianNetwork(edges)

    # add nodes to the model
    for node in nodes:
        if node not in model.nodes():
            model.add_node(node)

    generate_random_cpds(model, nodes)
    assert model.check_model()

    # Simulate data
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=n_samples, seed=seed)
    df = df[np.sort(df.columns)]

    # Label encoding
    enc = LabelEncoder()
    df_le = df.copy()
    for var in df.columns:
        enc.fit(df[var])
        df_le[var] = enc.transform(df[var])

    # Standardize or return float matrix
    if standardise:
        df_le_s = StandardScaler().fit_transform(df_le)
    else:
        df_le_s = df_le.to_numpy().astype(float)

    # Extract true DAG structure
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(model.edges())
    B_true = nx.adjacency_matrix(G).todense()
    B_pd = pd.DataFrame(B_true, columns=G.nodes(), index=G.nodes())
    B_pd = B_pd.reindex(sorted(df.columns), axis=0)
    B_pd = B_pd.reindex(sorted(df.columns), axis=1)
    B_true = B_pd.values

    logger.debug(f"Data shape: {df_le_s.shape}")
    logger.debug(f"Number of true edges: {len(model.edges())}")
    logger.debug(f"True BN edges: {model.edges()}")
    logger.debug(f"DAG? {nx.is_directed_acyclic_graph(G)}")
    logger.debug(f"True DAG shape: {B_true.shape}, True DAG edges: {B_true.sum()}")
    logger.debug(B_pd)

    return df_le_s, B_true


if __name__ == "__main__":
    # Example usage
    n_nodes = 5
    n_edges = 6
    n_samples = 1000
    seed = 42
    df_le_s, B_true = generate_random_bn_data(n_nodes, n_edges, n_samples, seed)
