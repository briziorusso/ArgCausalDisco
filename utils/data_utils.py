import os
import numpy as np
import pandas as pd
import networkx as nx
import logging
import igraph as ig
from pgmpy.readwrite import BIFReader
from PIL import Image
from sklearn.preprocessing import LabelEncoder, StandardScaler
from cd_algorithms.models import pc
from utils.helpers import random_stability
from utils.graph_utils import is_dag

BIF_FOLDER_MAP = {
    'cancer': 'small',
    'earthquake': 'small',
    'survey': 'small',
    'asia': 'small',
    'sachs': 'small',
    'alarm': 'medium',
    'child': 'medium',
    'insurance': 'medium',
    'hailfinder': 'large',
    'hepar2': 'large'
}

def load_bn_from_BIF(main_data_path, data_folder='bayesian', dataset_name='child', seed=1, verbose=False):
    bif_file = os.path.join(main_data_path, data_folder, BIF_FOLDER_MAP[dataset_name], dataset_name + '.bif', dataset_name + '.bif')
    image_file = os.path.join(main_data_path, data_folder, BIF_FOLDER_MAP[dataset_name], dataset_name + '.png')

    if verbose:
        logging.debug(f'Loading graph from {bif_file}')

    random_stability(seed)
    reader = BIFReader(bif_file)
    model = reader.get_model()

    if os.path.exists(image_file):
        if verbose:
            logging.debug(f'Loading graph image from {image_file}')
        model.image = Image.open(image_file)

    # Take the leaves as features
    __FEATURES = model.get_leaves()
    __ROOTS = model.get_roots()
    __NODES = model.nodes()
    __NOT_FEATURES = list({node for node in __NODES if node not in __FEATURES and node not in __ROOTS})

    if verbose:
        logging.debug(f'Nodes: {__NODES} ({len(__NODES)})')
        logging.debug(f'Features/Leaves: {__FEATURES} ({len(__FEATURES)})')
        logging.debug(f'Roots: {__ROOTS} ({len(__ROOTS)})')
        logging.debug(f'Intermediate (non-roots/non-leaves): {__NOT_FEATURES} ({len(__NOT_FEATURES)})')

    return model

def load_bnlearn_data_dag(dataset_name, data_path, sample_size, seed=1, standardise=True, print_info=False):
    assert dataset_name in BIF_FOLDER_MAP.keys(), "dataset name not recognised"
    ##Load Bayesian Network
    logging.info(f"==================Loading {dataset_name} dataset==================")
    random_stability(seed)
    bn = load_bn_from_BIF(main_data_path=data_path, dataset_name=dataset_name, seed=seed)
    ##Simulate data from BN
    df = bn.simulate(sample_size, seed=seed)
    ##Preprocess categorical data
    df = df[np.sort(df.columns)] ##Sort columns alphabetically to match DAG
    enc = LabelEncoder()
    df_le = df.copy()
    for var in df.columns:
        enc.fit(df[var])
        df_le[var] = enc.transform(df[var])
    if standardise:
        df_le_s = StandardScaler().fit_transform(df_le)
    else:
        df_le_s = df_le.to_numpy().astype(float)

    ##Extract true DAG from Bayesian network
    G = nx.from_edgelist(list(bn.edges()), create_using=nx.DiGraph)
    B_true = nx.adjacency_matrix(G).todense()
    B_pd = pd.DataFrame(B_true, columns=G.nodes(), index=G.nodes())
    ##Sort columns alphabetically to match data
    B_pd = B_pd.reindex(sorted(df.columns), axis=0)
    B_pd = B_pd.reindex(sorted(df.columns), axis=1)
    B_true = B_pd.values

    if print_info:
        logging.info(f"Data shape: {df_le_s.shape}")
        logging.info(f"Number of true edges: {len(bn.edges())}")
        logging.info(f"True BN edges: {bn.edges()}")
        logging.info(f"DAG? {nx.is_directed_acyclic_graph(G)}")
        logging.info(f"True DAG shape: {B_true.shape}, True DAG edges: {B_true.sum()}")
        logging.info(B_pd)

    return df_le_s, B_true



### From notears repo: https://github.com/xunzheng/notears
def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

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

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert is_dag(B_perm)
    return B_perm


#### From Causal-lean repo:

def simulate_discrete_data(
        num_of_nodes,
        sample_size,
        truth_DAG_directed_edges,
        random_seed=None):
    from pgmpy.models.BayesianNetwork import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.sampling import BayesianModelSampling

    def _simulate_cards():
        '''
        why we need this: to calculate cpd of a node with k parents,
            the conditions to be enumerated is the production of these k parents' cardinalities
            which will be exponentially slow w.r.t. k.
            so we want that, if a node has many parents (large k), these parents' cardinalities should be small

        denote peers_num: peers_num[i, j] = k (where k>0),
            means that there are k parents pointing to node i, and j is among these k parents.
        max_peers = peers_num.max(axis=0): the larger max_peers[j], the smaller card[j] should be.
        '''
        MAX_ENUMERATION_COMBINATION_NUM = 20
        in_degrees = adjacency_matrix.sum(axis=1)
        peers_num = in_degrees[:, None] * adjacency_matrix
        max_peers_num = peers_num.max(axis=0)
        max_peers_num[max_peers_num == 0] = 1 # to avoid division by 0 (for leaf nodes)
        cards = [np.random.randint(2, 1 + max(2, MAX_ENUMERATION_COMBINATION_NUM ** (1. / mpn)))
                    for mpn in max_peers_num]
        return cards

    def _random_alpha():
        DIRICHLET_ALPHA_LOWER, DIRICHLET_ALPHA_UPPER = 1., 5.
        return np.random.uniform(DIRICHLET_ALPHA_LOWER, DIRICHLET_ALPHA_UPPER)

    if random_seed is not None:
        state = np.random.get_state() # save the current random state
        np.random.seed(random_seed)  # set the random state to 42 temporarily, just for the following lines
    adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes))
    adjacency_matrix[tuple(zip(*truth_DAG_directed_edges))] = 1
    adjacency_matrix = adjacency_matrix.T

    cards = _simulate_cards()
    bn = BayesianNetwork(truth_DAG_directed_edges)  # so isolating nodes will echo error
    for node in range(num_of_nodes):
        if node not in bn.nodes(): bn.add_node(node) # add node if it is isolated
        parents = np.where(adjacency_matrix[node])[0].tolist()
        parents_card = [cards[prt] for prt in parents]
        rand_ps = np.array([np.random.dirichlet(np.ones(cards[node]) * _random_alpha()) for _ in
                            range(int(np.prod(parents_card)))]).T.tolist()

        cpd = TabularCPD(node, cards[node], rand_ps, evidence=parents, evidence_card=parents_card)
        bn.add_cpds(cpd)
    inference = BayesianModelSampling(bn)
    df = inference.forward_sample(size=sample_size, show_progress=False)
    topo_order = list(map(int, df.columns))
    topo_index = [-1] * len(topo_order)
    for ind, node in enumerate(topo_order): topo_index[node] = ind
    data = df.to_numpy()[:, topo_index].astype(np.int64)

    if random_seed is not None: np.random.set_state(state) # restore the random state
    return data

def simulate_linear_continuous_data(
        num_of_nodes,
        sample_size,
        truth_DAG_directed_edges,
        noise_type='gaussian',  # currently: 'gaussian' or 'exponential'
        random_seed=None,
        linear_weight_minabs=0.5,
        linear_weight_maxabs=0.9,
        linear_weight_netative_prob=0.5):
    if random_seed is not None:
        state = np.random.get_state() # save the current random state
        np.random.seed(random_seed)  # set the random state to 42 temporarily, just for the following lines
    adjacency_matrix = np.zeros((num_of_nodes, num_of_nodes))
    adjacency_matrix[tuple(zip(*truth_DAG_directed_edges))] = 1
    adjacency_matrix = adjacency_matrix.T
    weight_mask = np.random.uniform(linear_weight_minabs, linear_weight_maxabs, (num_of_nodes, num_of_nodes))
    weight_mask[np.unravel_index(np.random.choice(np.arange(weight_mask.size), replace=False,
                size=int(weight_mask.size * linear_weight_netative_prob)), weight_mask.shape)] *= -1.
    adjacency_matrix = adjacency_matrix * weight_mask
    mixing_matrix = np.linalg.inv(np.eye(num_of_nodes) - adjacency_matrix)
    if noise_type == 'gaussian':
        exogenous_noise = np.random.normal(0, 1, (num_of_nodes, sample_size))
    elif noise_type == 'exponential':
        exogenous_noise = np.random.exponential(1, (num_of_nodes, sample_size))
    else:
        raise NotImplementedError
    data = (mixing_matrix @ exogenous_noise).T  # in shape (sample_size, num_of_nodes)
    if random_seed is not None: np.random.set_state(state) # restore the random state
    return data


### From notears repo: https://github.com/xunzheng/notears
def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

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

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert is_dag(B_perm)
    return B_perm


def simulate_data_and_run_PC(G_true:nx.DiGraph, alpha:float, uc_rule:int=3, uc_priority:int=2, stable:bool=True, seed:int=42):
    """A function to simulate data and run PC algorithm

    Args:
        G_true (nx.DiGraph): the true graph
        alpha (float): significance level
        uc_rule (int, optional): unshielded collider rule. Defaults to 3.
        uc_priority (int, optional): unshielded collider priority. Defaults to 2.

    Returns:
        causallearn.CausalGraph: the learned graph
    """

    num_of_nodes = len(G_true.nodes)
    # num_of_nodes = max(sum(truth_DAG_directed_edges, ())) + 1

    truth_DAG_directed_edges = set([(int(e[0].replace("X",""))-1,int(e[1].replace("X",""))-1)for e in G_true.edges])

    logging.info(f"Simulating data with {num_of_nodes} nodes, {10000} samples...")
    data = simulate_discrete_data(num_of_nodes, 10000, truth_DAG_directed_edges, seed)

    logging.info(f"Running PC algorithm...")
    cg = pc(data=data, alpha=alpha, ikb=True, uc_rule=uc_rule, uc_priority=uc_priority, stable=stable, verbose=False)
    
    return data, cg