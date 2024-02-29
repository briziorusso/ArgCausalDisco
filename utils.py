import sys
import logging
from itertools import chain, combinations
from collections import defaultdict
import numpy as np
import networkx as nx
import igraph as ig
from datetime import datetime
# import cdt
# cdt.SETTINGS.rpath = '../R/R-4.1.2/bin/Rscript'
# from cdt.metrics import get_CPDAG

maxpc = False
if maxpc:
    sys.path.append("../causal-learn/")
    from causallearn.search.ConstraintBased.PC import pc
else:
    sys.path.append("cd_algorithms/")
    from PC import pc

sys.path.append("../causal-learn/tests/")
from utils_simulate_data import simulate_discrete_data, simulate_linear_continuous_data

# create logger
def logger_setup(output_file:str="causalaba"):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=f'.temp/{output_file}.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-8s %(module)-12s - %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    # format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    if len(logging.getLogger('').handlers) < 2:
        logging.getLogger('').addHandler(console)


def random_stability(seed_value=0, deterministic=True, verbose=False):
    '''
        seed_value : int A random seed
        deterministic : negatively effect performance making (parallel) operations deterministic
    '''
    if verbose:
        print('Random seed {} set for:'.format(seed_value))
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        if verbose:
            print(' - PYTHONHASHSEED (env)')
    except:
        pass
    try:
        import random
        random.seed(seed_value)
        if verbose:
            print(' - random')
    except:
        pass
    try:
        import numpy as np
        np.random.seed(seed_value)
        if verbose:
            print(' - NumPy')
    except:
        pass
    # try:
    #     import torch
    #     torch.manual_seed(seed_value)
    #     torch.cuda.manual_seed_all(seed_value)
    #     if verbose:
    #         print(' - PyTorch')
    #     if deterministic:
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    #         if verbose:
    #             print('   -> deterministic')
    # except:
        pass

def format_time(timestamp=datetime.now(), date=True, decimals=0):
    if date:
        return timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-7+decimals]
    else:
        return timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[11:-7+decimals]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def model_to_adjacency_matrix(model:list, num_of_nodes:int)->np.ndarray:
    adj_mat = np.zeros((num_of_nodes,num_of_nodes))
    for atom in model:
        if atom.name == 'arrow':
            adj_mat[int(atom.arguments[0].number)][int(atom.arguments[1].number)] = 1
    return adj_mat

def model_to_set_of_arrows(model:list, num_of_nodes:int)->set:
    arrows = set()
    for atom in model:
        if atom.name == 'arrow':
            arrows.add((atom.arguments[0].number,atom.arguments[1].number))
    return arrows

def set_of_models_to_set_of_graphs(models, n_nodes, mec_check=True):
    MECs = defaultdict(list)
    MEC_set = set()
    model_sets = set()
    # logging.info("   Checking MECs")
    for model in models:
        arrows = model_to_set_of_arrows(model, n_nodes)
        model_sets.add(frozenset(arrows))        
        if mec_check:
            adj = model_to_adjacency_matrix(model, n_nodes)
            cp_adj = dag2cpdag(adj)
            #cp_adj = get_CPDAG(adj)
            cp_adj_hashable = map(tuple, cp_adj)
            MECs[cp_adj_hashable] = list(adj.flatten())
            MEC_set.add(frozenset(cp_adj_hashable))
    logging.debug(f"   Number of MECs: {len(MEC_set)}")
    return model_sets, MECs

def extract_test_elements_from_symbol(symbol:str)->list:
    dep_type, elements = symbol.replace(").","").split("(")
    
    if "dep" in dep_type:
        X, Y, condset = elements.split(",")
        if condset == "empty":
            S = set()
        elif condset[0] == "s":
            S = set([int(e) for e in condset[1:].split("y")])
        else:
            raise ValueError(f"Unknown element {condset}")

        return int(X), S, int(Y), dep_type
    elif dep_type in ["arrow", "edge"]:
        X, Y = elements.split(",")
        return int(X), int(Y), dep_type

def find_all_d_separations_sets(G, verbose=True, debug=False):
    no_of_var = len(G.nodes)
    septests = []
    for comb in combinations(range(no_of_var), 2):
        if comb[0] != comb[1]:
            x = comb[0]
            y = comb[1]
            depth = 0
            while no_of_var-1 > depth:
                Neigh_x_noy = [f"X{k+1}" for k in range(no_of_var) if k != x and k != y]
                for S in combinations(Neigh_x_noy, depth):
                    s = set([int(s.replace('X',''))-1 for s in S])
                    s_str = 'empty' if len(S)==0 else 's'+'y'.join([str(i) for i in s])
                    if nx.algorithms.d_separated(G, {f"X{x+1}"}, {f"X{y+1}"}, set(S)):
                        logging.debug(f"X{x+1} and X{y+1} are d-separated by {S}")
                        septests.append(f"indep({x},{y},{s_str}).")
                    else:
                        # logging.info(f"X{x+1} and X{y+1} are not d-separated by {S}")
                        septests.append(f"dep({x},{y},{s_str}).")
                depth += 1
    return septests

def simulate_data_and_run_PC(G_true:nx.DiGraph, alpha:float, uc_rule:int=3, uc_priority:int=2):
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
    data = simulate_discrete_data(num_of_nodes, 10000, truth_DAG_directed_edges, 42)

    logging.info(f"Running PC algorithm...")
    cg = pc(data=data, alpha=alpha, ikb=True, uc_rule=uc_rule, uc_priority=uc_priority, verbose=False)
    
    return cg

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
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def mount_adjacency_list(adjacency_matrix):
    """
    Reads an adjacency matrix and returns the corres-
    ponding adjacency list (or adjacency map)
    """
    adjacency_list = {}
    for v1 in range(len(adjacency_matrix)):
        adjacency_list.update({v1: [v2 for v2 in range(len(adjacency_matrix[v1])) if adjacency_matrix[v1][v2] == 1]})
    return adjacency_list

def get_immoralities(adj_list):
    """
    Finds the set of immoralities in the adj_list
    """
    return [(v1, v3, v2) for v1 in adj_list for v2 in adj_list for v3 in adj_list[v1] \
            if v3 in adj_list[v2] and v1 < v2 and v2 not in adj_list[v1] and v1 not in adj_list[v2]]

def dag2skel(G):
    """Convert a DAG to a skeleton.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG

    Returns:
        G (np.ndarray): [d, d] binary adj matrix of skeleton
    """
    C = np.zeros(G.shape)
    G1 = mount_adjacency_list(G)
    edges = [(v1, v2) for v1 in range(len(G1)) for v2 in G1[v1]]
    for v1, v2 in edges:
        C[v1, v2] = -1
        C[v2, v1] = -1

    return C

def dag2cpdag(G):
    """Convert a DAG to a CPDAG.

    Args:
        G (np.ndarray): [d, d] binary adj matrix of DAG

    Returns:
        C (np.ndarray): [d, d] binary adj matrix of CPDAG
    """

    ###only leave the arrows that are part of a v-structure
    C = dag2skel(G)
    immoralities = get_immoralities(mount_adjacency_list(G))
    for v1, v3, v2 in immoralities:
        C[v1, v3] = 1
        C[v2, v3] = 1

    return C