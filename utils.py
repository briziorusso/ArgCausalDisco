import sys, os
import logging
from itertools import chain, combinations
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
from datetime import datetime
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
os.environ['R_HOME'] = '../R/R-4.1.2/bin/'
### To not have the WARNING: ignoring environment value of R_HOME 
### set the verbose to False in the launch_R_script function in:
### CausalDiscoveryToolbox/cdt/utils/R.py#L155
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cdt
from cdt.metrics import get_CPDAG, SHD, SID, SHD_CPDAG, SID_CPDAG, precision_recall
cdt.SETTINGS.rpath = '../R/R-4.1.2/bin/Rscript'

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

def initial_strength(p:float, len_S:int, alpha:float, base_strength:float, num_vars:int, verbose=False)->float:
    w_S = (1-len_S/(num_vars-2))
    # w_S = 1
    if p != None:
        if p < alpha:
            initial_strength = (1-0.5/alpha*p)*w_S
        else:
            initial_strength = ((alpha-0.5*p-0.5)/(alpha-1))*w_S
    else:
        initial_strength = base_strength
    return initial_strength

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

def is_dag(B):
    """Check if a matrix is a DAG"""
    return ig.Graph.Adjacency(B.tolist()).is_dag()

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

def dag2skel(G, unique=False):
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
        if not unique:
            C[v2, v1] = -1
    return C

def dag2cpdag(G, cdt_method=False):
    """Convert a DAG to a CPDAG.

    Args:
        G (np.ndarray): [d, d] binary adj matrix of DAG

    Returns:
        C (np.ndarray): [d, d] binary adj matrix of CPDAG
    """
    assert is_dag(G), 'Input graph is not a DAG'
    ###only leave the arrows that are part of a v-structure
    C = dag2skel(G)
    immoralities = get_immoralities(mount_adjacency_list(G))
    for v1, v3, v2 in immoralities:
        C[v1, v3] = 1
        C[v3, v1] = 0
        C[v2, v3] = 1
        C[v3, v2] = 0
    if cdt_method:
        C = (C != 0).astype(int)
    return C

### Largely from TrustworthyAI repo, with some modifications and the addition of SID from cdt.metrics
class MetricsDAG(object):
    """
    Compute various accuracy metrics for B_est.
    true positive(TP): an edge estimated with correct direction.
    true nagative(TN): an edge that is neither in estimated graph nor in true graph.
    false positive(FP): an edge that is in estimated graph but not in the true graph.
    false negative(FN): an edge that is not in estimated graph but in the true graph.
    reverse = an edge estimated with reversed direction.

    fdr: (reverse + FP) / (TP + FP)
    tpr: TP/(TP + FN)
    fpr: (reverse + FP) / (TN + FP)
    shd: undirected extra + undirected missing + reverse
    nnz: TP + FP
    precision: TP/(TP + FP)
    recall: TP/(TP + FN)
    F1: 2*(recall*precision)/(recall+precision)
    gscore: max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1

    Parameters
    ----------
    B_est: np.ndarray
        [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
    B_true: np.ndarray
        [d, d] ground truth graph, {0, 1}.
    """

    def __init__(self, B_est, B_true, sid=True):
        
        if not isinstance(B_est, np.ndarray):
            raise TypeError("Input B_est is not numpy.ndarray!")

        if not isinstance(B_true, np.ndarray):
            raise TypeError("Input B_true is not numpy.ndarray!")

        self.B_est = deepcopy(B_est)
        self.B_true = deepcopy(B_true)

        self.metrics = MetricsDAG._count_accuracy(self.B_est, self.B_true, sid)

    @staticmethod
    def _count_accuracy(B_est, B_true, sid=True, decimal_num=4):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.
        sid: bool
            If True, compute SID.
        decimal_num: int
            Result decimal numbers.


        Return
        ------
        metrics: dict
            fdr: float
                (reverse + FP) / (TP + FP)
            tpr: float
                TP/(TP + FN)
            fpr: float
                (reverse + FP) / (TN + FP)
            shd: int
                undirected extra + undirected missing + reverse
            nnz: int
                TP + FP
            precision: float
                TP/(TP + FP)
            recall: float
                TP/(TP + FN)
            F1: float
                2*(recall*precision)/(recall+precision)
            gscore: float
                max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """

        ## Do not allow self loops
        if (np.diag(B_est)).any():
            raise ValueError('Graph contains self loops')
        ## Only allow 0, 1, -1
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        
        B_est_unique = deepcopy(B_est)
        # trans cpdag [0, 1] to [-1, 0, 1], -1 is undirected edge in CPDAG
        if ((B_est_unique == 1) & (B_est_unique.T == 1)).any():
            cpdag = True
            for i in range(len(B_est_unique)):
                for j in range(len(B_est_unique[i])):
                    if B_est_unique[i, j] == B_est_unique[j, i] == 1:
                        B_est_unique[i, j] = -1
                        B_est_unique[j, i] = 0
        if (B_est_unique == -1).any():  # cpdag
            cpdag = True
            ## only one entry in the pair of undirected edges should be -1
            if ((B_est_unique == -1) & (B_est_unique.T == -1)).any():
                for i in range(len(B_est_unique)):
                    for j in range(len(B_est_unique[i])):
                        if B_est_unique[i, j] == B_est_unique[j, i] == -1:
                            B_est_unique[i, j] = -1
                            B_est_unique[j, i] = 0
                assert not ((B_est_unique == -1) & (B_est_unique.T == -1)).any()
                assert not ((B_est_unique == -1) & (B_est_unique.T == -1)).any()
        else:  # dag
            cpdag = False
            if not ((B_est == 0) | (B_est == 1)).all():
                raise ValueError('B_est should take value in {0,1}')
            if not is_dag(B_est):
                raise ValueError('B_est should be a DAG')
        d = B_true.shape[0]
        
        # linear index of nonzeros
        pred_und = np.flatnonzero(B_est_unique == -1)
        pred = np.flatnonzero(B_est_unique == 1)
        cond = np.flatnonzero(B_true)
        cond_reversed = np.flatnonzero(B_true.T)
        cond_skeleton = np.concatenate([cond, cond_reversed])
        # true pos
        true_pos = np.intersect1d(pred, cond, assume_unique=True) if len(pred) > 0 else np.array([])
        # treat undirected edge favorably
        true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True) if len(pred_und) > 0 else np.array([])
        true_pos = np.concatenate([true_pos, true_pos_und])
        # false pos
        false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
        false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
        false_pos = np.concatenate([false_pos, false_pos_und])
        # reverse
        extra = np.setdiff1d(pred, cond, assume_unique=True)
        reverse = np.intersect1d(extra, cond_reversed, assume_unique=True) if len(extra) > 0 else np.array([])
        # compute ratio
        pred_size = len(pred) + len(pred_und)
        cond_neg_size = 0.5 * d * (d - 1) - len(cond)
        fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
        tpr = float(len(true_pos)) / max(len(cond), 1)
        fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
        # structural hamming distance 
        # pred_lower = np.flatnonzero(np.tril(B_est_unique + B_est.T))
        # cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
        # extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
        # missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
        # shd = len(extra_lower) + len(missing_lower) + len(reverse)        
        ### this, although standard in some packages,
        ### treats undirected edge as a present edge in the CPDAG 
        ### Replacing with SHD from cdt
        if cpdag:
            shd = MetricsDAG._cal_SHD_CPDAG(B_est, B_true)
        else:
            shd = SHD(B_true, B_est, False)

        W_p = pd.DataFrame(B_est_unique)
        W_true = pd.DataFrame(B_true)

        # gscore = MetricsDAG._cal_gscore(W_p, W_true)
        precision, recall, F1 = MetricsDAG._cal_precision_recall(W_p, W_true)

        mt = {'nnz': pred_size, 'fdr': fdr, 'tpr': tpr, 'fpr': fpr,  
              'precision': precision, 'recall': recall, 'F1': F1,#, 'gscore': gscore
              'shd': shd}

        for i in mt:
            mt[i] = round(mt[i], decimal_num)   

        if sid and not cpdag:
            mt['sid'] = MetricsDAG._cal_SID(B_est, B_true)
        elif sid and cpdag:
            mt['sid'] = MetricsDAG._cal_SID_CPDAG(B_est, B_true)
       
        return mt

    @staticmethod
    def _cal_gscore(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        score: float
            max(0, (TP-FP))/(TP+FN), A score ranges from 0 to 1
        """
        
        num_true = W_true.sum(axis=1).sum()
        score = np.nan
        if num_true!=0:
            # true_positives
            num_tp =  (W_p + W_true).map(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
            # False Positives + Reversed Edges
            num_fn_r = (W_p - W_true).map(lambda elem:1 if elem==1 else 0).sum(axis=1).sum()
            score = np.max((num_tp-num_fn_r,0))/num_true
            
        return score

    @staticmethod
    def _cal_precision_recall(W_p, W_true):
        """
        Parameters
        ----------
        W_p: pd.DataDrame
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        W_true: pd.DataDrame
            [d, d] ground truth graph, {0, 1}.
        
        Return
        ------
        precision: float
            TP/(TP + FP)
        recall: float
            TP/(TP + FN)
        F1: float
            2*(recall*precision)/(recall+precision)
        """

        assert(W_p.shape==W_true.shape and W_p.shape[0]==W_p.shape[1])
        if (W_p == -1).any().any():
            W_p = pd.DataFrame((W_p != 0).astype(int))
        if (W_true == -1).any().any():
            W_true = pd.DataFrame((W_true != 0).astype(int))

        TP = (W_p + W_true).map(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
        TP_FP = W_p.sum(axis=1).sum()
        TP_FN = W_true.sum(axis=1).sum()
        precision = TP/TP_FP
        recall = TP/TP_FN
        F1 = 2*(recall*precision)/(recall+precision)
        
        return precision, recall, F1
    
    @staticmethod
    def _cal_SID(B_est, B_true):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1}.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        SID: float
            Structural Intervention Distance
        """
        assert is_dag(B_true), 'B_true should be a DAG'
        assert is_dag(B_est), 'B_est should be a DAG'
        return SID(B_true, B_est).flat[0]
    
    @staticmethod
    def _cal_SHD_CPDAG(B_est, B_true):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}.

        Return
        ------
        SHD: int
            Structural Hamming Distance of CPDAG
        """
        assert is_dag(B_true), 'B_true should be a DAG'
        ### treat undirected edge as a present edge in the CPDAG
        ### the difference will be in the missed immoralities
        return SHD(dag2cpdag(B_true,True), (B_est != 0).astype(int), False)

    @staticmethod
    def _cal_SID_CPDAG(B_est, B_true):
        """
        Parameters
        ----------
        B_est: np.ndarray
            [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG.
        B_true: np.ndarray
            [d, d] ground truth graph, {0, 1}. Needs to be a DAG.

        Return
        ------
        SID_CPDAG_low: float
            Lower bound of Structural Intervention Distance
        SID_CPDAG_high: float
            Upper bound of Structural Intervention Distance
        """
        assert is_dag(B_true), 'B_true should be a DAG'
        ### treat undirected edge as a present edge in the CPDAG
        ### the difference will be in the missed immoralities
        SID_CPDAG_low, SID_CPDAG_high = [a.flat[0] for a in SID_CPDAG(B_true, (B_est != 0).astype(int))]
        return SID_CPDAG_low, SID_CPDAG_high
