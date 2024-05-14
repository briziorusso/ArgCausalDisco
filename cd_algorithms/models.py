import sys
import os
import yaml
import argparse
import time
import numpy as np
import networkx as nx
import pandas as pd
import torch
import pydot
import logging

# from tqdm.auto import tqdm
# from collections import defaultdict
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import ParameterGrid
import gc
gc.set_threshold(0,0,0)

from abapc import ABAPC

try:
    sys.path.append("cd_algorithms/")
    from PC import pc
except:
    logging.info('PC not installed')
    sys.path.append('../causal-learn/')
    from causallearn.search.ConstraintBased.PC import pc

### Import from cloned repos
sys.path.append("../notears/")
# sys.path.append("../causal-learn/")
# from causallearn.search.ConstraintBased.PC import pc
# sys.path.append('../trustworthyAI/gcastle/')
# sys.path.append('../CausalDiscoveryToolbox/')
# sys.path.append('../py-causal/src/')

from notears import utils as nt_utils
# from baselines.notears_gpu import *
# from notears.nonlinear import NotearsMLP, notears_nonlinear
# from notears.linear import notears_linear

# from causallearn.search.ScoreBased.GES import ges

from utils import random_stability, get_freer_gpu

from notears import utils as nt_utils

try:
    from castle.algorithms import MCSL#, GraNDAG, NotearsNonlinear, Notears
except:
    logging.info('Castle not installed')
    sys.path.append('../trustworthyAI/gcastle/')
    from castle.algorithms import MCSL#, GraNDAG, NotearsNonlinear, Notears
    
try:
    import cdt
    cdt.SETTINGS.rpath = '../R/R-4.1.2/bin/Rscript'
    from cdt.causality.graph import CAM
    os.environ['R_HOME'] = '../R/R-4.1.2/bin/'
except:
    logging.info('CDT or R components not installed')
    sys.path.append('../CausalDiscoveryToolbox/')
    import cdt
    cdt.SETTINGS.rpath = '../R/R-4.1.2/bin/Rscript'
    os.environ['R_HOME'] = '../R/R-4.1.2/bin/'

import warnings
warnings.filterwarnings("ignore")

def run_method(X, 
               method:str, 
               seed:int, 
               debug:bool=False, 
               selection:str='bot', 
               priority:int=2, 
               test_name:str='kci', 
               test_alpha:float=0.01, 
               device:int=0,
               scenario:str=''
               ):
    """
    Runs the causal discovery method specified by method on the data X

    Parameters
    ----------
    X : np.array or pd.DataFrame to run the method on
    method : str Name of the causal discovery method to run
    seed : int Random seed
    debug : bool Whether to logging.info debug statements
    selection : str Selection rule for SPC (default: 'bot')
    priority : int Priority rule for SPC (default: 2)
    test_name : str Name of the independence test to use for PC
    test_alpha : float Significance level for independence test (default: 0.01)
    device : str Device to run Notears on

    Returns
    -------
    W_est : np.array
        Estimated causal graph
    elapsed : float
        Time taken to run the method

    """

    ##---------MODELS--------------
    if method == 'nt':
        if device == '':
            device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Running on: {device}")
        notears_from = 'notears'
        if notears_from == 'castle':
            start = time.time()
            random_stability(seed)
            fitted = NotearsNonlinear()
            fitted.device = device
            fitted.learn(X)
            W_est = fitted.causal_matrix
        else:
            sys.path.append("../notears/")
            from notears.nonlinear import NotearsMLP, notears_nonlinear
            if device == '':
                device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else "cpu")

            start = time.time()
            random_stability(seed)
            fitted = NotearsMLP(dims=[X.shape[1], 10, 1], bias=True, device=device)
            W_est = notears_nonlinear(fitted, X, lambda1=0.01, lambda2=0.01)

        elapsed = time.time() - start
        logging.info(f'Time taken for Notears: {round(elapsed,2)}s')

    elif method == 'nt_lin':
        start = time.time()
        random_stability(seed)
        fitted = Notears()
        fitted.learn(X)
        W_est = fitted.causal_matrix

        elapsed = time.time() - start
        logging.info(f'Time taken for Notears: {round(elapsed,2)}s')

    elif method == 'mcsl':
        if device == '':
            device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Running on: {device}")

        start = time.time()
        random_stability(seed)
        fitted = MCSL(model_type='nn',
                iter_step=100,
                rho_thresh=1e20,
                init_rho=1e-5,
                rho_multiply=10,
                graph_thresh=0.5,
                l1_graph_penalty=2e-3,
                random_seed=seed,
                device_type='gpu',
                device_ids=device)
        
        random_stability(seed)
        fitted.learn(X)
        W_est = fitted.causal_matrix

        elapsed = time.time() - start
        logging.info(f'Time taken for MCSL: {round(elapsed,2)}s')

    elif method == 'grandag':
        if device == '':
            device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Running on: {device}")

        d = {'model_name': 'NonLinGauss', 'nonlinear': 'leaky-relu', 'optimizer': 'sgd', 'norm_prod': 'paths', 'device_type': 'gpu'}
        start = time.time()
        random_stability(seed)
        fitted = GraNDAG(input_dim=X.shape[1], iterations=1000, device_type='gpu', device_ids=device)
        
        random_stability(seed)
        fitted.learn(X)
        W_est = fitted.causal_matrix

        elapsed = time.time() - start
        logging.info(f'Time taken for GraN-DAG: {round(elapsed,2)}s')

    elif method == 'pc':
        random_stability(seed)
        ## uc_priority=3: prioritize stronger colliders
        fitted = pc(data=X, alpha=test_alpha, indep_test=test_name, uc_rule=0, uc_priority=2, show_progress=False, 
                    verbose=False)
        # fitted.draw_pydot_graph()
        W_est = fitted.G.graph.T
        elapsed = fitted.PC_elapsed
        logging.info(f'Time taken for PC: {round(elapsed,2)}s')

    elif method == 'cpc':
        random_stability(seed)
        fitted = pc(data=X, alpha=test_alpha, indep_test=test_name, uc_rule=4, uc_priority=priority, 
                    selection=selection, show_progress=False, verbose=debug)
        # fitted.draw_pydot_graph()
        W_est = fitted.G.graph.T
        elapsed = fitted.PC_elapsed
        logging.info(f'Time taken for CPC: {round(elapsed,2)}s')

    elif method == 'mpc':
        random_stability(seed)
        fitted = pc(data=X, alpha=test_alpha, indep_test=test_name, uc_rule=5, uc_priority=priority, 
                    selection=selection, show_progress=False, verbose=debug)
        # fitted.draw_pydot_graph()
        W_est = fitted.G.graph.T
        elapsed = fitted.PC_elapsed
        logging.info(f'Time taken for MPC: {round(elapsed,2)}s')

    elif method == 'pc_max':
        random_stability(seed)
        fitted = pc(data=X, alpha=test_alpha, indep_test=test_name, uc_rule=1, uc_priority=3, show_progress=False, 
                    verbose=False)
        # fitted.draw_pydot_graph()
        W_est = fitted.G.graph.T
        elapsed = fitted.PC_elapsed
        logging.info(f'Time taken for max-PC: {round(elapsed,2)}s')

    elif method == 'abapc':
        random_stability(seed)
        start = time.time()
        W_est = ABAPC(data=X, alpha=test_alpha, indep_test=test_name, scenario=scenario)
        elapsed = time.time() - start
        logging.info(f'Time taken for ABAPC: {round(elapsed,2)}s')

    elif method == 'ges':
        random_stability(seed)
        start = time.time()
        fitted = ges(X, verbose=False)
        elapsed = time.time() - start
        # model4.draw_pydot_graph()
        W_est = fitted['G'].graph
        logging.info(f'Time taken for GES: {round(elapsed,2)}s')

    elif method == 'fgs':
        from pycausal.pycausal import pycausal as pyc
        
        start = time.time()

        jm = pyc()
        try:
            jm.start_vm()
        except: 
            pass

        from pycausal import search as s               
        fitted = s.tetradrunner()
        random_stability(seed)
        try:
            fitted.run(algoId = 'fges', dfs = pd.DataFrame(X, columns=[f'X{c}' for c in range(1, X.shape[1]+1)]), scoreId = 'sem-bic', dataType = 'continuous',
                    maxDegree = -1, faithfulnessAssumed = True, verbose = False)

            graph = fitted.getTetradGraph()

            dot_str = jm.tetradGraphToDot(graph)
            graphs = pydot.graph_from_dot_data(dot_str)

            W_est = nx.adjacency_matrix(nx.nx_pydot.from_pydot(graphs[0])).todense()

            if W_est.shape[0] != X.shape[1]:
                ### If the graph is not fully connected, we need to add edges to make it so
                logging.debug('Graph is not fully connected, adding edges')
                g = nx.nx_pydot.from_pydot(graphs[0])
                g.add_nodes_from([f"X{d}" for d in range(1, X.shape[1]+1)])
                W_est = nx.adjacency_matrix(g).todense()            

        except:
            logging.debug("FGES failed, returning None")
            W_est = None
        elapsed = time.time() - start

        logging.info(f'Time taken for FGS: {round(elapsed,2)}s')

    elif method == 'cam':
        random_stability(seed)
        start = time.time()
        fitted = CAM()
        try:
            W_est = nx.adjacency_matrix(fitted.predict(pd.DataFrame(X))).todense()
        except:
            logging.debug("CAM failed, returning None")
            W_est = None
        elapsed = time.time() - start
        logging.info(f'Time taken for CAM: {round(elapsed,2)}s')

    return W_est, elapsed
