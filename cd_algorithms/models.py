"""Copyright 2024 Fabrizio Russo, Department of Computing, Imperial College London

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__author__ = "Fabrizio Russo"
__email__ = "fabrizio@imperial.ac.uk"
__copyright__ = "Copyright (c) 2024 Fabrizio Russo"

import sys
import os
import time
import networkx as nx
import pandas as pd
import torch
import pydot
import logging
import gc
gc.set_threshold(0,0,0)
from abapc import ABAPC
from cd_algorithms.PC import pc
from utils.helpers import random_stability, get_freer_gpu

try:
    from castle.algorithms import MCSL, GraNDAG, NotearsNonlinear, Notears
except:
    logging.info('Castle not installed')
    sys.path.append('../trustworthyAI/gcastle/')
    from castle.algorithms import MCSL, GraNDAG, NotearsNonlinear, Notears

notears_from = 'notears' ## 'castle' or 'notears'
try:
    from notears.nonlinear import NotearsMLP, notears_nonlinear
except:
    sys.path.append('../notears/')
    from notears.nonlinear import NotearsMLP, notears_nonlinear
  
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
               selection:str='neg', 
               priority:int=2, 
               test_name:str='kci', 
               scenario:str='',
               test_alpha:float=0.01,
               extra_tests:bool=False,
               device:str=''
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
        if notears_from == 'castle':
            start = time.time()
            random_stability(seed)
            fitted = NotearsNonlinear()
            fitted.device = device
            fitted.learn(X)
            W_est = fitted.causal_matrix
        else:
            start = time.time()
            random_stability(seed)
            fitted = NotearsMLP(dims=[X.shape[1], 10, 1], bias=True)#, device=device)
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

    elif method == 'spc':
        random_stability(seed)
        fitted = pc(data=X, alpha=test_alpha, indep_test=test_name, uc_rule=3, uc_priority=priority, 
                    selection=selection, extra_tests=extra_tests, show_progress=False, verbose=debug)
        # fitted.draw_pydot_graph()
        W_est = fitted.G.graph.T ### CausalLearn PC returns the transpose of the adjacency matrix
        elapsed = fitted.PC_elapsed
        logging.info(f'Time taken for SPC: {round(elapsed,2)}s')

    elif method == 'abapc':
        random_stability(seed)
        start = time.time()
        W_est = ABAPC(data=X, alpha=test_alpha, indep_test=test_name, scenario=scenario, S_weight=True)
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
