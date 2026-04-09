"""
run all existing methods on random graphs: random, mpc, nt, fgs for up 20 nodes.
"""

import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from pathlib import Path
from argparse import ArgumentParser

from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.cd_algorithms.models import run_method
from ArgCausalDisco.utils.graph_utils import DAGMetrics, dag2cpdag
from ArgCausalDisco.utils.helpers import random_stability, logger_setup
from ArgCausalDisco.utils.data_utils import simulate_dag
from src.utils.gen_random_nx import generate_random_bn_data

from src.utils.configure_r import configure_r
from logger import logger
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED

RESULTS_DIR = Path('results/existing/random_graphs/')
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
LOAD_SAVED = False

N_NODES = [3, 4, 5, 6, 7, 8, 10, 15, 20]
N_NODES_DOUBLE = [10, 15, 20]
N_NODES_ABAPC = [3, 4, 5, 6, 7]


model_list = [
    'random',
    'mpc',
    'nt',
    'abapc',
    'fgs'
]

names_dict = {'pc': 'PC',
              'pc_max': 'Max-PC',
              'fgs': 'FGS',
              'spc': 'Shapley-PC',
              'mpc': 'MPC',
              'cpc': 'CPC',
              'abapc': 'ABAPC (Ours)',
              'cam': 'CAM',
              'nt': 'NOTEARS-MLP',
              'mcsl': 'MCSL-MLP',
              'ges': 'GES',
              'random': 'Random'}


def parse_args():
    parser = ArgumentParser(description="Run causal discovery methods on bnlearn datasets")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each method')
    parser.add_argument('--sample-size', type=int, default=SAMPLE_SIZE, help='Sample size for each run')
    parser.add_argument('--device', type=int, default=0, help='Device to use for computation')
    return parser.parse_args()


def main(n_runs=50, sample_size=SAMPLE_SIZE, device=0):
    # configure R settings for the cdt package
    configure_r()
    version = f'bnlearn_{n_runs}rep'
    mt_path = RESULTS_DIR / f"all_existing_methods_metrics_dag.csv"
    mt_cpdag_path = RESULTS_DIR / f"all_existing_methods_metrics_cpdag.csv"

    # setup previous causal disco codebase logger
    logger_setup(str(RESULTS_DIR / f'log_all_methods.log'))

    res_columns = ['nnz', 'fdr', 'tpr', 'fpr', 'precision', 'recall', 'F1', 'shd', 'sid']
    if LOAD_SAVED:
        try:
            mt_res = pd.read_csv(mt_path)
            mt_res_cpdag = pd.read_csv(mt_cpdag_path)
            logger.info(f"Loaded previous results from {mt_path} and {mt_cpdag_path}")
        except FileNotFoundError:
            logger.warning(f"Previous results not found at {mt_path} or {mt_cpdag_path}, starting fresh.")
            mt_res = pd.DataFrame()
            mt_res_cpdag = pd.DataFrame()
    else:
        mt_res = pd.DataFrame()
        mt_res_cpdag = pd.DataFrame()

    for n_nodes in tqdm(N_NODES, desc="tqdm # nodes"):
        if n_nodes in N_NODES_DOUBLE:
            edges_to_iterate = [n_nodes, n_nodes*2]
        else:
            edges_to_iterate = [n_nodes]
        for n_edges in edges_to_iterate:
            for method in tqdm(model_list, desc="tqdm Methods"):

                if method == 'abapc' and n_nodes not in N_NODES_ABAPC:
                    logger.info(f"Skipping {method} on #nodes = {n_nodes} as it is not applicable.")
                    continue

                random_stability(SEED)
                seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
                logger.info(f"Running {method} on |V|={n_nodes} |E|={n_edges} for {n_runs} times with seeds {seeds_list}")

                for seed in tqdm(seeds_list, desc="tqdm Seeds"):
                    X_s, B_true = generate_random_bn_data(
                            n_nodes=n_nodes,
                            n_edges=n_edges,
                            n_samples=sample_size,
                            seed=seed,
                            standardise=True
                    )
                    logger.info(f"True graph {B_true}")
                    if method == 'random':
                        random_stability(seed)
                        start = time.time()
                        B_est = simulate_dag(d=B_true.shape[1], s0=B_true.sum().astype(int), graph_type='ER')
                        elapsed = time.time() - start
                        mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                        mt_dag = DAGMetrics(B_est, B_true).metrics
                    elif method in model_list:
                        W_est, elapsed = run_method(X_s,
                                                    method,
                                                    seed,
                                                    test_alpha=ALPHA,
                                                    test_name=INDEP_TEST,
                                                    device=device,
                                                    scenario=f"{method}_{version}_nodes{n_nodes}_edges{n_edges}")
                        if 'Tensor' in str(type(W_est)):
                            W_est = np.asarray([list(i) for i in W_est])
                        logger_setup(str(RESULTS_DIR / f'log_all_methods.log'), continue_logging=True)
                        if W_est is None:
                            mt_cpdag = {col: np.nan for col in res_columns}
                            mt_dag = {col: np.nan for col in res_columns}
                        else:
                            try:
                                B_est = (W_est != 0).astype(int)
                                mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                            except Exception as e:
                                mt_cpdag = {col: np.nan for col in res_columns}
                            try:
                                B_est = (W_est > 0).astype(int)
                                mt_dag = DAGMetrics(B_est, B_true).metrics
                            except Exception as e:
                                mt_dag = {col: np.nan for col in res_columns}
                    else:
                        raise ValueError(f"Method {method} not recognized")

                    # calculate metrics
                    logger.info({'n_nodes': n_nodes,  'n_edges': n_edges, 'model': names_dict[method], 'elapsed': elapsed, **mt_dag})
                    logger.info({'n_nodes': n_nodes,  'n_edges': n_edges, 'model': names_dict[method], 'elapsed': elapsed, **mt_cpdag})

                    mt_res = pd.concat([mt_res, 
                                        pd.DataFrame([{'n_nodes': n_nodes,
                                                    'n_edges': n_edges, 
                                                    'model': names_dict[method], 
                                                    'seed': seed,
                                                    'elapsed': elapsed, 
                                                    **mt_dag}])
                                        ], ignore_index=True)
                    if type(mt_cpdag['sid']) == tuple:
                        mt_sid_low = mt_cpdag['sid'][0]
                        mt_sid_high = mt_cpdag['sid'][1]
                    else:
                        mt_sid_low = mt_cpdag['sid']
                        mt_sid_high = mt_cpdag['sid']
                    mt_cpdag.pop('sid')
                    mt_cpdag['sid_low'] = mt_sid_low
                    mt_cpdag['sid_high'] = mt_sid_high

                    mt_res_cpdag = pd.concat([mt_res_cpdag, 
                                                pd.DataFrame([{'n_nodes': n_nodes,
                                                            'n_edges': n_edges, 
                                                            'model': names_dict[method], 
                                                            'seed': seed,
                                                            'elapsed': elapsed, 
                                                            **mt_cpdag}])
                                                ], ignore_index=True)
                    # cache intermediate results
                    mt_res.to_csv(mt_path, index=False)
                    mt_res_cpdag.to_csv(mt_cpdag_path, index=False)

    if 'fgs' in model_list:
        # to properly stop the JVM for py-causal
        from pycausal.pycausal import pycausal as pyc
        jm = pyc()
        try:
            jm.stop_vm()
        except:
            pass

if __name__ == "__main__":
    args = parse_args()
    main(n_runs=args.n_runs,
         sample_size=args.sample_size,
         device=args.device)
