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

from src.abapc import get_arrow_sets, get_best_model
from src.utils.bn_utils import get_dataset
from src.utils.configure_r import configure_r
from logger import logger
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


RESULTS_DIR = Path('results/extension_based_semantics')
RESULTS_DIR.mkdir(exist_ok=True)
LOAD_SAVED = False

dataset_list = [
    'cancer',
    'earthquake',
    'survey',
    # 'asia'
]
model_list = [
    'pure_abapc',
    'random',
    'mpc',
    'abapc',
    'nt',
    # ,'fgs' TODO: make this method work
]

names_dict = {'pc': 'PC',
              'pc_max': 'Max-PC',
              'fgs': 'FGS',
              'spc': 'Shapley-PC',
              'mpc': 'MPC',
              'cpc': 'CPC',
              'abapc': 'ABAPC (Ours)',
              'pure_abapc': 'ABAPC (ASPforABA)',
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
    mt_path = RESULTS_DIR / f"stored_results_{version}.csv"
    mt_cpdag_path = RESULTS_DIR / f"stored_results_{version}_cpdag.csv"

    # setup previous causal disco codebase logger
    logger_setup(str(RESULTS_DIR / f'log_{version}.log'))

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

    for dataset_name in tqdm(dataset_list, desc="tqdm Datasets"):
        for method in tqdm(model_list, desc="tqdm Methods"):
            random_stability(SEED)
            seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
            logger.info(f"Running {method} on {dataset_name} for {n_runs} times with seeds {seeds_list}")

            method_res = []
            method_res_cpdag = []
            for seed in tqdm(seeds_list, desc="tqdm Seeds"):
                X_s, B_true = get_dataset(dataset_name,
                                          seed=seed,
                                          sample_size=sample_size)
                if method == 'random':
                    random_stability(seed)
                    start = time.time()
                    B_est = simulate_dag(d=B_true.shape[1], s0=B_true.sum().astype(int), graph_type='ER')
                    elapsed = time.time() - start
                    mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                    mt_dag = DAGMetrics(B_est, B_true).metrics
                elif method in model_list:
                    if method == 'pure_abapc':
                        start = time.time()
                        stable_arrow_sets, cg, _, _ = get_arrow_sets(X_s,
                                                                      seed=seed,
                                                                      alpha=ALPHA,
                                                                      indep_test=INDEP_TEST)
                        _, W_est, _ = get_best_model(stable_arrow_sets,
                                                     n_nodes=X_s.shape[1],
                                                     cg=cg,
                                                     alpha=ALPHA)
                        elapsed = time.time() - start
                    else:
                        W_est, elapsed = run_method(X_s,
                                                    method,
                                                    seed,
                                                    test_alpha=ALPHA,
                                                    test_name=INDEP_TEST,
                                                    device=device,
                                                    scenario=f"{method}_{version}_{dataset_name}")
                    if 'Tensor' in str(type(W_est)):
                        W_est = np.asarray([list(i) for i in W_est])
                    logger_setup(str(RESULTS_DIR / f'log_{version}.log'), continue_logging=True)
                    if W_est is None:
                        mt_cpdag = {col: np.nan for col in res_columns}
                        mt_dag = {col: np.nan for col in res_columns}
                    else:
                        B_est = (W_est != 0).astype(int)
                        mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                        B_est = (W_est > 0).astype(int)
                        mt_dag = DAGMetrics(B_est, B_true).metrics
                else:
                    raise ValueError(f"Method {method} not recognized")

                # calculate metrics
                logger.info({'dataset': dataset_name, 'model': names_dict[method], 'elapsed': elapsed, **mt_dag})
                logger.info({'dataset': dataset_name, 'model': names_dict[method], 'elapsed': elapsed, **mt_cpdag})

                method_res.append({'dataset': dataset_name, 'model': names_dict[method], 'elapsed': elapsed, **mt_dag})
                if type(mt_cpdag['sid']) == tuple:
                    mt_sid_low = mt_cpdag['sid'][0]
                    mt_sid_high = mt_cpdag['sid'][1]
                else:
                    mt_sid_low = mt_cpdag['sid']
                    mt_sid_high = mt_cpdag['sid']
                mt_cpdag.pop('sid')
                mt_cpdag['sid_low'] = mt_sid_low
                mt_cpdag['sid_high'] = mt_sid_high
                method_res_cpdag.append(
                    {'dataset': dataset_name, 'model': names_dict[method], 'elapsed': elapsed, **mt_cpdag})
            method_sum = pd.DataFrame(method_res).groupby(['dataset', 'model'], as_index=False).agg(
                ['mean', 'std']).round(2).reset_index(drop=True)
            method_sum.columns = method_sum.columns.map('_'.join).str.strip('_')
            mt_res = pd.concat([mt_res, method_sum], sort=False)

            method_sum = pd.DataFrame(method_res_cpdag).groupby(
                ['dataset', 'model'], as_index=False).agg(['mean', 'std']).round(2).reset_index(drop=True)
            method_sum.columns = method_sum.columns.map('_'.join).str.strip('_')
            mt_res_cpdag = pd.concat([mt_res_cpdag, method_sum], sort=False)

            # cache intermediate results
            mt_res.to_csv(mt_path, index=False)
            mt_res_cpdag.to_csv(mt_cpdag_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(n_runs=args.n_runs,
         sample_size=args.sample_size,
         device=args.device)
