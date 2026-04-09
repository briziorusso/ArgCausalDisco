"""
In experiments I noticed that when calling old ABAPC implementation with a different interface
than the one provided in the original repo it runs faster. 
It might be caused by the model strength sorting being done faster in this new interface.
There is a test checking that this implementation gives the same results as the original one.
This test is in tests/end_to_end/test_our_wrappers_for_old_abapc.py.

So here ABAPC is rerun, to benchmark new runtime and use that in runtime reports.
"""

import sys
sys.path.insert(0, 'aspforaba/')
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

import time
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from logger import logger

from src.abapc import  get_cg_and_facts, get_best_model
from src.utils.gen_random_nx import generate_random_bn_data
from src.utils.configure_r import configure_r
from ArgCausalDisco.utils.helpers import random_stability

from src.abapc import get_models_from_facts
from src.utils.metrics import get_metrics
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


configure_r()

RESULT_DIR = Path("./results/existing/abapc_old_but_faster_on_random/")
N_NODES = [3, 4, 5, 6, 7]


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradual ABA experiment v1")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    return parser.parse_args()


def main(n_runs):
    facts_path = Path('./facts')
    facts_path.mkdir(parents=True, exist_ok=True)
    cpdag_metrics_df = pd.DataFrame()
    dag_metrics_df = pd.DataFrame()

    for n_nodes in N_NODES:
        n_edges = n_nodes
        logger.info(f"Running experiment for n_nodes: {n_nodes}, n_edges: {n_edges}")

        random_stability(SEED)
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()

        for seed in tqdm(seeds_list, desc=f"Running |V|={n_nodes} |E|={n_edges} with {n_runs} seeds"):
            X_s, B_true = generate_random_bn_data(
                    n_nodes=n_nodes,
                    n_edges=n_edges,
                    n_samples=SAMPLE_SIZE,
                    seed=seed,
                    standardise=True
            )
            random_stability(seed)
            start = time.time()
            cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
            facts = sorted(facts, key=lambda x: x.score, reverse=True)  # sort by score

            # original
            base_location = str(facts_path / f'nodes{n_nodes}_edges{n_edges}_{seed}_facts.lp')
            model_sets = get_models_from_facts(facts, seed, n_nodes, base_location=base_location)
            best_model, best_B_est, best_I = get_best_model(model_sets, n_nodes, cg, alpha=ALPHA)
            elapsed = time.time() - start

            base_info = {
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'model': 'ABAPC (Original)',
                'seed': seed,
                'elapsed': elapsed,
            }

            mt_cpdag, mt_dag = get_metrics(best_B_est, B_true)
            # add base info
            mt_cpdag.update(base_info)
            mt_dag.update(base_info)

            # append to dataframes
            cpdag_metrics_df = pd.concat([cpdag_metrics_df, 
                                            pd.DataFrame([mt_cpdag])], 
                                            ignore_index=True)
            dag_metrics_df = pd.concat([dag_metrics_df, 
                                        pd.DataFrame([mt_dag])], 
                                        ignore_index=True)
            # save to csv
            cpdag_metrics_df.to_csv(RESULT_DIR / f'cpdag_metrics.csv', index=False)
            dag_metrics_df.to_csv(RESULT_DIR / f'dag_metrics.csv', index=False)


if __name__ == "__main__":
    args = parse_args()
    n_runs = args.n_runs

    # create results directory
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # run the experiment
    main(n_runs)

    logger.info("Experiment completed successfully.")
