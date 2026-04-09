import sys
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, 'aspforaba/')
sys.path.insert(0, '.')
import sitecustomize

import time
import argparse
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

from logger import logger

from src.abapc import get_cg_and_facts
from ArgCausalDisco.utils.helpers import random_stability
from src.utils.bn_utils import get_dataset
from src.utils.fact_utils import check_if_fact_is_true
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


RESULT_DIR = Path("./results/existing/runtime_sourcing_facts_bnlearn_graphs")
DATASETS = ['cancer', 'earthquake', 'survey', 'asia', 'sachs', 'child', 'insurance']

DAG_NODES_MAP = {'cancer': 5,
                 'earthquake': 5,
                 'survey': 6,
                 'asia': 8,
                 'sachs': 11,
                 'child': 20,
                 'insurance': 27}

def parse_args():
    parser = argparse.ArgumentParser(description="runtime_sourcing_facts_bnlearn_graphs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    return parser.parse_args()

def main(n_runs):
    results = pd.DataFrame()

    for dataset in DATASETS:
        n_nodes = DAG_NODES_MAP[dataset]
        logger.info(f"Running experiment for dataset: {dataset}")

        random_stability(SEED)
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()

        for seed in tqdm(seeds_list, desc=f"Running dataset {dataset} with {n_runs} seeds"):
            X_s, B_true = get_dataset(dataset,
                                 seed=seed,
                                 sample_size=SAMPLE_SIZE)
            random_stability(seed)
            start = time.time()
            cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
            elapsed = time.time() - start

            fact_metadata = []
            for fact in facts:
                is_true = check_if_fact_is_true(fact, B_true)
                fact_metadata.append({'is_true': is_true,
                                      'node1': int(fact.node1),
                                      'node2': int(fact.node2),
                                      'node_set': sorted([int(i) for i in fact.node_set]),
                                      'relation': fact.relation.value,
                                      'score': fact.score})
            fact_metadata_json = json.dumps(fact_metadata)
            # Record runtime and the maximum size of conditioning set of the facts
            results = pd.concat([results, pd.DataFrame({
                'n_nodes': [n_nodes],
                'dataset': [dataset],
                'seed': [seed],
                'elapsed_time': [elapsed],
                'fact_metadata': [fact_metadata_json],
                'true_dag': [json.dumps(B_true.tolist())],
            })], ignore_index=True)
            results.to_csv(RESULT_DIR / 'runtime_results.csv', index=False)


if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    main(args.n_runs)
    logger.info("Runtime sourcing facts experiment completed successfully!")
    logger.info(f"Results saved to {RESULT_DIR / 'runtime_results.csv'}")
