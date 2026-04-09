import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize


from src.utils.gen_random_nx import generate_random_bn_data
from ArgCausalDisco.utils.helpers import random_stability, logger_setup
from ArgCausalDisco.abapc import ABAPC
from src.abapc import get_arrow_sets, get_best_model
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


ALPHA = 0.01
INDEP_TEST = 'fisherz'
EDGE_NODE_RATIO = 1
N_NODES = [3, 4, 5, 6, 7, 8, 9, 10]


def parse_args():
    parser = ArgumentParser(description="Compare ABAPC implementations on random DAGs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for the experiment')
    parser.add_argument('--sample-size', type=int, default=SAMPLE_SIZE, help='Sample size for the random DAGs')
    parser.add_argument('--output-dir', type=str, default='results/extension_based_semantics', help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger_setup(str(out_path / 'log_compare_abapc.log'))

    random_stability(SEED)
    seeds_list = np.random.randint(0, 10000, (args.n_runs,)).tolist()
    df = pd.DataFrame(columns=['n_nodes', 'n_edges', 'seed', 'pure_abapc_elapsed', 'old_elapsed'])

    for n_nodes in tqdm(N_NODES, desc="tqdm Nodes"):
        n_edges = int(n_nodes * EDGE_NODE_RATIO)

        for seed in tqdm(seeds_list, desc="tqdm Seeds"):

            X_s, B_true = generate_random_bn_data(
                n_nodes=n_nodes,
                n_edges=n_edges,
                n_samples=args.sample_size,
                seed=seed,
                standardise=True
            )

            start = time.time()
            stable_arrow_sets, cg, _, _ = get_arrow_sets(X_s,
                                                          seed=seed,
                                                          alpha=ALPHA,
                                                          indep_test=INDEP_TEST)
            _, W_est_pure, _ = get_best_model(stable_arrow_sets,
                                              n_nodes=n_nodes,
                                              cg=cg,
                                              alpha=ALPHA)
            pure_abapc_elapsed = time.time() - start

            # old implementation
            start = time.time()
            W_est_old = ABAPC(data=X_s,
                              seed=seed,
                              alpha=ALPHA,
                              indep_test=INDEP_TEST,
                              scenario=f"abapc_nodes{n_nodes}_edges{n_edges}_random",
                              out_mode="opt")
            old_elapsed = time.time() - start

            df = pd.concat([df, pd.DataFrame([{
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'seed': seed,
                'pure_abapc_elapsed': pure_abapc_elapsed,
                'old_elapsed': old_elapsed
            }])], ignore_index=True)

            df.to_csv(out_path / f'compare_abapc_random.csv', index=False)


if __name__ == "__main__":
    main()
