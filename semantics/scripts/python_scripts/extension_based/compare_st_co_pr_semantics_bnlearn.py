import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

import time
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from src.utils.configure_r import configure_r
from src.utils.enums import SemanticEnum
from src.abapc import get_arrow_sets, get_best_model
from src.utils.bn_utils import get_dataset
from ArgCausalDisco.utils.helpers import random_stability, logger_setup
from ArgCausalDisco.utils.graph_utils import DAGMetrics, dag2cpdag
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


EDGE_NODE_RATIO = 1
DATASETS = [
    'cancer',
    'earthquake',
    'survey',
    # 'asia'
]


def parse_args():
    parser = ArgumentParser(description="Compare ABAPC implementations on random DAGs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for the experiment')
    parser.add_argument('--sample-size', type=int, default=SAMPLE_SIZE, help='Sample size for the random DAGs')
    parser.add_argument('--output-dir', type=str, default='results/extension_based_semantics', help='Directory to save results')
    return parser.parse_args()


def main():
    args = parse_args()
    configure_r()
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger_setup(str(out_path / 'log_compare_sem_bnlearn.log'))

    random_stability(SEED)
    seeds_list = np.random.randint(0, 10000, (args.n_runs,)).tolist()
    df = pd.DataFrame()

    for dataset_name in tqdm(DATASETS, desc="tqdm Datasets"):

        for seed in tqdm(seeds_list, desc="tqdm Seeds"):
            X_s, B_true = get_dataset(dataset_name,
                                      seed=seed,
                                      sample_size=args.sample_size)
            n_nodes = X_s.shape[1]

            arrow_sets_dict = dict()
            combined_res_dict = dict()
            for sem in [SemanticEnum.ST, SemanticEnum.CO, SemanticEnum.PR]:

                start = time.time()
                arrow_sets, cg, num_facts, facts = get_arrow_sets(X_s,
                                                                  seed=seed,
                                                                  alpha=ALPHA,
                                                                  indep_test=INDEP_TEST,
                                                                  semantics=sem)
                best_model, best_W_est, best_I = get_best_model(arrow_sets,
                                                                n_nodes=n_nodes,
                                                                cg=cg,
                                                                alpha=ALPHA)
                elapsed = time.time() - start

                arrow_sets_dict[sem.name] = set(arrow_sets)

                B_est = (best_W_est != 0).astype(int)
                mt_cpdag = DAGMetrics(dag2cpdag(B_est), B_true).metrics
                B_est = (best_W_est > 0).astype(int)
                mt_dag = DAGMetrics(B_est, B_true).metrics

                combined_res_dict[f'{sem.name}_elapsed'] = elapsed
                combined_res_dict[f'{sem.name}_best_model'] = best_model
                combined_res_dict[f'{sem.name}_best_I'] = best_I
                combined_res_dict[f'{sem.name}_total_num_facts'] = len(facts)
                combined_res_dict[f'{sem.name}_used_num_facts'] = num_facts
                combined_res_dict[f'{sem.name}_num_models'] = len(arrow_sets_dict[sem.name])
                combined_res_dict[f'{sem.name}_mt_dag'] = json.dumps(mt_dag, default=str)
                combined_res_dict[f'{sem.name}_mt_cpdag'] = json.dumps(mt_cpdag, default=str)

            combined_res_dict.update({
                'n_nodes': n_nodes,
                'dataset_name': dataset_name,
                'seed': seed,
                'is_best_st_in_all_co': combined_res_dict[f'{SemanticEnum.ST.name}_best_model'] in arrow_sets_dict[SemanticEnum.CO.name],
                'is_best_co_in_all_st': combined_res_dict[f'{SemanticEnum.CO.name}_best_model'] in arrow_sets_dict[SemanticEnum.ST.name],
                'is_best_st_in_all_pr': combined_res_dict[f'{SemanticEnum.ST.name}_best_model'] in arrow_sets_dict[SemanticEnum.PR.name],
                'is_best_pr_in_all_co': combined_res_dict[f'{SemanticEnum.PR.name}_best_model'] in arrow_sets_dict[SemanticEnum.CO.name],
                'is_all_st_subset_of_all_co': arrow_sets_dict[SemanticEnum.ST.name].issubset(arrow_sets_dict[SemanticEnum.CO.name]),
                'is_all_st_subset_of_all_pr': arrow_sets_dict[SemanticEnum.ST.name].issubset(arrow_sets_dict[SemanticEnum.PR.name]),
                'is_all_pr_subset_of_all_st': arrow_sets_dict[SemanticEnum.PR.name].issubset(arrow_sets_dict[SemanticEnum.ST.name]),
                'is_all_pr_subset_of_all_co': arrow_sets_dict[SemanticEnum.PR.name].issubset(arrow_sets_dict[SemanticEnum.CO.name]),
            })

            df = pd.concat([df, pd.DataFrame([combined_res_dict])], ignore_index=True)

            df.to_csv(out_path / f'compare_semantics_bnlearn.csv', index=False)


if __name__ == "__main__":
    main()
