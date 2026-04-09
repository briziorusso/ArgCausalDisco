import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

from ArgCausalDisco.abapc import ABAPC
from src.utils.gen_random_nx import generate_random_bn_data
from ArgCausalDisco.utils.helpers import random_stability, logger_setup
from src.abapc import get_arrow_sets, get_best_model
from src.utils.enums import SemanticEnum
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


EDGE_NODE_RATIO = 1
N_NODES = [3, 4, 5, 6, 7,
        #    8, 9, 10
    ]


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
    df = pd.DataFrame()

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

            arrow_sets_dict = dict()
            combined_res_dict = dict()
            for sem in [SemanticEnum.ST, SemanticEnum.CO, SemanticEnum.PR]:

                start = time.time()
                arrow_sets, cg, num_facts, facts = get_arrow_sets(X_s,
                                                                  seed=seed,
                                                                  alpha=ALPHA,
                                                                  indep_test=INDEP_TEST,
                                                                  semantics=sem)
                best_model, best_B_est, best_I = get_best_model(arrow_sets,
                                                                n_nodes=n_nodes,
                                                                cg=cg,
                                                                alpha=ALPHA)
                elapsed = time.time() - start

                arrow_sets_dict[sem.name]= set(arrow_sets)

                combined_res_dict[f'{sem.name}_elapsed'] = elapsed
                combined_res_dict[f'{sem.name}_best_model'] = best_model
                combined_res_dict[f'{sem.name}_best_I'] = best_I
                combined_res_dict[f'{sem.name}_total_num_facts'] = len(facts)
                combined_res_dict[f'{sem.name}_used_num_facts'] = num_facts
                combined_res_dict[f'{sem.name}_num_models'] = len(arrow_sets_dict[sem.name])

            start = time.time()
            _ = ABAPC(data=X_s,
                      seed=seed,
                      alpha=ALPHA,
                      indep_test=INDEP_TEST,
                      scenario=f"abapc_nodes{n_nodes}_edges{n_edges}_random",
                      out_mode="opt")
            old_elapsed = time.time() - start
            combined_res_dict['abapc_existing_elapsed'] = old_elapsed

            combined_res_dict.update({
                'n_nodes': n_nodes,
                'n_edges': n_edges,
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

            df.to_csv(out_path / f'compare_semantics_random.csv', index=False)


if __name__ == "__main__":
    main()
