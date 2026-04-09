'''
Experiment with V2 scalable implementation.
On random generated bayesian graphs.
'''

import sys
sys.path.insert(0, 'aspforaba/')
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

import argparse
import numpy as np
import pandas as pd
import time
from functools import partial
from pathlib import Path
from tqdm import tqdm

from ArgCausalDisco.utils.helpers import random_stability
from GradualABA.semantics.modular.LinearInfluence import LinearInfluence
from GradualABA.semantics.modular.SetMinAggregation import SetMinAggregation
from logger import logger
from src.abapc import get_cg_and_facts, score_model_original
from src.gradual.run import reset_weights, run_model, set_weights_according_to_facts
from src.gradual.scalable.bsaf_builder_v2 import BSAFBuilderV2
from src.gradual.search_best_model import limited_depth_search_best_model
from src.gradual.search_best_model_every_step import search_best_model_every_step
from src.gradual.semantic_modules.TopDiffAggregation import TopDiffAggregation
from src.utils.configure_r import configure_r
from src.utils.gen_random_nx import generate_random_bn_data
from src.utils.metrics import get_metrics
from src.utils.resource_utils import MemoryUsageExceededException
from src.utils.utils import check_arrows_dag, get_matrix_from_arrow_set, parse_arrow
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


configure_r()


RESULT_DIR = Path("./results/gradual/v2_random_graphs_3_to_6_nodes")

N_NODES = [3, 4, 5, 6, 7]
TIMEOUT = 30 * 60  # 5 minutes


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradual ABA experiment v2 on random graphs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    parser.add_argument('--search-depth', type=int, default=10, help='Search depth for limited depth search')
    parser.add_argument('--use-every-step-search', type=str, choices=['true', 'false'], default='false',
                        help='Whether to use every-step search or single search at the top arrow.')
    return parser.parse_args()


def main(n_runs,
         search_depth,
         use_every_step_search=False):
    cpdag_metrics_df = pd.DataFrame()
    dag_metrics_df = pd.DataFrame()

    if use_every_step_search:
        search_algo = search_best_model_every_step
    else:
        search_algo = limited_depth_search_best_model

    for n_nodes in N_NODES:
        n_edges = n_nodes  # for now, only one edge per node
        
        dataset = f'random_graphs_n{n_nodes}_e{n_edges}'
        logger.info(f"Running experiment for dataset: {dataset}")

        random_stability(SEED)
        seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()

        # create bsaf
        start_bsaf_creation = time.time()
        try:
            bsaf_builder = BSAFBuilderV2(
                n_nodes=n_nodes,
                include_collider_tree_arguments=True,
                neighbourhood_n_nodes=n_nodes,
                max_conditioning_set_size=n_nodes-2)  # everything is maximal for given n_nodes, full scale
            bsaf = bsaf_builder.create_bsaf()
            assumptions_dict = bsaf_builder.name_to_assumption
        except MemoryUsageExceededException as e:
            logger.error(
                f"Memory usage exceeded while creating BSAF for {dataset}")
            continue
        elapsed_bsaf_creation = time.time() - start_bsaf_creation

        for seed in tqdm(seeds_list, desc=f"Running {dataset} with {n_runs} seeds"):

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
            elapsed_fact_sourcing = time.time() - start

            # run gradual aba to get the strengths
            start_model_solution = time.time()
            reset_weights(assumptions_dict)
            set_weights_according_to_facts(assumptions_dict, facts)
            output = run_model(
                n_nodes=n_nodes,
                bsaf=bsaf,
                model_name=None,
                set_aggregation=SetMinAggregation(),
                aggregation=TopDiffAggregation(),
                influence=LinearInfluence(conservativeness=1),
                conservativeness=1,
                iterations=25,
            )
            elapsed_model_solution = time.time() - start_model_solution
            is_converged = all(output.has_converged_map.values())

            score_original_prefilled = partial(score_model_original,
                                            n_nodes=n_nodes,
                                            cg=cg,
                                            alpha=ALPHA,
                                            return_only_I=True)

            sorted_arrows = sorted([(strength, arrow) for arrow, strength in output.arrow_strengths.items()],
                                reverse=True)
            sorted_arrows = [parse_arrow(arrow) for _, arrow in sorted_arrows]

            start_orig_ranking = time.time()
            original_ranking_I, best_model_original_ranking = search_algo(
                steps_ahead=search_depth,
                sorted_arrows=sorted_arrows,
                is_dag=check_arrows_dag,
                get_score=score_original_prefilled
            )
            elapsed_orig_ranking = time.time() - start_orig_ranking

            add_info = {
                'dataset': dataset,
                'seed': seed,
                'n_nodes': n_nodes,
                'n_edges': n_edges,
                'neighbourhood_n_nodes': n_nodes,
                'max_cycle_length': bsaf_builder.max_cycle_size,
                'max_ct_depth': bsaf_builder.max_collider_tree_depth if bsaf_builder.include_collider_tree_arguments else -1,
                'max_path_length': bsaf_builder.max_path_length,
                'max_c_set_size': bsaf_builder.max_conditioning_set_size,
                'search_depth': search_depth,
                'elapsed_bsaf_creation': elapsed_bsaf_creation,
                'elapsed_model_solution': elapsed_model_solution,
                'is_converged': is_converged,
                'fact_ranking_method': 'v2',
                'model_ranking_method': 'original_ranking',
                'num_edges_est': len(best_model_original_ranking),
                'best_model': best_model_original_ranking,
                'aba_elapsed': elapsed_fact_sourcing,  # No ABA elapsed time in this context, just the time taken to source facts
                'ranking_elapsed': elapsed_orig_ranking,
                'best_I': original_ranking_I,
            }
            logger.info(f"Run finnished: {add_info}")

            # get and store metrics
            best_B_est = get_matrix_from_arrow_set(best_model_original_ranking, n_nodes)
            mt_cpdag, mt_dag = get_metrics(best_B_est, B_true)

            # add base info
            mt_cpdag.update(add_info)
            mt_dag.update(add_info)

            logger.info(f"Metrics; CPDAG: {mt_cpdag}, DAG: {mt_dag}")

            # append to dataframes
            cpdag_metrics_df = pd.concat([cpdag_metrics_df,
                                        pd.DataFrame([mt_cpdag])],
                                        ignore_index=True)
            dag_metrics_df = pd.concat([dag_metrics_df,
                                        pd.DataFrame([mt_dag])],
                                    ignore_index=True)

            # save to csv
            cpdag_metrics_df.to_csv(RESULT_DIR / 'cpdag_metrics.csv', index=False)
            dag_metrics_df.to_csv(RESULT_DIR / 'dag_metrics.csv', index=False)



if __name__ == "__main__":
    args = parse_args()

    # create results directory
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # run the experiment
    main(n_runs=args.n_runs,
         search_depth=args.search_depth,
         use_every_step_search=(args.use_every_step_search.lower() == 'true'))

    logger.info("Experiment completed successfully.")
