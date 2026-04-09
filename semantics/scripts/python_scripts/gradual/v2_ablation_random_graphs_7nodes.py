'''
Ablation study for gradual ABA v2 on random graphs with 7 nodes.
This script runs the model with different parameters and records the results.
It iterates over neighbourhood_n_nodes, c_set_size, and use_collider_arguments and search depth.
It uses a fixed number of edges (7) and nodes (7) in the random graphs.
The results are saved to CSV files in the specified result directory.
The script is designed to be run with a specified number of runs (default 50).
It uses a timeout to prevent long-running processes and handles exceptions gracefully.
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
from src.utils.resource_utils import MemoryUsageExceededException, TimeoutException, timeout
from src.utils.utils import check_arrows_dag, get_matrix_from_arrow_set, parse_arrow
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


configure_r()

RESULT_DIR = Path("./results/gradual/v2_ablation_random_graphs_7nodes")

N_NODES = 7
N_EDGES = 7  # Number of edges in the random graph

NEIGHBOURHOOD_N_NODES = [3, 4, 5, 6, 7]
C_SET_SIZE = [0, 1, 2, 3, 4, 5]
SEARCH_DEPTH = [4, 5, 6, 7, 8, 9, 10]

DEFAULTS = {
    'neighbourhood_n_nodes': 7,
    'c_set_size': 5,
    'search_depth': 10,
    'use_collider_arguments': True,
}

TIMEOUT = 10 * 60  # 10 minutes


@timeout(TIMEOUT)  # Set a timeout of 10 minutes for the model run
def run_model_with_timeout(n_nodes, bsaf, model_name, set_aggregation, aggregation, influence, conservativeness, iterations):
    """
    Run the model with a timeout to prevent long-running processes.
    """
    return run_model(
        n_nodes=n_nodes,
        bsaf=bsaf,
        model_name=model_name,
        set_aggregation=set_aggregation,
        aggregation=aggregation,
        influence=influence,
        conservativeness=conservativeness,
        iterations=iterations
    )


def get_sorted_arrows(seed, 
                      X_s, 
                      assumptions_dict, 
                      bsaf,  
                      n_nodes):
    """
    Get sorted arrows based on their dialectical strengths.
    This function sources facts from the data, runs the model to get strengths,
    and returns the sorted arrows.
    If the model run times out, it returns None for all outputs.
    Args:
        seed (int): Random seed for reproducibility.
        X_s (np.ndarray): Sample data.
        assumptions_dict (dict): Dictionary of assumptions for the BSAF.
        bsaf (BSAF): The BSAF object containing the assumptions.
        n_nodes (int): Number of nodes in the Bayesian network.
    Returns:
        tuple: A tuple containing:
            - elapsed_fact_sourcing (float): Time taken to source facts.
            - elapsed_model_solution (float): Time taken to run the model.
            - is_converged (bool): Whether the model has converged.
            - score_original_prefilled (function): Function to score the original model.
            - sorted_arrows (list): List of sorted arrows based on their strengths.
        Or None for all outputs if the model run times out.
    """
    random_stability(seed)
    start_fact_sourcing = time.time()
    cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
    facts = sorted(facts, key=lambda x: x.score, reverse=True)  # sort by score
    elapsed_fact_sourcing = time.time() - start_fact_sourcing

    # run gradual aba to get the strengths
    start_model_solution = time.time()
    reset_weights(assumptions_dict)
    set_weights_according_to_facts(assumptions_dict, facts)
    try:
        output = run_model_with_timeout(
            n_nodes=n_nodes,
            bsaf=bsaf,
            model_name=None,
            set_aggregation=SetMinAggregation(),
            aggregation=TopDiffAggregation(),
            influence=LinearInfluence(conservativeness=1),
            conservativeness=1,
            iterations=25,
        )
    except TimeoutException as e:
        return None, None, None, None, None
    else:
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

        return (elapsed_fact_sourcing, 
                elapsed_model_solution, 
                is_converged, 
                score_original_prefilled, 
                sorted_arrows)


def record_results(
    seed,
    n_nodes,
    n_edges,
    bsaf_builder,
    elapsed_fact_sourcing,
    elapsed_bsaf_creation,
    elapsed_model_solution,
    elapsed_orig_ranking,
    search_depth,
    is_converged,
    best_model_original_ranking,
    original_ranking_I,
    B_true
):
    """Record the results of the model run and update the metrics dataframes."""
    add_info = {
        'dataset': f'random_graph_{n_nodes}_{n_edges}',
        'seed': seed,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'neighbourhood_n_nodes': DEFAULTS['neighbourhood_n_nodes'],
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
        'aba_elapsed': elapsed_fact_sourcing,  # No ABA elapsed time in this context, but takes time for mpc
        'ranking_elapsed': elapsed_orig_ranking,
        'best_I': original_ranking_I,
    }
    logger.info(f"Run finished: {add_info}")

    # get and store metrics
    best_B_est = get_matrix_from_arrow_set(best_model_original_ranking, n_nodes)
    mt_cpdag, mt_dag = get_metrics(best_B_est, B_true)

    # add base info
    mt_cpdag.update(add_info)
    mt_dag.update(add_info)

    logger.info(f"Metrics; CPDAG: {mt_cpdag}, DAG: {mt_dag}")

    # append to dataframes
    cpdag_metrics_df = pd.read_csv(RESULT_DIR / 'cpdag_metrics.csv') if (RESULT_DIR / 'cpdag_metrics.csv').exists() else pd.DataFrame()
    dag_metrics_df = pd.read_csv(RESULT_DIR / 'dag_metrics.csv') if (RESULT_DIR / 'dag_metrics.csv').exists() else pd.DataFrame()
    cpdag_metrics_df = pd.concat([cpdag_metrics_df,
                                pd.DataFrame([mt_cpdag])],
                                ignore_index=True)
    dag_metrics_df = pd.concat([dag_metrics_df,
                                pd.DataFrame([mt_dag])],
                            ignore_index=True)

    # save to csv
    cpdag_metrics_df.to_csv(RESULT_DIR / 'cpdag_metrics.csv', index=False)
    dag_metrics_df.to_csv(RESULT_DIR / 'dag_metrics.csv', index=False)


def run_and_record_custom_neighbouthood_c_set_coll_arg(
    neighbourhood_n_nodes,
    c_set_size,
    use_collider_arguments,
    seeds_list,
    search_algo,
):
    """Run the model with custom neighbourhood_n_nodes, c_set_size, and use_collider_arguments.
        the search depth is fixed to the default value.

        Will build the BSAF with the given parameters and run the model for each seed.
        Records the results in the provided dataframes.
    Args:
        neighbourhood_n_nodes (int): Number of nodes in the neighbourhood.
        c_set_size (int): Size of the conditioning set.
        use_collider_arguments (bool): Whether to include collider tree arguments.
        seeds_list (list): List of random seeds to use for the model runs.
        cpdag_metrics_df (pd.DataFrame): DataFrame to store CPDAG metrics.
        dag_metrics_df (pd.DataFrame): DataFrame to store DAG metrics.
    Returns:
        None: The function records the results in the provided dataframes.
    """

    start_bsaf_creation = time.time()
    bsaf_builder = BSAFBuilderV2(
        n_nodes=N_NODES,
        include_collider_tree_arguments=use_collider_arguments,
        neighbourhood_n_nodes=neighbourhood_n_nodes,
        max_conditioning_set_size=c_set_size)  # everything is maximal for given n_nodes, full scale
    bsaf = bsaf_builder.create_bsaf()
    assumptions_dict = bsaf_builder.name_to_assumption
    elapsed_bsaf_creation = time.time() - start_bsaf_creation
    for seed in tqdm(seeds_list):
        X_s, B_true = generate_random_bn_data(
            n_nodes=N_NODES,
            n_edges=N_EDGES,
            n_samples=SAMPLE_SIZE,
            seed=seed,
            standardise=True
        )
        (elapsed_fact_sourcing, 
        elapsed_model_solution, 
        is_converged, 
        score_original_prefilled, 
        sorted_arrows) = get_sorted_arrows(
            seed=seed,
            X_s=X_s,
            assumptions_dict=assumptions_dict,
            bsaf=bsaf,
            n_nodes=N_NODES
        )
        if elapsed_fact_sourcing is None:
            logger.error(f"Model run for seed {seed} timed out. Skipping this seed.")
            continue
        start_orig_ranking = time.time()
        original_ranking_I, best_model_original_ranking = search_algo(
            steps_ahead=DEFAULTS['search_depth'],
            sorted_arrows=sorted_arrows,
            is_dag=check_arrows_dag,
            get_score=score_original_prefilled
        )
        elapsed_orig_ranking = time.time() - start_orig_ranking
        record_results(
            seed=seed,
            n_nodes=N_NODES,
            n_edges=N_EDGES,
            bsaf_builder=bsaf_builder,
            elapsed_fact_sourcing=elapsed_fact_sourcing,
            elapsed_bsaf_creation=elapsed_bsaf_creation,
            elapsed_model_solution=elapsed_model_solution,
            elapsed_orig_ranking=elapsed_orig_ranking,
            search_depth=DEFAULTS['search_depth'],
            is_converged=is_converged,
            best_model_original_ranking=best_model_original_ranking,
            original_ranking_I=original_ranking_I,
            B_true=B_true)


def parse_args():
    parser = argparse.ArgumentParser(description="Run gradual ABA experiment v2 on random graphs")
    parser.add_argument('--n-runs', type=int, default=50, help='Number of runs for each dataset')
    parser.add_argument('--use-every-step-search', type=str, choices=['true', 'false'], default='false',
                        help='Whether to use every-step search or single search at the top arrow.')
    return parser.parse_args()


def main(n_runs, use_every_step_search=False):
    logger.info("Starting new experiment, metrics will be saved to file.")
    # Delete existing results files if they exist
    (RESULT_DIR / 'cpdag_metrics.csv').unlink(missing_ok=True)
    (RESULT_DIR / 'dag_metrics.csv').unlink(missing_ok=True)

    if use_every_step_search:
        search_algo = search_best_model_every_step
        logger.info("Using every-step search algorithm.")
    else:
        search_algo = limited_depth_search_best_model
        logger.info("Using single search at the top arrow algorithm.")

    random_stability(SEED)
    seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()
    
    # Iterate all search depth parameters:
    logger.info(f"Running experiment with {n_runs} runs, seeds: {seeds_list}, iterating over search depths: {SEARCH_DEPTH}, ")
    start_bsaf_creation = time.time()
    bsaf_builder = BSAFBuilderV2(
        n_nodes=N_NODES,
        include_collider_tree_arguments=DEFAULTS['use_collider_arguments'],
        neighbourhood_n_nodes=DEFAULTS['neighbourhood_n_nodes'],
        max_conditioning_set_size=DEFAULTS['c_set_size'])  # everything is maximal for given n_nodes, full scale
    bsaf = bsaf_builder.create_bsaf()
    assumptions_dict = bsaf_builder.name_to_assumption
    elapsed_bsaf_creation = time.time() - start_bsaf_creation
    for seed in tqdm(seeds_list):
        X_s, B_true = generate_random_bn_data(
            n_nodes=N_NODES,
            n_edges=N_EDGES,
            n_samples=SAMPLE_SIZE,
            seed=seed,
            standardise=True
        )
        (elapsed_fact_sourcing, 
         elapsed_model_solution, 
         is_converged, 
         score_original_prefilled, 
         sorted_arrows) = get_sorted_arrows(
            seed=seed,
            X_s=X_s,
            assumptions_dict=assumptions_dict,
            bsaf=bsaf,
            n_nodes=N_NODES
        )
        if elapsed_fact_sourcing is None:
            logger.error(f"Model run for seed {seed} timed out. Skipping this seed.")
            continue

        for search_depth in SEARCH_DEPTH:
            start_orig_ranking = time.time()
            original_ranking_I, best_model_original_ranking = search_algo(
                steps_ahead=search_depth,
                sorted_arrows=sorted_arrows,
                is_dag=check_arrows_dag,
                get_score=score_original_prefilled
            )
            elapsed_orig_ranking = time.time() - start_orig_ranking
            record_results(
                seed=seed,
                n_nodes=N_NODES,
                n_edges=N_EDGES,
                bsaf_builder=bsaf_builder,
                elapsed_fact_sourcing=elapsed_fact_sourcing,
                elapsed_bsaf_creation=elapsed_bsaf_creation,
                elapsed_model_solution=elapsed_model_solution,
                elapsed_orig_ranking=elapsed_orig_ranking,
                search_depth=search_depth,
                is_converged=is_converged,
                best_model_original_ranking=best_model_original_ranking,
                original_ranking_I=original_ranking_I,
                B_true=B_true)

    # Iterate through neighbourhood_n_nodes and c_set_size
    logger.info(f"Running experiment with {n_runs} runs, seeds: {seeds_list}, iterating over neighbourhood_n_nodes: {NEIGHBOURHOOD_N_NODES}")
    for neighbourhood_n_nodes in NEIGHBOURHOOD_N_NODES:
        if neighbourhood_n_nodes != DEFAULTS['neighbourhood_n_nodes']:
            run_and_record_custom_neighbouthood_c_set_coll_arg(
                neighbourhood_n_nodes=neighbourhood_n_nodes,
                c_set_size=DEFAULTS['c_set_size'],
                use_collider_arguments=DEFAULTS['use_collider_arguments'],
                seeds_list=seeds_list,
                search_algo=search_algo,
            )
    
    # Iterate through c_set_size
    logger.info(f"Running experiment with {n_runs} runs, seeds: {seeds_list}, iterating over c_set_size: {C_SET_SIZE}")
    for c_set_size in C_SET_SIZE:
        if c_set_size != DEFAULTS['c_set_size']:
            run_and_record_custom_neighbouthood_c_set_coll_arg(
                neighbourhood_n_nodes=DEFAULTS['neighbourhood_n_nodes'],
                c_set_size=c_set_size,
                use_collider_arguments=DEFAULTS['use_collider_arguments'],
                seeds_list=seeds_list,
                search_algo=search_algo,
            )

    # Iterate through use_collider_arguments
    logger.info(f"Running experiment with {n_runs} runs, seeds: {seeds_list}, iterating over use_collider_arguments: {DEFAULTS['use_collider_arguments']}") 
    run_and_record_custom_neighbouthood_c_set_coll_arg(
        neighbourhood_n_nodes=DEFAULTS['neighbourhood_n_nodes'],
        c_set_size=DEFAULTS['c_set_size'],
        use_collider_arguments=False,  # Opposite of default
        seeds_list=seeds_list,
        search_algo=search_algo
    )


if __name__ == "__main__":
    args = parse_args()

    # create results directory
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # run the experiment
    main(n_runs=args.n_runs,
         use_every_step_search=(args.use_every_step_search.lower() == 'true'))

    logger.info("Experiment completed successfully.")
