"""
This script will run Gradual Causal ABA V2 for a given dataset chosen from bnlearn
benchmark datasets. It will also record various metrics such as time taken and the SID score.
"""
import sys
sys.path.insert(0, 'aspforaba/')
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

from src.utils.utils import check_arrows_dag, get_matrix_from_arrow_set, parse_arrow
from src.utils.metrics import get_metrics
from src.utils.bn_utils import get_dataset
from src.utils.configure_r import configure_r
from src.constants import DAG_NODES_MAP, DAG_EDGES_MAP
from src.gradual.semantic_modules.TopDiffAggregation import TopDiffAggregation
from src.gradual.search_best_model import limited_depth_search_best_model
from src.gradual.search_best_model_every_step import search_best_model_every_step
from src.gradual.scalable.bsaf_builder_v2 import BSAFBuilderV2
from src.gradual.run import reset_weights, run_model, set_weights_according_to_facts
from src.abapc import get_cg_and_facts, score_model_original
from logger import logger
from GradualABA.semantics.modular.SetMinAggregation import SetMinAggregation
from GradualABA.semantics.modular.LinearInfluence import LinearInfluence
from ArgCausalDisco.utils.helpers import random_stability
from tqdm import tqdm
from pathlib import Path
from functools import partial
import time
import pandas as pd
import numpy as np
import argparse
from src.constants import ALPHA, INDEP_TEST, SAMPLE_SIZE, SEED


configure_r()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate params for Gradual Causal ABA on random graphs.")
    parser.add_argument('-p', '--path', type=str, help='Path to save results CSV file.')
    parser.add_argument('-d', '--search-depth', type=int, default=12,
                        help='Search depth for finding the best arrow configuration.')
    parser.add_argument('-n', '--name', type=str, 
                        choices=list(DAG_NODES_MAP.keys()),
                        help='Name of the bnlearn dataset to use.')
    parser.add_argument('-l', '--neighbourhood-n-nodes', type=int, help='Neighbourhood size to test.')
    parser.add_argument('-c', '--c-set-size', type=int, help='Conditioning set size to test.')
    parser.add_argument('--n-runs', type=int,
                        default=50, help='Number of runs to perform.')
    parser.add_argument('-b', '--batch-number', type=int, default=1,
                        help='Batch number for parallel runs.')
    parser.add_argument('-t', '--total-batches', type=int, default=1,
                        help='Total number of batches for parallel runs.')
    parser.add_argument('--use-every-step-search', type=str, choices=['true', 'false'], default='false',
                        help='Whether to use every-step search or single search at the top arrow.')
    return parser.parse_args()


def record_results(
    result_dir,
    seed,
    dataset_name,
    neighbourhood_n_nodes,
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
    n_nodes = DAG_NODES_MAP[dataset_name]
    n_edges = DAG_EDGES_MAP[dataset_name]
    add_info = {
        'dataset': dataset_name,
        'seed': seed,
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'neighbourhood_n_nodes': neighbourhood_n_nodes,
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
    cpdag_metrics_df = pd.read_csv(result_dir / 'cpdag_metrics.csv') if (result_dir / 'cpdag_metrics.csv').exists() else pd.DataFrame()
    dag_metrics_df = pd.read_csv(result_dir / 'dag_metrics.csv') if (result_dir / 'dag_metrics.csv').exists() else pd.DataFrame()
    cpdag_metrics_df = pd.concat([cpdag_metrics_df,
                                pd.DataFrame([mt_cpdag])],
                                ignore_index=True)
    dag_metrics_df = pd.concat([dag_metrics_df,
                                pd.DataFrame([mt_dag])],
                            ignore_index=True)

    # save to csv
    cpdag_metrics_df.to_csv(result_dir / 'cpdag_metrics.csv', index=False)
    dag_metrics_df.to_csv(result_dir / 'dag_metrics.csv', index=False)


def get_facts_and_graphs_for_seed(seed, name):
    X_s, B_true = get_dataset(
        dataset_name=name,
        seed=seed,
        sample_size=SAMPLE_SIZE
    )
    random_stability(seed)
    start_fact_sourcing = time.time()
    cg, facts = get_cg_and_facts(X_s, alpha=ALPHA, indep_test=INDEP_TEST)
    facts = sorted(facts, key=lambda x: x.score, reverse=True)  # sort by score
    elapsed_fact_sourcing = time.time() - start_fact_sourcing
    return X_s, B_true, cg, facts, elapsed_fact_sourcing


def get_sorted_arrows(seed, 
                      facts,
                      elapsed_fact_sourcing,
                      cg,
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
        iterations=50,
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

    return (elapsed_fact_sourcing, 
            elapsed_model_solution, 
            is_converged, 
            score_original_prefilled, 
            sorted_arrows)


def main(path,
         search_depth,
         name,
         neighbourhood_n_nodes,
         c_set_size,
         n_runs,
         batch_number,
         total_batches,
         use_every_step_search):
    n_nodes = DAG_NODES_MAP[name]

    random_stability(SEED)
    seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()

    # Separate seets for the current batch
    batch_size = n_runs // total_batches
    start_index = (batch_number - 1) * batch_size
    end_index = min(start_index + batch_size, n_runs)
    seeds_list = seeds_list[start_index:end_index]

    # get all facts and causal graphs for the seeds
    facts_and_cg = dict()
    for seed in tqdm(seeds_list):
        X_s, B_true, cg, facts, elapsed_fact_sourcing = get_facts_and_graphs_for_seed(seed, name)
        facts_and_cg[seed] = (X_s, B_true, cg, facts, elapsed_fact_sourcing)

    start_bsaf_creation = time.time()
    bsaf_builder = BSAFBuilderV2(
        n_nodes=n_nodes,
        include_collider_tree_arguments=True,
        neighbourhood_n_nodes=neighbourhood_n_nodes,
        max_conditioning_set_size=c_set_size)  # everything is maximal for given n_nodes, full scale
    bsaf = bsaf_builder.create_bsaf()
    assumptions_dict = bsaf_builder.name_to_assumption
    elapsed_bsaf_creation = time.time() - start_bsaf_creation


    for seed in tqdm(seeds_list):
        X_s, B_true, cg, facts, elapsed_fact_sourcing = facts_and_cg[seed]
        (elapsed_fact_sourcing, 
        elapsed_model_solution, 
        is_converged, 
        score_original_prefilled, 
        sorted_arrows) = get_sorted_arrows(
            seed=seed,
            facts=facts,
            elapsed_fact_sourcing=elapsed_fact_sourcing,
            cg=cg,
            assumptions_dict=assumptions_dict,
            bsaf=bsaf,
            n_nodes=n_nodes
        )

        start_orig_ranking = time.time()
        if use_every_step_search:
            searcher = search_best_model_every_step
        else:
            searcher = limited_depth_search_best_model

        original_ranking_I, best_model_original_ranking = searcher(
            steps_ahead=search_depth,
            sorted_arrows=sorted_arrows,
            is_dag=check_arrows_dag,
            get_score=score_original_prefilled
        )

        elapsed_orig_ranking = time.time() - start_orig_ranking
        record_results(
            result_dir=path,
            seed=seed,
            dataset_name=name,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
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


if __name__ == "__main__":
    args = parse_args()
    RESULT_DIR = Path(args.path)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # Run the main function with parsed arguments
    main(
        path=RESULT_DIR,
        search_depth=args.search_depth,
        name=args.name,
        neighbourhood_n_nodes=args.neighbourhood_n_nodes,
        c_set_size=args.c_set_size,
        n_runs=args.n_runs,
        batch_number=args.batch_number,
        total_batches=args.total_batches,
        use_every_step_search=(args.use_every_step_search.lower() == 'true'),
    )

    logger.info("All runs completed successfully.")
