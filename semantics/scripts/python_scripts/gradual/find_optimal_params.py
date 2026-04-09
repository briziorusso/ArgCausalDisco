"""
This script is used to find the optimal parameters for the GradualABA algorithm.
It runs for a specified number of nodes, edges, neighbourhood size, c-set-size, use-collider-arguments flag and number of runs.
It then stores the metrics for each run in a CSV file.
"""
import sys
sys.path.insert(0, 'aspforaba/')
sys.path.insert(0, 'GradualABA/')
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')
sys.path.insert(0, '.')
import sitecustomize

from src.utils.utils import check_arrows_dag, get_matrix_from_arrow_set, parse_arrow
from src.utils.resource_utils import TimeoutException, timeout
from src.utils.metrics import get_metrics
from src.utils.gen_random_nx import generate_random_bn_data
from src.utils.configure_r import configure_r
from src.gradual.semantic_modules.TopDiffAggregation import TopDiffAggregation
from src.gradual.search_best_model import limited_depth_search_best_model
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
TIMEOUT = 60 * 60  # 1 hour in seconds


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate params for Gradual Causal ABA on random graphs.")
    parser.add_argument('-p', '--path', type=str, help='Path to save results CSV file.')
    parser.add_argument('-d', '--search-depth', type=int, default=10,
                        help='Search depth for finding the best arrow configuration.')
    parser.add_argument('-n', '--n-nodes', type=int, help='Number of nodes in the causal graph.')
    parser.add_argument('-k', '--n-edges', type=int, help='Number of edges in the causal graph.')
    parser.add_argument('-l', '--neighbourhood-n-nodes', type=int, help='Neighbourhood size to test.')
    parser.add_argument('-c', '--c-set-size', type=int, help='Conditioning set size to test.')
    parser.add_argument('-u', '--use-collider-arguments', type=str, choices=['true', 'false'],
                        default='true', help='Whether to use collider arguments (true/false).')
    parser.add_argument('--n-runs', type=int,
                        default=20, help='Number of runs to perform.')
    return parser.parse_args()


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


def record_results(
    result_dir,
    seed,
    n_nodes,
    n_edges,
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
    add_info = {
        'dataset': f'random_graph_{n_nodes}_{n_edges}',
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


def get_facts_and_graphs_for_seed(seed, n_nodes, n_edges):
    X_s, B_true = generate_random_bn_data(
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_samples=SAMPLE_SIZE,
            seed=seed,
            standardise=True
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


def main(n_nodes,
         n_edges,
         use_collider_arguments,
         neighbourhood_n_nodes,
         c_set_size,
         search_depth,
         n_runs,
         result_dir):
   
    random_stability(SEED)
    seeds_list = np.random.randint(0, 10000, (n_runs,)).tolist()

    # get all facts and causal graphs for the seeds
    facts_and_cg = dict()
    for seed in tqdm(seeds_list):
        X_s, B_true, cg, facts, elapsed_fact_sourcing = get_facts_and_graphs_for_seed(seed, n_nodes, n_edges)
        facts_and_cg[seed] = (X_s, B_true, cg, facts, elapsed_fact_sourcing)

    start_bsaf_creation = time.time()
    bsaf_builder = BSAFBuilderV2(
        n_nodes=n_nodes,
        include_collider_tree_arguments=use_collider_arguments,
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
        if elapsed_fact_sourcing is None:
            logger.error(f"Model run for seed {seed} timed out. Skipping this seed.")
            continue
        start_orig_ranking = time.time()
        original_ranking_I, best_model_original_ranking = limited_depth_search_best_model(
            steps_ahead=search_depth,
            sorted_arrows=sorted_arrows,
            is_dag=check_arrows_dag,
            get_score=score_original_prefilled
        )
        elapsed_orig_ranking = time.time() - start_orig_ranking
        record_results(
            result_dir=result_dir,
            seed=seed,
            n_nodes=n_nodes,
            n_edges=n_edges,
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
    result_dir = Path(args.path)
    result_dir.mkdir(parents=True, exist_ok=True)

    # Save the arguments to a file
    with open(result_dir / 'args.txt', 'w') as f:
        f.write(str(args))

    logger.info(f"Starting Gradual Causal ABA param evaluation with args: {args}")

    main(
        n_nodes=args.n_nodes,
        n_edges=args.n_edges,
        use_collider_arguments=args.use_collider_arguments == 'true',
        neighbourhood_n_nodes=args.neighbourhood_n_nodes,
        c_set_size=args.c_set_size,
        search_depth=args.search_depth,
        n_runs=args.n_runs,
        result_dir=result_dir
    )
    logger.info("Finished Gradual Causal ABA param evaluation.")
    logger.info(f"Results saved to {result_dir}")
