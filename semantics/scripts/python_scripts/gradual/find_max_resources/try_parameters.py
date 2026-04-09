"""
Run BSAF creation with certain set of parameters and check if resources are sufficient
"""
import sys

sys.path.insert(0, '.')
import sitecustomize

from src.gradual.scalable.bsaf_builder_v2 import BSAFBuilderV2, check_memory_usage_gb
from src.utils.resource_utils import MemoryUsageExceededException
import time
import json
import pandas as pd
from logger import logger
import argparse
from pathlib import Path


def update_and_save_data(data,
                         path,
                         n_nodes,
                         use_collider_arguments,
                         neighbourhood_n_nodes,
                         c_set_size,
                         memory_usage_exceeded,
                         memory_usage,
                         elapsed_bsaf_creation):
    data = pd.concat([data, pd.DataFrame({
        'n_nodes': [n_nodes],
        'use_collider_arguments': [use_collider_arguments],
        'neighbourhood_n_nodes': [neighbourhood_n_nodes],
        'c_set_size': [c_set_size],
        'memory_usage_exceeded': [memory_usage_exceeded],
        'memory_usage_pct': [memory_usage],
        'bsaf_creation_elapsed_time': [elapsed_bsaf_creation]
    })], ignore_index=True)
    data.to_csv(path, index=False)


def check_resources_sufficient(data,
                               path,
                               n_nodes,
                               use_collider_arguments,
                               neighbourhood_n_nodes,
                               c_set_size):
    start_bsaf_creation = time.time()
    memory_usage_exceeded = False
    try:
        bsaf_builder = BSAFBuilderV2(
            n_nodes=n_nodes,
            include_collider_tree_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            max_conditioning_set_size=c_set_size)  # everything is maximal for given n_nodes, full scale
        bsaf = bsaf_builder.create_bsaf()
        memory_usage = check_memory_usage_gb()
    except MemoryUsageExceededException as e:
        logger.info(f"neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                    f"c_set_size={c_set_size}. Error: {e}")
        memory_usage = check_memory_usage_gb()
        memory_usage_exceeded = True

    elapsed_bsaf_creation = time.time() - start_bsaf_creation

    update_and_save_data(data,
                         path,
                         n_nodes,
                         use_collider_arguments,
                         neighbourhood_n_nodes,
                         c_set_size,
                         memory_usage_exceeded,
                         memory_usage,
                         elapsed_bsaf_creation)

    return memory_usage_exceeded


def parse_args():
    parser = argparse.ArgumentParser(description="Find maximum resources for BSAF creation.")
    parser.add_argument('-n', '--n-nodes', type=int, help='Number of nodes in the causal graph.')
    parser.add_argument('-l', '--neighbourhood-n-nodes', type=int, help='Neighbourhood size to test.')
    parser.add_argument('-c', '--c-set-size', type=int, help='Conditioning set size to test.')
    parser.add_argument('-u', '--use-collider-arguments', type=str, choices=['true', 'false'],
                        help='Whether to use collider arguments (true/false).')
    parser.add_argument('-p', '--path', type=str, help='Path to save results CSV file.')
    parser.add_argument('-o', '--output', type=str, help='Path to save output json file.')
    return parser.parse_args()


def main():
    args = parse_args()
    if Path(args.path).exists():
        data = pd.read_csv(args.path)
    else:
        data = pd.DataFrame(columns=['n_nodes',
                                     'use_collider_arguments',
                                     'neighbourhood_n_nodes',
                                     'c_set_size',
                                     'memory_usage_exceeded',
                                     'memory_usage_pct',
                                     'bsaf_creation_elapsed_time'])
    use_collider_arguments = args.use_collider_arguments == 'true'
    memory_usage_exceeded = check_resources_sufficient(
        data=data,
        path=args.path,
        n_nodes=args.n_nodes,
        use_collider_arguments=use_collider_arguments,
        neighbourhood_n_nodes=args.neighbourhood_n_nodes,
        c_set_size=args.c_set_size)

    output_data_json = {'memory_usage_exceeded': memory_usage_exceeded}
    with open(args.output, 'w') as f:
        json.dump(output_data_json, f, indent=4)


if __name__ == "__main__":
    main()
