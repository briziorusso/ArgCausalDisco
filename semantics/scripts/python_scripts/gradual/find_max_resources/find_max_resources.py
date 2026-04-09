"""
This script finds the maximum neighbourhood parameter and the maximum conditioning set size parameters.
The priority is:
    1. collider arguments,
    2. neighbourhood size,
    3. conditioning set size.

This means that it first sets the collider arguments to True,
then finds greatest neighbourhood size with c-set size 0,
then finds greatest c-set size with neighbourhood size determined above.
"""
import sys

sys.path.insert(0, '.')
import sitecustomize

from logger import logger
from pathlib import Path
import json
import subprocess


RESULTS_DIR = Path("./results/gradual/find_max_resources")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_NODES = [20, 11, 8]  # Number of nodes corresponding to datasets asia, sachs, child

MIN_NEIGHBOURHOOD_SIZE = 3
MIN_C_SET_SIZE = 0


def check_resources_sufficient(path,
                               output,
                               n_nodes,
                               use_collider_arguments,
                               neighbourhood_n_nodes,
                               c_set_size):
    try:
        subprocess.run([
            "python", "scripts/python_scripts/gradual/find_max_resources/try_parameters.py",
            "--n-nodes", f"{n_nodes}",
            "--neighbourhood-n-nodes", f"{neighbourhood_n_nodes}",
            "--c-set-size", f"{c_set_size}",
            "--use-collider-arguments", "true" if use_collider_arguments else "false",
            "--path", f"{path}",
            "--output", f"{output}"
        ],
            capture_output=True,
            text=True,
            check=True)
    except subprocess.CalledProcessError as e:
        print("Child failed!")
        print("Return code:", e.returncode)
        print("Error output:", e.stderr)
        raise Exception("Failed to run the subprocess for checking resources. Here is the error: " + e.stderr)

    with open(output, 'r') as f:
        output_json = json.load(f)
        memory_usage_exceeded = output_json.get('memory_usage_exceeded')
        assert isinstance(memory_usage_exceeded, bool), "memory_usage_exceeded must be a boolean"
    return memory_usage_exceeded


def find_max_c_set_size(n_nodes,
                        min_c_set_size,
                        use_collider_arguments,
                        neighbourhood_n_nodes,
                        path,
                        output):
    """
    Finds the maximum c-set size for given n_nodes, use_collider_arguments and neighbourhood_n_nodes.
    It starts with c_set_size = min_c_set_size and increases it until memory usage exceeds the limit.
    """
    c_set_size = min_c_set_size
    while True:
        if c_set_size > n_nodes - 2:
            logger.info(f"Conditioning set size {c_set_size} exceeds number of nodes {n_nodes}. "
                        f"Breaking loop.")
            c_set_size = n_nodes - 2
            break

        memory_usage_exceeded = check_resources_sufficient(
            path=path,
            output=output,
            n_nodes=n_nodes,
            use_collider_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            c_set_size=c_set_size)

        if memory_usage_exceeded:
            # If memory usage exceeded, break and go one step back
            logger.info(f"Memory usage exceeded for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                        f"c_set_size={c_set_size}.")
            c_set_size -= 1
            break
        else:
            # If memory usage is fine, try with more resources by increasing c_set_size
            logger.info(f"Memory usage is fine for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                        f"c_set_size={c_set_size}.")
            c_set_size += 1
    return c_set_size


def find_max_neighbourhood_size(n_nodes,
                                min_neighbourhood_n_nodes,
                                use_collider_arguments,
                                c_set_size,
                                path,
                                output):
    """ Finds the maximum neighbourhood size for given n_nodes, use_collider_arguments and c_set_size.
    It starts with neighbourhood_n_nodes = min_neighbourhood_n_nodes and increases it
    until memory usage exceeds the limit.
    """
    neighbourhood_n_nodes = min_neighbourhood_n_nodes
    while True:
        if neighbourhood_n_nodes > n_nodes:
            logger.info(f"Neighbourhood size {neighbourhood_n_nodes} exceeds number of nodes {n_nodes}. "
                        f"Breaking loop.")
            neighbourhood_n_nodes = n_nodes
            break

        memory_usage_exceeded = check_resources_sufficient(
            path=path,
            output=output,
            n_nodes=n_nodes,
            use_collider_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            c_set_size=c_set_size)
        if memory_usage_exceeded:
            # If memory usage exceeded, break and go one step back
            logger.info(f"Memory usage exceeded for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                        f"c_set_size={c_set_size}.")
            neighbourhood_n_nodes -= 1
            break
        else:
            # If memory usage is fine, try with more resources by increasing neighbourhood size
            logger.info(f"Memory usage is fine for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                        f"c_set_size={c_set_size}.")
            neighbourhood_n_nodes += 1
    return neighbourhood_n_nodes


def main():
    path = RESULTS_DIR / 'run_metadata.csv'
    output = RESULTS_DIR / 'output.json'

    for n_nodes in N_NODES:
        logger.info(f"Processing case with {n_nodes} nodes")
        for use_collider_arguments in [True, False]:
            # Find maximum neighbourhood size for the current collider arguments
            max_neighbourhood_n_nodes = find_max_neighbourhood_size(
                n_nodes=n_nodes,
                min_neighbourhood_n_nodes=MIN_NEIGHBOURHOOD_SIZE,
                use_collider_arguments=use_collider_arguments,
                c_set_size=MIN_C_SET_SIZE,
                path=path,
                output=output)
            logger.info(f"Maximum neighbourhood_n_nodes found: {max_neighbourhood_n_nodes} for "
                        f"use_collider_arguments={use_collider_arguments}.")
            # If neighbourhood size is less than minimum, then no resource is enough
            if max_neighbourhood_n_nodes < MIN_NEIGHBOURHOOD_SIZE:
                logger.info(f"Memory usage exceeded for minimum resource parameters for {n_nodes} nodes.")
                continue
            else:
                # Find maximum c_set_size for each neighbourhood size
                for neighbourhood_n_nodes in range(MIN_NEIGHBOURHOOD_SIZE, max_neighbourhood_n_nodes + 1):
                    logger.info(f"Finding maximum c_set_size for n_nodes={n_nodes}, "
                                f"use_collider_arguments={use_collider_arguments}, "
                                f"neighbourhood_n_nodes={neighbourhood_n_nodes}.")
                    max_c_set_size = find_max_c_set_size(
                        n_nodes=n_nodes,
                        min_c_set_size=MIN_C_SET_SIZE + 1,  # Start from 1 to avoid c_set_size = 0 which has already been checked
                        use_collider_arguments=use_collider_arguments,
                        neighbourhood_n_nodes=neighbourhood_n_nodes,
                        path=path,
                        output=output)
                    logger.info(f"Maximum c_set_size found: {max_c_set_size} for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                                f"use_collider_arguments={use_collider_arguments}.")
        logger.info(f"Completed processing for {n_nodes} nodes\n")


if __name__ == "__main__":
    main()
