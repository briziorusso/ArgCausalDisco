#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=128gb:cpu_type=rome
#PBS -N nodes_20_neigh_3_cset_3

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"

python scripts/python_scripts/gradual/find_optimal_params.py \
    --path "./results/gradual/find_optimal_params/nodes_20_neigh_3_cset_3" \
    --n-nodes 20 \
    --n-edges 20 \
    --neighbourhood-n-nodes 3 \
    --c-set-size 3 \
    --search-depth 10 \
    --use-collider-arguments "true" \
    --n-runs 10

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the find_optimal_params experiment script."
    exit 1
fi
