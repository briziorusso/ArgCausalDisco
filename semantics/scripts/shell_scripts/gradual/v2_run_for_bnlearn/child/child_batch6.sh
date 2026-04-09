#!/bin/bash
#PBS -l walltime=13:00:00
#PBS -l select=1:ncpus=8:mem=128gb:cpu_type=rome
#PBS -N v2_bnlearn_child_batch6

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"


python scripts/python_scripts/gradual/v2_run_for_bnlearn.py \
    --path "./results/gradual/v2_run_for_bnlearn/child/batch6" \
    --search-depth 12 \
    --name "child" \
    --neighbourhood-n-nodes 3 \
    --c-set-size 3 \
    --n-runs 50 \
    --batch-number 6 \
    --total-batches 10

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the find_optimal_params experiment script."
    exit 1
fi
