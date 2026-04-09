#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=16:mem=32gb:cpu_type=rome
#PBS -N runtime_sourcing_facts_bnlearn_graphs

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"

python scripts/python_scripts/extension_based/runtime_sourcing_facts_bnlearn_graphs.py \
    --n-runs 50

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the bnlearn experiment script."
    exit 1
fi
