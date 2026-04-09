#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=8:mem=128gb:cpu_type=rome
#PBS -N find_max_resources

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"

python scripts/python_scripts/gradual/find_max_resources/find_max_resources.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the bnlearn experiment script."
    exit 1
fi
