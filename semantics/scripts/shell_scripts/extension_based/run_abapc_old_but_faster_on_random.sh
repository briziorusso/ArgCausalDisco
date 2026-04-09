#!/bin/bash
#PBS -l walltime=6:00:00
#PBS -l select=1:ncpus=16:mem=128gb:cpu_type=rome
#PBS -N abapc_old_but_faster_on_random

# Setup env variables and the python venv
cd $PBS_O_WORKDIR
. scripts/shell_scripts/setup.sh

module load R
export CUDA_VISIBLE_DEVICES="[]"

python scripts/python_scripts/extension_based/run_abapc_old_but_faster_on_random.py \
    --n-runs 50

if [ $? -ne 0 ]; then
    echo "Error: Failed to run the bnlearn experiment script."
    exit 1
fi
