#!/bin/bash

cd $PBS_O_WORKDIR
export PYTHONPATH=.
# Read and export environment variables from .env file
if [ -f "${PBS_O_WORKDIR}/.env" ]; then
    echo "Loading environment variables from .env file..."
    while IFS= read -r line; do
        # Skip empty lines and comments (lines starting with #)
        if [ -n "$line" ] && ! echo "$line" | grep -q '^#'; then
            # Export the variable
            export "$line"
            echo "Exported: $line"
        fi
    done < "${PBS_O_WORKDIR}/.env"
else
    echo "env file not found. Exiting."
    exit 1
fi

# Set up conda
eval "$(~/miniforge3/bin/conda shell.bash hook)"

source activate "aba-env"

