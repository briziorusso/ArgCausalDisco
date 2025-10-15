# ArgCausalDisco

## Overview

- Causal discovery with ABAPC/Causal ABA + LLM priors.
- Datasets: `bnlearn/*.bifxml`, `synthetic/*.bifxml`.

## Key scripts

- `priors_assessments.ipynb`: generates raw LLM priors and aggregates them.
- `experiment_llm.py`: runs Causal ABA with/without priors and saves CSVs/reports.
- `abapc.py`: runs PC → encodes facts → calls `CausalABA`.
- `causalaba.py`: ASP solver pipeline (clingo) with optional `PriorKnowledge`.
- `causal-llm-bfs/run_heuristic_batch.py`: runs the Causal-LLM-BFS experiment.
- `export_bfs_result.py`: exports results from `causal-llm-bfs` to `results/causal-bfs-*.csv`.

## Results files

- Raw LLM priors (per-run): `results/llm_constraints/synthetic.json`, `results/llm_constraints/bnlearn-desc.json`.
- Aggregated priors (per-dataset): `results/llm_constraints/synthetic-consensus.json`, `results/llm_constraints/bnlearn-desc-consensus.json`.
- ABAPC-LLM Experiment outputs: `results/ABAPC-LLM/synthetic-results.csv`, `results/ABAPC-LLM/bnlearn-desc-results.csv`.
- Summaries: `results/ABAPC-LLM/synthetic-report.csv`, `results/ABAPC-LLM/bnlearn-desc-report.csv`.
- Merged analytics: `results/ABAPC-LLM/merged_synthetic.csv`, `results/ABAPC-LLM/merged_bnlearn-desc.csv`.
- Causal-LLM-BFS results: `results/causal-bfs-bnlearn-results.csv`, `results/causal-bfs-synthetic-results.csv`.

## Reproduce

### Setup

1. Clone the repo and `cd` into it.
2. Create and activate a Python virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Install `clingo` (ASP solver):
    - Follow instructions at https://potassco.org/clingo/ to install `clingo`.
    - Ensure `clingo` is in your system PATH (you can check by running `clingo --version` in your terminal).

### LLM configuration

Set environment variables for Gemini API key and activate LiteLLM proxy:
```bash
export GEMINI_API_KEY="your_api_key"
litellm --config priors/litellm.yaml --detailed_debug
```

### Run experiments

1. **ABAPC-LLM**:
    1. Generate/refresh priors in `priors_assessments.ipynb`.
    2. Run `experiment_llm.py` to create results, reports and produce `results/ABAPC-LLM/merged_*.csv`.
2. **Causal-LLM-BFS**:
    1. `cd` to `causal-llm-bfs/`.
    2. Run `run_heuristic_batch.py` with arguments. The script will check for already run results in the log and skip those. It will only stop until causal-bfs produce an DAG prediction. To exclude variable descriptions from the prompts, add the `--exclude_desc` flag. E.g.
        ```bash
        python run_heuristic_batch.py --heuristic_dir ../bnlearn --alg llm_bfs_with_statistics --n_samples 5000 --logdir logs
        python run_heuristic_batch.py --heuristic_dir ../synthetic --alg llm_bfs_with_statistics --n_samples 5000 --logdir logs --exclude_desc
        ```
    3. `cd` back to the project root and run `python export_bfs_result.py` to produce `results/causal-bfs-*.csv`.
3. **Other Baselines**:
    Run `experiments.py` to produce `results/*.npy`. See [README_experiment.md](README_experiment.md) for details.
4. **Analysis**:
    Run `notebooks/experiments_bnlearn.ipynb`, `notebooks/experiments_causenet.ipynb`, `notebooks/constraints_comparsion.ipynb` for the final analysis and plots.
