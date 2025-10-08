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

1. **ABAPC-LLM**:
    1. Generate/refresh priors in `priors_assessments.ipynb`.
    2. Run `experiment_llm.py` to create results, reports and produce `results/ABAPC-LLM/merged_*.csv`.
2. **Causal-LLM-BFS**:
    1. Run `causal-llm-bfs/run_heuristic_batch.py`.
    2. Run `python export_bfs_result.py` to produce `results/causal-bfs-*.csv`.
3. **Other Baselines**:
    Run `experiments.py` to produce `results/*.npy`. See [README_experiment.md](README_experiment.md) for details.
4. **Analysis**:
Run `notebooks/experiments_bnlearn.ipynb` and `notebooks/experiments_causenet.ipynb` for the final analysis and plots.
