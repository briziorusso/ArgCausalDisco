# ArgCausalDisco

## Overview

- Causal discovery with ABAPC/Causal ABA + LLM priors.
- Datasets: `bnlearn/*.bifxml`, `synthetic/*.bifxml`.

## Key scripts

- `priors_assessments.ipynb`: generates raw LLM priors and aggregates them.
- `experiment_llm.py`: runs Causal ABA with/without priors and saves CSVs/reports.
- `abapc.py`: runs PC → encodes facts → calls `CausalABA`.
- `causalaba.py`: ASP solver pipeline (clingo) with optional `PriorKnowledge`.
- `join.py`: merges prior JSON and experiment CSVs into `merged_*.csv`.

## Results files

- Raw LLM priors (per-run): `results_synthetic_with_desc.json`, `results_bnlearn_with_desc.json`.
- Aggregated priors (per-dataset): `synthetic.json`, `bnlearn.json`, `child.json`.
- Experiment outputs: `synthetic-results.csv`, `bnlearn-results.csv`, `child-results.csv`.
- Summaries: `synthetic-report.csv`, `bnlearn-report.csv`, `child-report.csv`.
- Merged analytics: `merged_synthetic.csv`, `merged_bnlearn.csv`, `merged_child.csv`.

## Reproduce

1. Generate/refresh priors in `priors_assessments.ipynb`.
2. Run `experiment_llm.py` to create results and reports.
3. Run `python join.py` to produce `merged_*.csv`.
