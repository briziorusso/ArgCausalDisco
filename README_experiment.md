# Unified Experiment Runner (experiments.py)

This script unifies the functionality of the CausaNet and BNLearn experiment drivers into a single CLI with resume/progress tracking.

File: `experiments.py`

- Sources: `--source {causenet|bnlearn}`
- Saves per-run progress CSVs and aggregated summaries (`.npy`)
- Supports resume (`--resume`) and loading previous summaries (`--load_res`)
- Per-run seeds, consistent logging, and filtering over datasets

## Quick Start

- CausaNet graphs (default):
  - `python experiments.py --source causenet --models mpc fgs nt --n_runs 5`
  - `python experiments.py --source causenet --models mpc_llm --prior_json results/llm_constraints/synthetic-desc-consensus.json --n_runs 5`
- BNLearn datasets:
  - `python experiments.py --source bnlearn --names child asia --models mpc fgs --n_runs 5`
- Show options:
  - `python experiments.py --help`

## Outputs

- Logs: `results/log_<version>.log`
- Progress CSVs: `results/progress/<version>/<dataset>__<method>_{dag|cpdag}.csv`
- Summaries: `results/stored_results_<version>.npy` and `results/stored_results_<version>_cpdag.npy`
  - DAG columns include means/stds for: elapsed, nnz, fdr, tpr, fpr, precision, recall, F1, shd, sid
  - CPDAG columns include: elapsed, nnz, fdr, tpr, fpr, precision, recall, F1, shd, sid_low, sid_high

Use `--resume` to continue runs using progress CSVs. Use `--load_res` to start from existing summaries (the script writes backups `*_bkp.npy` when saving).

## CausaNet Mode

- Source: `--source causenet` (or omit; it is the default)
- Graphs folder: `--bifxml_dir synthetic`
- Data simulation: `--simulate_with {internal|pyagrum}` (default: internal)
- Standardisation: default False for CausaNet; override with `--standardise/--no-standardise`

Examples:
- Run 50 reps on all graphs for two models:
  - `python experiments.py --models mpc fgs --n_runs 50`
- Filter by filename metadata (parsed from names like `dag_<N>_nodes_<E>_edges_<heur>_<type>.bifxml`):
  - `--nodes {5|10|15}`
  - `--edges_class {d|1.5d}`
  - `--heur {none|degrees|semantics}`
  - `--type {random|er|sf}`
- Filename-based filters:
  - `--names dag_10_nodes_10_edges_none_er.bifxml`
  - `--include nodes_10` `--glob 'dag_*_nodes_*_edges_*.bifxml'` `--regex 'nodes_10.*edges'`
- Prior-aware MPC:
  - `--models mpc_llm --prior_json results/llm_constraints/synthetic-desc-consensus.json`
  - Missing or empty priors fall back to ordinary MPC under the `MPC-LLM` label.

## BNLearn Mode

- Source: `--source bnlearn`
- Data root: `--bn_data_path datasets`
- Standardisation: default True for BNLearn; override with `--standardise/--no-standardise`
- Dataset names come from `utils/data_utils.py:BIF_FOLDER_MAP` (e.g., child, asia, sachs, alarm, ...)

Examples:
- Run on a subset:
  - `python experiments.py --source bnlearn --names child asia --models mpc fgs nt --n_runs 20`
- Select by substring:
  - `python experiments.py --source bnlearn --include chi --models mpc`

Directory structure expected (example for `child`):
- `<bn_data_path>/bayesian/<size>/child.bif/child.bif`
- Optional PNG at `<bn_data_path>/bayesian/<size>/child.png`

## Models

Pass one or more with `--models`:
- `pc`, `pc_max`, `fgs`, `spc`, `mpc`, `mpc_llm`, `cpc`, `abapc`, `cam`, `nt`, `mcsl`, `ges`, `random`, `random_edge`
- `random`: uniform random DAG with random edge count
- `random_edge`: random DAG matching the true |E|
- `mpc_llm`: Majority PC with causal-learn background knowledge from a consensus prior JSON.

Notes:
- Some models require optional deps (e.g., CDT for `cam`, castle/notears for `nt`/`mcsl`, R setup for CAM). See `cd_algorithms/models.py`.

## Reproducibility

- The script generates a list of seeds per dataset/method. Progress CSVs store the seed used per run for resume.

## Common Options

- `--version <name>`: tag for log/progress/summary files
- `--results_dir <dir>`: output location (default: `results`)
- `--prior_json <path>`: consensus prior JSON for prior-aware baselines such as `mpc_llm`
- `--sample_size <int>`: per-run sample size (default: 5000)
- `--n_runs <int>`: repetitions per dataset/method
- `--device <idx>`: device index forwarded to methods like NOTEARS-MLP
- `--resume`: resume from progress CSVs
- `--load_res`: start with existing summary `.npy` files
- `--save_res/--no-save_res`: control writing summaries

## Example Workflows

- CausaNet, resume a large job with progress and summaries:
  - `python experiments.py --version causenet_bifxml_50rep --models mpc fgs nt --resume`
- BNLearn, small smoke test:
  - `python experiments.py --source bnlearn --names child --models mpc --n_runs 2 --sample_size 1000`

## Troubleshooting

- No datasets selected: ensure filters match; for BNLearn use `--names` or `--include`.
- CAM/NOTEARS errors: verify optional dependencies and R path; see `cd_algorithms/models.py`.
- pyAgrum not installed: keep `--simulate_with internal` (default) in CausaNet mode.

