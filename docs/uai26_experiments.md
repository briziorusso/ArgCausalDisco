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
- `pc`, `pc_max`, `fgs`, `spc`, `mpc`, `mpc_llm`, `cpc`, `abapc`, `cam`, `nt`, `mcsl`, `ges`, `grasp`, `boss`, `random`, `random_edge`
- `mpc_llm`: MPC with consensus LLM constraints injected as background knowledge through `--prior_json`
- `random`: uniform random DAG with random edge count
- `random_edge`: random DAG matching the true |E|

Notes:
- Some models require optional deps (e.g., CDT for `cam`, castle/notears for `nt`/`mcsl`, R setup for CAM). See `cd_algorithms/models.py`.

## Reproducibility

- The script generates a list of seeds per dataset/method. Progress CSVs store the seed used per run for resume.

## Common Options

- `--version <name>`: tag for log/progress/summary files
- `--results_dir <dir>`: output location (default: `results`)
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

## UAI 2026 Final Experiment Set

Run these commands from the repository root:

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-worktrees/llm-uai2026-freeze
```

The paper uses Wilks G2 CI tests with `alpha=0.01`. The result tag used by the notebooks is:

```text
g2a01
```

### 1. LLM Priors

Start the LiteLLM proxy for Gemini-backed runs:

```bash
export GEMINI_API_KEY="your_key"
litellm --config priors/litellm.yaml --detailed_debug
```

Generate Gemini priors with descriptions and without descriptions:

```bash
python scripts/run_llm_priors.py --bifxml_dir synthetic --output_prefix synthetic-desc --prior_model gemini-2.5-flash --parse_model gemini-2.5-flash-lite --repeats 5
python scripts/run_llm_priors.py --bifxml_dir synthetic --output_prefix synthetic --prior_model gemini-2.5-flash --parse_model gemini-2.5-flash-lite --repeats 5 --exclude_descriptions
python scripts/run_llm_priors.py --bifxml_dir bnlearn --output_prefix bnlearn-desc --prior_model gemini-2.5-flash --parse_model gemini-2.5-flash-lite --repeats 5
python scripts/run_llm_priors.py --bifxml_dir bnlearn --output_prefix bnlearn --prior_model gemini-2.5-flash --parse_model gemini-2.5-flash-lite --repeats 5 --exclude_descriptions
```

The prior assessment notebook is available at:

```text
notebooks/priors_assessments.ipynb
```

GPT-5-mini ablation results are expected at:

```text
results/llm_constraints/synthetic-desc-gpt5mini.json
results/llm_constraints/synthetic-desc-gpt5mini-consensus.json
```

If rerunning that ablation, configure the appropriate LiteLLM/OpenAI alias first, then use the same `scripts/run_llm_priors.py` interface with `--output_prefix synthetic-desc-gpt5mini`.

### 2. Statistical And Search Baselines

The non-ABAPC baselines analysed in the paper are run with `experiments.py`: Random, FGS, NOTEARS-MLP, GRaSP, BOSS, MPC, and MPC-LLM. The version names below are the ones loaded by the paper notebooks. Some names are historical, but should be kept unless the notebooks are updated.

#### CauseNet

Run the CauseNet baseline summaries:

```bash
python experiments.py --source causenet --models random --version test_rnd_spc --test_name g2 --test_alpha 0.01 --n_runs 50 --resume
python experiments.py --source causenet --models fgs nt --version causenet_base --test_name g2 --test_alpha 0.01 --n_runs 50 --resume
python experiments.py --source causenet --models grasp boss --version causenet_boss_grasp --test_name g2 --test_alpha 0.01 --n_runs 50 --resume
```

Run the final CauseNet MPC/MPC-LLM baseline with the same consensus Gemini constraints used by ABAPC-LLM:

```bash
python experiments.py --source causenet --models mpc mpc_llm --prior_json results/llm_constraints/synthetic-desc-consensus.json --version causenet_mpc_desc_g2a01 --test_name g2 --test_alpha 0.01 --n_runs 50 --resume
```

Run the GPT-5-mini MPC-LLM ablation on CauseNet:

```bash
python experiments.py --source causenet --models mpc_llm --prior_json results/llm_constraints/synthetic-desc-gpt5mini-consensus.json --version causenet_mpc_llm_gpt5mini_desc_g2a01 --test_name g2 --test_alpha 0.01 --n_runs 50 --resume
```

#### BNLearn

Run the five small-graph baseline summaries:

```bash
python experiments.py --source bnlearn --names asia cancer earthquake sachs survey --models fgs nt --version bnlearn_big_fgs_nt --n_runs 50 --resume
python experiments.py --source bnlearn --names asia cancer earthquake sachs survey --models random mpc --version bnlearn_big_rnd_mpc --test_name g2 --test_alpha 0.01 --n_runs 50 --resume
```

Run the `child` baseline summaries:

```bash
python experiments.py --source bnlearn --names child --models random --version bnlearn_child_base --n_runs 50 --resume
python experiments.py --source bnlearn --names child --models fgs nt mpc --version bnlearn_child_base2 --n_runs 50 --resume
```

Run GRaSP and BOSS on the six paper `bnlearn` graphs:

```bash
python experiments.py --source bnlearn --names asia cancer earthquake sachs survey child --models grasp boss --version bnlearn_boss_grasp --n_runs 50 --resume
```

Run the final BNLearn MPC/MPC-LLM baseline with the same consensus Gemini constraints used by ABAPC-LLM:

```bash
python experiments.py --source bnlearn --names asia cancer earthquake sachs survey child --models mpc mpc_llm --prior_json results/llm_constraints/bnlearn-desc-consensus.json --version bnlearn_mpc_desc_g2a01 --test_name g2 --test_alpha 0.01 --n_runs 50 --resume
```

These commands write summaries such as:

```text
results/stored_results_test_rnd_spc.npy
results/stored_results_causenet_base.npy
results/stored_results_causenet_boss_grasp.npy
results/stored_results_causenet_mpc_desc_g2a01.npy
results/stored_results_bnlearn_big_fgs_nt.npy
results/stored_results_bnlearn_big_rnd_mpc.npy
results/stored_results_bnlearn_child_base.npy
results/stored_results_bnlearn_child_base2.npy
results/stored_results_bnlearn_boss_grasp.npy
results/stored_results_bnlearn_mpc_desc_g2a01.npy
results/stored_results_causenet_mpc_llm_gpt5mini_desc_g2a01.npy
```

Progress CSVs are stored under `results/progress/<version>/`; use `--resume` to continue interrupted runs.

### 3. ABAPC And ABAPC-LLM

Run the final ABAPC/ABAPC-LLM comparisons:

```bash
python experiment_llm.py --types synthetic-desc bnlearn-desc synthetic bnlearn --test_name g2 --test_alpha 0.01 --output_suffix g2a01 --resume
```

Run the GPT-5-mini ABAPC-LLM ablation:

```bash
python experiment_llm.py --types synthetic-desc-gpt5mini --test_name g2 --test_alpha 0.01 --output_suffix g2a01 --resume
```

`experiment_llm.py` checkpoints after each completed graph:

```text
results/ABAPC-LLM/checkpoints/
```

If a job finishes checkpoints but fails while writing final files, rebuild the final CSV/report/merged files without rerunning:

```bash
python experiment_llm.py --types synthetic-desc bnlearn-desc synthetic bnlearn synthetic-desc-gpt5mini --test_name g2 --test_alpha 0.01 --output_suffix g2a01 --finalize-only
```

Expected merged outputs include:

```text
results/ABAPC-LLM/merged_synthetic-desc_g2a01.csv
results/ABAPC-LLM/merged_bnlearn-desc_g2a01.csv
results/ABAPC-LLM/merged_synthetic_g2a01.csv
results/ABAPC-LLM/merged_bnlearn_g2a01.csv
results/ABAPC-LLM/merged_synthetic-desc-gpt5mini_g2a01.csv
```

### 4. Causal-LLM-BFS

Run Causal-LLM-BFS from its submodule-style folder, then export compact CSV results from the repository root:

```bash
cd causal-llm-bfs
python run_heuristic_batch.py --heuristic_dir ../bnlearn --alg llm_bfs_with_statistics --n_samples 5000 --logdir logs
python run_heuristic_batch.py --heuristic_dir ../synthetic --alg llm_bfs_with_statistics --n_samples 5000 --logdir logs --exclude_desc
cd ..
python scripts/export_bfs_result.py
```

Tracked paper inputs include:

```text
results/causal-bfs-bnlearn-results.csv
results/causal-bfs-synthetic-results.csv
```

### 5. Collect Paper Outputs

After these experiment runs complete, collect the paper tables, notebooks, and figure files using:

```text
docs/uai26_collect_results.md
```
