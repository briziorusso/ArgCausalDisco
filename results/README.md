# Result Artefacts

This directory keeps the compact artefacts needed to reproduce the UAI 2026
tables and paper checks.

Tracked artefacts:

- `stored_results_*.npy` summary arrays consumed by the table scripts.
- `ABAPC-LLM/*g2a01*.csv` and related final tagged CSV exports.
- `causal-bfs-*-results.csv` LLM-BFS exports used as one-run reference rows.
- `tables/final/` numbered paper tables and their compact check files.
- `tables/synthetic*_structural_summary_*.csv` and
  `tables/structural*_tests_abapcllm_bh_*.csv` summary/test inputs.

Ignored local artefacts:

- progress folders, checkpoints, estimated graphs, and blocked-required
  audit/debug folders produced while running experiments.
- notebook-exported figures under `results/figs/`.
- run logs, `.out` files, backups, and archived table snapshots.

Ignored artefacts are intentionally not deleted by cleanup commits; they can be
regenerated or retained locally without affecting the public result history.
