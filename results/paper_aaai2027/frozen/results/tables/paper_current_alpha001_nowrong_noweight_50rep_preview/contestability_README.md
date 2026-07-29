# Empirical contestability sidecar

This directory is the output target for the contestability experiment described
in `EmpiricalContestabilityHandoff.md`.  The existing matched-baseline CSV and
TeX files are inputs/canonical artifacts and are not modified.

## Requirements

- Frozen OptABA-PC `summary.json`, emitted `wc_inc_mus_lex_bb_optN_*.lp`, and
  `matched_baseline_fact_records.csv` artifacts already present in this
  worktree.
- The `aba-env` Python interpreter, which supplies clingo 5.8.0.
- The five frozen datasets and seeds 2026--2075. Eight-node datasets are
  deliberately rejected by the default scope.

The source summaries contain accepted counts and weights but not accepted fact
identities. The runner therefore re-proves each emitted LP, validates its
witness against both published quantities, and freezes the reconstructed
witness in the evidence JSONL before running challenges. Resuming reuses that
frozen witness by source-LP SHA-256.

## Input inventory (fast/read-only)

From the repository root:

```bash
python scripts/run_contestability_experiments.py --dry-run
```

The frozen summaries contain 742 expected challenges: Cancer (5), 207;
Earthquake (5), 4; Survey (6), 507; ER (5), 0; and SF (5), 24.

## Long run

Run the solver stage in `tmux`:

```bash
tmux new-session -s contestability
cd .
set -o pipefail
python \
  scripts/run_contestability_experiments.py \
  2>&1 | tee results/paper_aaai2027/frozen/results/tables/paper_current_alpha001_nowrong_noweight_50rep_preview/contestability_run.log
```

Detach with `Ctrl-b d`. Re-running the same command resumes from atomically
checkpointed challenge/evidence rows. Use `--retry-unproved` to retry explicit
timeout/unproved rows, normally together with a larger `--timeout-sec`.

The solver stage writes:

- `contestability_challenges.csv`: one row per baseline-released fact;
- `contestability_solve_evidence.jsonl`: frozen baseline witnesses and
  per-challenge solver evidence; and
- `contestability_manifest.json`: scope, provenance, validation, counts, and
  file hashes.

## Table build (fast)

After the manifest reports `"status": "complete"`:

```bash
python scripts/build_contestability_table.py
```

This writes `contestability_dataset_summary.csv` and
`table_empirical_contestability.tex`, then adds their hashes to the manifest.
Only rows with proved baseline and forced optima contribute numerical margins.
Timeout, infeasible, and other unproved challenges remain explicit counts; the
ER (5) zero-challenge row remains in the table.
