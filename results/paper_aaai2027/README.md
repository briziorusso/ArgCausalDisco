# Frozen AAAI-27 evidence

This directory is the curated empirical release for the OptABA-PC paper.

- `frozen/results/` contains the allowlisted per-seed repair, graph, fact, and
  contestability evidence needed by the production scripts.
- `derived/tables/` is the frozen output of the final analysis pipeline.
- `recomputed/` is intentionally untracked working space for regenerated
  tables and audits.
- `manifest.json` records the protocol, included result versions, completion
  exceptions, file counts, byte counts, and tree hashes.
- `FROZEN_SHA256SUMS` covers every per-seed frozen input.
- `SHA256SUMS` covers the complete curated release except itself.

The primary synthetic reconstruction results use one edge per node. Dense
two-edge-per-node results are retained because the paper reports density
sensitivity and uses the five-node dense instances in its bounded exact
contestation audit. See `CONTESTABILITY_SCOPE.md`.

No corrupted-fact, Shapley-PC, smoke, 10-repetition, or development-alpha run
is part of this directory.

Validate without modifying any artefact:

```bash
python scripts/validate_release_artifacts.py
```

Regenerate the final tables into a separate directory:

```bash
python scripts/build_final_experiment_tables.py \
  --results-dir results/paper_aaai2027/frozen/results \
  --mcs-results-dir results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked \
  --out-dir results/paper_aaai2027/recomputed/tables
```
