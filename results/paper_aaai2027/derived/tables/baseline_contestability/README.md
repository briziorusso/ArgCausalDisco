# Baseline enforcement and contestability comparisons

This directory keeps three deliberately different comparisons.

- `abapc_optimality_summary.csv` compares ABA-PC's heuristic release cost with
  the independently re-solved and proved OptABA-PC optimum on the identical
  facts and weights used by the 250-instance contestability audit.
- `mpc_enforcement_audit.csv` checks whether each returned Majority-PC CPDAG
  satisfies every outcome in its recorded matched G2 trace.  It is an
  enforcement-gap diagnostic, not a contestability margin.
- `aspcr_native/aspcr_contestability_challenges.csv` contains one exact
  forced-fact optimization for every fact failed by the saved ASPCR-DAG
  optimum.  The challenged fact is hard and all other native Bayesian
  log-weighted facts remain soft.

The generated TeX tables are `table_contestability_method_overview.tex`,
`table_contestability_margin_comparison.tex`, and
`table_mpc_enforcement_gap.tex`. JSON manifests record input and
output hashes, solver provenance, completion counts, and the distinction in
semantics.

Reproduce from the repository root after activating the pinned environment:

```bash
python scripts/audit_mpc_test_enforcement.py
python scripts/run_aspcr_contestability.py --workers 4 --timeout-sec 300
python scripts/build_baseline_contestability_tables.py
```

The ASPCR runner is resumable by stable fact id.  It reconstructs and proves
each unforced baseline objective before accepting challenge results, and it
does not rerun the Bayesian tests because the complete tested trace and weights
are already stored in the validated ASPCR artifacts.
