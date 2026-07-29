# OptABA-PC: optimal repair for contestable causal discovery

OptABA-PC is a solver-backed extension of Causal Assumption-based Argumentation
for finite-sample causal discovery. Conditional-independence (CI) tests can be
mutually inconsistent, so no DAG satisfies them all. ABA-PC restores coherence
by releasing facts in a fixed strength order. OptABA-PC instead makes the
released facts an explicit minimum-cost correction set and computes an optimum
with Answer Set Programming (ASP) weak constraints.

The implementation provides:

- an incremental Bayes-ball encoding of CI constraints over DAGs;
- optimal weighted CI-fact repair with a stable-model/MCS projection;
- the retained and released fact sets and every compatible CPDAG/DAG;
- executable checks of the paper's formal properties;
- matched ABA-PC, MPC, FGS, and DAG-restricted ASPCR experiments;
- compatibility, contestability, statistical-analysis, and table pipelines.

The central guarantee is about evidence correspondence: every graph returned by
ABA-PC or OptABA-PC satisfies every CI fact retained by its repair. OptABA-PC
additionally minimizes total release cost. MPC and FGS are graph-estimation
references; ASPCR-DAG minimizes a different graph--test disagreement objective.

## Repository map

| Path | Purpose |
| --- | --- |
| `causalaba_increm.py` | Incremental Bayes-ball Causal ABA program and ABA-PC repair |
| `causalaba_mus.py` | OptABA-PC weak-constraint objective and optimal correction sets |
| `scripts/wc_opt_strategy_sweep.py` | Matched ABA-PC/OptABA-PC experiments |
| `scripts/run_matched_baseline_experiments.py` | MPC, FGS, and ASPCR-DAG experiments |
| `cd_algorithms/run_aspcr_csvdata.R` | Auditable DAG-restricted ASPCR wrapper |
| `verify_formal_properties.py` | Root entry point for the dependency-free formal checks |
| `scripts/verify_formal_properties.py` | Formal-property checker implementation |
| `scripts/validate_matched_aspcr_results.py` | Per-seed ASPCR graph/fact/provenance validator |
| `scripts/build_final_experiment_tables.py` | Final statistics, compatibility audit, and tables |
| `scripts/run_contestability_experiments.py` | OptABA-PC hard-retention contestations |
| `scripts/audit_mpc_test_enforcement.py` | MPC full-trace enforcement audit |
| `scripts/run_aspcr_contestability.py` | ASPCR-DAG failed-fact sensitivity analysis |
| `tests/` | Deterministic unit and pipeline tests |
| `configs/paper_aaai2027.json` | Machine-readable final protocol and artefact versions |
| `results/paper_aaai2027/` | Curated frozen evidence, derived outputs, and checksums |

## Quick verification

Create the pinned Python environment and run the fast checks:

```bash
conda env create -f environment.yml
conda activate optaba-pc-repro
python -m pytest -q tests
python verify_formal_properties.py
python scripts/validate_release_artifacts.py
```

The formal checker is dependency-free. It exhaustively compares the implemented
Bayes-ball rules with independent active-path and Shachter references on 571
labelled DAGs and 26,370 ordered queries, then checks stable-model projection,
optimal MCS costs, private conflicts, hard retention, and exact contestation
margins on the paper examples.

## Final empirical protocol

The primary study uses 50 seeds (2026--2075) and 5,000 observations per seed.
Cancer, Earthquake, Survey, and Asia use the bundled bnlearn BIF networks.
Synthetic ER and SF DAGs have five or eight nodes and one edge per node in the
primary experiment; two-edge-per-node runs are a reported density-sensitivity
analysis. ABA-PC, OptABA-PC, and MPC use the G-squared CI test at
`alpha=0.01`. FGS uses SEM-BIC. ASPCR-DAG retains its published Bayesian
log-weighted CI configuration on the same samples.

ABA-PC and OptABA-PC receive identical CI facts and weights. Their runs use the
incremental encoding, no conditioning-set weight multiplier, a 300-second solve
limit, a separate 120-second compatible-graph evaluation limit, four clingo
threads, branch-and-bound optimization, and `optN` enumeration.

No injected fact corruption, Shapley-PC result, smoke run, 10-repetition run, or
development `alpha=0.05` experiment belongs to the released evidence.

## Reproducing the paper artefacts

Detailed installation, external ASPCR setup, exact commands, expected partial
evaluations, and table regeneration are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
The curated result index is in
[`results/paper_aaai2027/README.md`](results/paper_aaai2027/README.md), and every
released artefact is covered by the accompanying SHA-256 inventories.

To rebuild the paper tables directly from the frozen evidence:

```bash
python scripts/build_final_experiment_tables.py \
  --results-dir results/paper_aaai2027/frozen/results \
  --mcs-results-dir results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked \
  --out-dir results/paper_aaai2027/recomputed/tables
```

## Software and licence

The final environment is CPU-only Python 3.12.1 with clingo 5.8.0. FGS requires
OpenJDK 19 and `py-causal`; ASPCR-DAG requires R 4.1.2 and the separately
distributed Hyttinen--Eberhardt--Järvisalo code package. Exact package versions
are pinned in `environment.yml`, `requirements-repro.txt`, and
`REPRODUCIBILITY.md`. Before rerunning FGS, install its wrapper after environment
creation with `python -m pip install --no-deps -r requirements-fgs.txt`; the
compatible Java bridge is already pinned in the main environment.

Unless a file states otherwise, this repository is released under the Apache
License 2.0. Third-party datasets and external implementations retain their own
terms.
