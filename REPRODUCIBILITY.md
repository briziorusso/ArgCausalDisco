# Reproducing the OptABA-PC experiments

This document fixes the software, data, method settings, commands, and expected
completion state for the paper. Commands assume the repository root as the
working directory. The machine-readable counterpart is
`configs/paper_aaai2027.json`.

## 1. Execution environment

The reported runs were CPU-only on Ubuntu 22.04.5 LTS (Linux 6.8.0-124,
x86-64), using one Intel Xeon w5-2455X socket with 12 physical cores and 24
hardware threads, a 3.20 GHz base frequency, a 4.60 GHz maximum frequency, and
128 GB RAM (125 GiB visible). No GPU was used.

The primary environment used Python 3.12.1, clingo 5.8.0, NumPy 1.26.4,
pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1, NetworkX 3.4.2,
causal-learn 0.1.4.1, pgmpy 1.0.0, py-causal 1.2.1, and OpenJDK 19.0.2.
All direct Python dependencies are pinned in `requirements-repro.txt`.

```bash
conda env create -f environment.yml
conda activate optaba-pc-repro
python -m pytest -q tests
python verify_formal_properties.py
```

The formal checker itself uses only the Python standard library.

The historical `pycausal==1.2.1` metadata requests the obsolete distribution
name `javabridge`; the Python 3.12-compatible distribution is
`python-javabridge==4.0.4`, which provides the same `javabridge` module. Install
the FGS wrapper without re-resolving that stale dependency declaration:

```bash
python -m pip install --no-deps -r requirements-fgs.txt
python -c "import javabridge, pycausal"
```

## 2. Data and shared protocol

The repository contains the BIF definitions for Cancer, Earthquake, Survey,
and Asia. Each seed forward-samples a fresh 5,000-row dataset from the fixed
network and CPTs. Each synthetic seed generates both a new graph and its
categorical data-generating distribution. No downloaded observational dataset
is required.

The final seeds are 2026--2075. The primary synthetic ER/SF configuration uses
one edge per node; two edges per node are used only for the reported density
sensitivity and the bounded five-node contestability audit. ABA-PC, OptABA-PC,
and MPC use G-squared CI tests at `alpha=0.01`. There is no injected fact
corruption and no conditioning-set-size multiplier.

The `alpha001` token in historical result-version names is only a legacy label.
The saved metadata and configurations are authoritative: the shared G-squared
experiments use 0.01, while ASPCR uses its native Bayesian configuration.

The method settings were fixed from the published/default implementations and
the preceding Causal ABA protocol; no validation-set or reconstruction-score
tuning was performed:

| Method | Role and final settings |
| --- | --- |
| MPC | Stable Majority-PC; G-squared; `alpha=0.01`; majority collider rule `uc_rule=5`; priority 2; no conditioning-depth limit. |
| ABA-PC | Receives MPC's CI trace; releases facts from lowest to highest strength until coherent; incremental Bayes-ball encoding. |
| OptABA-PC | Same facts, weights, and graph encoding as ABA-PC; MUS-reified weak constraints; clingo `bb`, `optN`, four threads; minimum total release weight. |
| FGS | Tetrad/py-causal FGS; SEM-BIC score; maximum degree `-1`; faithfulness assumed; other Tetrad defaults. |
| ASPCR-DAG | Native Bayesian CI test; log weights; prior independence 0.4; Bayesian alpha 20; exhaustive `n-2` schedule; acyclic causally sufficient encoding; clingo `crafty`; 25,000-second solver limit. |

ABA repair runs use a 300-second solver limit and a separate 120-second
compatible-set evaluation limit. Time limits are computational controls, not
tuned model parameters.

## 3. ABA-PC and OptABA-PC

Fixed bnlearn networks:

```bash
for dataset in cancer earthquake survey asia; do
  python scripts/wc_opt_strategy_sweep.py \
    --source bnlearn --bnlearn-dataset "$dataset" --bn-data-path datasets \
    --bn-standardise --sample-size 5000 --seed 2026 --reps 50 \
    --alpha 0.01 --pc-indep-test gsq --strategies bb --objectives lex \
    --encodings inc --opt-modes optN --reifications mus \
    --timeout-sec 300 --graph-eval-timeout 120 --no-condset-weight \
    --threads 4 --skip-base-baseline \
    --out-dir results/reproduced/aba_fixed/"$dataset"
done
```

Primary sparse synthetic networks:

```bash
for specification in "ER 5" "ER 8" "SF 5" "SF 8"; do
  set -- $specification
  graph_type=$1
  nodes=$2
  python scripts/wc_opt_strategy_sweep.py \
    --n-nodes "$nodes" --graph-type "$graph_type" --edge-per-node 1 \
    --sample-size 5000 --seed 2026 --reps 50 --alpha 0.01 \
    --pc-indep-test gsq --strategies bb --objectives lex --encodings inc \
    --opt-modes optN --reifications mus --timeout-sec 300 \
    --graph-eval-timeout 120 --no-condset-weight --threads 4 \
    --skip-base-baseline \
    --out-dir results/reproduced/aba_sparse/"${graph_type,,}${nodes}"
done
```

Repeat the synthetic commands with `--edge-per-node 2` for the density
sensitivity runs.

## 4. MPC and FGS

The following commands reproduce the saved result versions. `--hard-exit` is
used after FGS to ensure that the Tetrad JVM does not keep the process alive.

```bash
python scripts/run_matched_baseline_experiments.py \
  --version paper_bnlearn_alpha001_nowrong_noweight_50rep_mpc_fgs \
  --methods mpc fgs --datasets cancer earthquake survey asia \
  --results-dir results/reproduced --n-runs 50 --seed-start 2026 \
  --sample-size 5000 --test-alpha 0.01 --test-name gsq \
  --edge-per-node 2 --hard-exit

python scripts/run_matched_baseline_experiments.py \
  --version paper_er_sf_sparse_alpha001_noweight_50rep_mpc_fgs \
  --methods mpc fgs --datasets er5 er8 sf5 sf8 \
  --results-dir results/reproduced --n-runs 50 --seed-start 2026 \
  --sample-size 5000 --test-alpha 0.01 --test-name gsq \
  --edge-per-node 1 --hard-exit

python scripts/run_matched_baseline_experiments.py \
  --version paper_er_sf_alpha001_nowrong_noweight_50rep_mpc \
  --methods mpc --datasets er5 er8 sf5 sf8 \
  --results-dir results/reproduced --n-runs 50 --seed-start 2026 \
  --sample-size 5000 --test-alpha 0.01 --test-name gsq \
  --edge-per-node 2

python scripts/run_matched_baseline_experiments.py \
  --version paper_er_sf_alpha001_nowrong_noweight_50rep_fgs \
  --methods fgs --datasets er5 er8 sf5 sf8 \
  --results-dir results/reproduced --n-runs 50 --seed-start 2026 \
  --sample-size 5000 --test-alpha 0.01 --test-name gsq \
  --edge-per-node 2 --hard-exit
```

Add `--resume` to any interrupted matched-baseline command. Resume uses
successful seed identities, not row counts.

## 5. ASPCR-DAG

ASPCR-DAG depends on the R/ASP package distributed with Hyttinen, Eberhardt,
and Järvisalo (UAI 2014). Obtain the authors' “Constraint-based Causal Discovery
with ASP” code package from the software link on Antti Hyttinen's publication
page and place it outside this repository. The package distributed by the
authors does not state a redistribution licence, so it is not vendored here.

The reported environment used R 4.1.2 with pcalg 2.7-11, graph 1.72.0,
Rgraphviz 2.38.0, deal 1.2-42, combinat 0.0-8, hash 2.2.6.3, and stringr
1.5.1. Set portable locations explicitly:

```bash
export ASPCR_ROOT=/path/to/aspcr-hyttinen2014uai
export ASPCR_R_DIR="$ASPCR_ROOT/R"
export RSCRIPT=/path/to/R-4.1.2/bin/Rscript
export CLINGO_BIN_DIR="$(dirname "$(command -v clingo)")"

python scripts/run_matched_baseline_experiments.py \
  --version paper_aspcr_dag_alpha001_n5000_50rep \
  --methods aspcr_log_dag \
  --datasets cancer earthquake survey er5 sf5 \
  --results-dir results/reproduced --n-runs 50 --seed-start 2026 \
  --sample-size 5000 --test-alpha 0.001 \
  --aspcr-r-dir "$ASPCR_R_DIR" --rscript "$RSCRIPT" \
  --clingo-bin-dir "$CLINGO_BIN_DIR" --fail-fast
```

`--test-alpha 0.001` is retained in the historical ASPCR run metadata but does
not configure the ASPCR tests: ASPCR uses the native Bayesian settings listed
above. It is not presented as sharing the G-squared threshold.

Validate every saved trace and graph:

```bash
python scripts/validate_matched_aspcr_results.py \
  --results-dir results/reproduced \
  --version paper_aspcr_dag_alpha001_n5000_50rep \
  --datasets cancer earthquake survey er5 sf5 \
  --n-runs 50 --seed-start 2026 --sample-size 5000 --test-alpha 0.001
```

Five-node traces must contain exactly 80 constraints; Survey must contain 240.
Every accepted DAG must be acyclic, have zero bidirected edges, and have
row-level fact counts consistent with its aggregate diagnostics and objective.

## 6. Contestability and correspondence audits

The reported OptABA-PC audit uses Cancer, Earthquake, Survey, and the dense
five-node ER/SF sensitivity instances. Its exact scope is recorded in
`results/paper_aaai2027/CONTESTABILITY_SCOPE.md`.

```bash
python scripts/run_contestability_experiments.py
python scripts/build_contestability_table.py \
  --output-dir results/paper_aaai2027/recomputed/contestability_optabapc
```

The MPC enforcement and ASPCR-DAG sensitivity commands can be pointed at the
frozen release inputs using their documented CLI path arguments:

```bash
python scripts/audit_mpc_test_enforcement.py --help
python scripts/run_aspcr_contestability.py --help
python scripts/build_baseline_contestability_tables.py --help
```

The frozen evidence contains 742/742 proved OptABA-PC hard-retention solves,
400 MPC full-trace audits, and 3,133/3,133 proved ASPCR-DAG sensitivity solves.
The latter are sensitivity probes, not an ASPCR redress guarantee.

## 7. Rebuild and validate the tables

```bash
python scripts/build_final_experiment_tables.py \
  --results-dir results/paper_aaai2027/frozen/results \
  --mcs-results-dir results/paper_aaai2027/frozen/results/final_mcs_experiments_er_sf_alpha001_nowrong_noweight_50rep_chunked \
  --out-dir results/paper_aaai2027/recomputed/tables

python scripts/validate_release_artifacts.py
```

No missing outcome is imputed. The frozen completion record contains these
exceptions:

- Asia OptABA-PC: seed 2047 unavailable (49/50).
- Survey MPC compatible-graph evaluation: 38/50.
- ER(8) MPC compatible-graph evaluation: seed 2030 unavailable (49/50).
- SF(5) FGS: seeds 2037 and 2057 unavailable (48/50).

All other reported method/dataset rows have 50 saved seeds, subject to a metric
being inapplicable (for example, CI-fact F1 for FGS).

## 8. Frozen artefact integrity

`scripts/prepare_release_artifacts.py` constructs the curated release from an
allowlist of final result versions. It removes machine-local paths from copied
metadata, omits exploratory runs, and writes SHA-256 inventories. It refuses to
overwrite an existing release unless `--force` is supplied.

```bash
python scripts/prepare_release_artifacts.py --force
python scripts/validate_release_artifacts.py
```

The validator checks the protocol, hashes, expected result versions, absence of
machine-local paths, and the declared completion exceptions. It does not alter
the frozen evidence.
