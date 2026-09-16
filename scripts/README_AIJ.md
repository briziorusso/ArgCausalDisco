# AIJ experiments and LaTeX tables

The `AIJ` branch starts from `dev` commit `0cfa9c269055ebe4064ab30f9c679843f9c70810`.
This directory contains the entry points for reevaluating existing graphs,
building the paper tables, and running the missing baselines. Run commands
from the repository root in the existing experiment environment.

## Files and outputs

| File | Purpose |
| --- | --- |
| `reevaluate_aij_graphs.py` | Reevaluate the 420 archived graphs without rerunning learners. |
| `build_aij_tables.py` | Generate the 14 retained LaTeX tables, paired tests and provenance manifest. |
| `run_aij_baselines.py` | Preflight, launch/resume, validate and evaluate ASPCR-DAG or FGS-BDeu. |
| `run_matched_baseline_experiments.py` | Per-seed experiment runner and raw artifact writer. |
| `validate_matched_aspcr_results.py` | Check DAG constraints, native CI traces, objectives and matched inputs. |
| `run_fgs_once.py` | Run one FGES search in an isolated JVM and preserve edge endpoints and node order. |
| `../utils/aij_graph_metrics.py` | Explicit graph decoding, complete CPDAG conversion, edge metrics and exact SID bounds. |

The ASPCR runner and R wrapper derive from the corrected MCS implementation at
[`261451bb20701f93773852376514353af1d11b24`](https://github.com/briziorusso/ArgCausalDisco/tree/261451bb20701f93773852376514353af1d11b24).
The AIJ adaptation adds explicit seed identities, input reuse, compatible resume
checks and the corrected graph evaluation. The external ASPCR distribution is
not bundled.

## Server: two resumable commands

After the branch has been pushed, update the server checkout:

```bash
git fetch origin
git switch AIJ                         # first use: git switch --track origin/AIJ
git pull --ff-only origin AIJ
python -m pip install -r requirements-aij-evaluation.txt 'pandas<3'
```

Use the working FGS/ASPCR environment from the earlier experiments. FGS needs
`pycausal`, its Java bridge and Tetrad; ASPCR needs R, the authors' R dependencies
and clingo. The pandas bound avoids a known pgmpy sampling incompatibility with
pandas 3. The preflight checks actual dependencies before the experiment starts.

Inspect the pending runs without launching either solver:

```bash
python scripts/run_aij_baselines.py --method aspcr --plan
python scripts/run_aij_baselines.py --method fgs --plan
```

ASPCR-DAG uses Cancer, Earthquake and Survey (30 runs). Set the external package
location, then run this command inside tmux:

```bash
export ASPCR_R_DIR=/path/to/aspcr-hyttinen2014uai/R
# Set RSCRIPT or CLINGO_BIN_DIR only if Rscript/clingo are not on PATH.
mkdir -p results/logs
set -o pipefail
python -u scripts/run_aij_baselines.py --method aspcr \
  2>&1 | tee -a results/logs/aij_aspcr.log
```

FGS uses discrete **BDeu**, with equivalent sample size 15 and structure prior 1,
matching the MCS configuration. It retains undirected CPDAG edges and resolves
node names in input-column order. Observed categorical values are encoded as
integer categories; no continuous discretisation is introduced. The default is
all six AIJ datasets (60 runs):

```bash
mkdir -p results/logs
set -o pipefail
python -u scripts/run_aij_baselines.py --method fgs \
  2>&1 | tee -a results/logs/aij_fgs_bdeu.log
```

The jobs can run in separate tmux sessions. Repeat the same command to resume;
successful seeds are skipped and failed seeds retried. Do not run two copies
of the same method/version concurrently. An explicit `--datasets` selection is
supported, but keep it unchanged when resuming that result version. Use
`--preflight-only` to check dependencies and inputs without the full experiment.

Both jobs use 5000 observations and seeds
`7816 3578 2656 2688 2494 183 7977 3199 316 8266`. The first four datasets reuse
the tracked `datasets/data_npy/` inputs. Sachs and Child are regenerated with
the original sampler and seed; their true graph/node order is checked against
the archived run. Because their original samples are not archived, identical
seed values alone cannot certify identical observations across changed sampling
dependencies. Every invocation records sample and code hashes.

ASPCR retains native Bayesian CI tests (`p=0.4`, `alpha=20`) and log weights.
The G-squared threshold in the comparison metadata does not replace those tests.
Its inherited solver limit is 25000 seconds per run, excluding preprocessing.

Output versions are `aij_aspcr_dag_native_n5000_matched10_v2` and
`aij_fgs_bdeu_n5000_matched10_v2`; they do not overwrite older results:

- `results/progress/<version>/`: per-seed status and edge scores;
- `results/estimated_graphs/<version>/`: original estimates, true graphs and metadata;
- `results/aspcr_matched/<version>/`: ASPCR inputs and solver/CI traces;
- `results/matched10/`: invocation commands and input/code hashes;
- `results/aij_evaluation/reruns/<version>/`: validated DAG/CPDAG scores, exact SID
  bounds, attaining DAGs and a manifest, written after the cohort passes validation.

A nonzero exit means the cohort or its validation/evaluation is incomplete.
The launcher never silently substitutes a partial cohort into the paper tables.
The current table snapshot retains the historical SEM-BIC FGS comparison;
the new BDeu and matched ASPCR results remain separately identifiable until
the completed cohorts are incorporated into the comparison design.

## Rebuild the retained tables

```bash
python scripts/reevaluate_aij_graphs.py
python scripts/build_aij_tables.py --paper-dir ../CausalABA-AIJ
python -m unittest discover -s tests -v
```

Adjust `--paper-dir` to the TeX checkout location. The build creates
`results/tables/aij/` and copies only the LaTeX inputs, `preview.tex`,
`table_build_manifest.json` and `SOURCES.md` to the paper's `tables/aij/`.
Compile the preview with `pdflatex preview.tex` in that directory. The main
manuscript is not rewritten. Commit the code before the final table build so
the paper manifest links to the exact committed implementation.

| Review tables | Files | Source |
| --- | --- | --- |
| 1–2 | `table_aij_cpdag_core_*.tex` | Archived matched-ten raw graphs, reevaluated. |
| 3–4 | `table_aij_cpdag_precision_recall_*.tex` | Same graphs; edge, skeleton and arrowhead scores. |
| 5–6 | `table_aij_dag_*.tex` | Native DAGs or seeded consistent extensions. |
| 7–8 | `table_aij_graph_size_*.tex` | Evaluated estimates and reference graphs. |
| 9 | `table_aij_main_delta.tex` | Within-seed bb-nor minus bb effects. |
| 10–11 | `table_aij_runtime_*.tex` | Matched-ten timings plus explicitly labelled legacy cohorts. |
| 12 | `table_aij_variants.tex` | Encoding variants; cohort/sample sizes shown. |
| 13 | `table_aij_semantics.tex` | Separate 50-run summaries, including saved notebook output for CO-max. |
| 14 | `table_aij_alpha_methods.tex` | Archived within-seed threshold comparison. |

The initial review's tables 15–18 are retired: fact changes, two fact-quality
tables, and the separate 50-seed ASPCR cohort. AP/NDCG source records for the
ranking figure have not been recovered; no replacement values are invented.
An optional `--ranking-csv` can generate that additional table from original
per-run `dataset,method,seed,AP,NDCG` data, once available.

`table_build_manifest.json` records the code revision, input/output SHA-256
hashes, seeds, settings and limitations. `SOURCES.md` provides clickable links
to that revision. CSVs, NPZ caches, preview PDFs and audit figures are generated
locally and excluded from code commits. Original tracked experiments are kept.

## Evaluation conventions

Canonical adjacency matrices store row-to-column directed edges as 1 and
undirected edges symmetrically. Learner-specific endpoints are decoded before
binarisation. Complete DAG-to-CPDAG conversion preserves every compelled arrow.

Overall P/R/F1 compare edge states: undirected, forward-directed and
reverse-directed. Skeleton scores ignore direction; arrowhead scores count
actual arrows. F1 is `2 TP / (|predicted| + |reference|)`. Precision is undefined
for an empty prediction, while F1 against a nonempty reference is zero.
SHD counts one edit per differing unordered pair state. CPDAG evaluation uses
the true CPDAG; DAG evaluation uses the true DAG.

SID is evaluated against the true DAG. Exact lower/upper bounds optimise over
compatible DAGs using dynamic programming over simplicial-sink elimination in
each chain component. Attaining DAGs are saved and independently scored using
`gadjid.sid`. Resource exhaustion produces unavailable bounds, never loose
values labelled exact. Both SID and SHD are normalised by the true DAG edge
count, so their normalised values can exceed one.

Five Survey MPC outputs (seeds 7816, 2688, 183, 7977 and 316) have no consistent
DAG extension. Their returned edges remain evaluable, but their DAG and
equivalence-class SID scores are unavailable. Reported means and paired tests
use defined measurements; counts and excluded seeds are retained. This output
validity finding alone does not establish inconsistency of all input CI tests.
Sachs has no compelled reference arrowheads, so its AH comparisons are `n/a`.
`--` denotes an unavailable value; neither symbol replaces an observed zero.

Primary tables bold the best mean. Dagger/double dagger indicate significantly
worse/better performance than ABAPC (bb-nor), using paired two-sided t-tests and
Holm correction across the five comparisons within each dataset/metric at 0.05.
Absence of a marker does not assert equivalence. Historical cohorts are excluded
from these tests. The historical FGS DOT export lacks endpoint/node-label
provenance; its native CPDAG cannot be recovered from those files.

Tests check endpoint decoding, complete conversion, permutation-invariant F1,
invalid PDAG rejection, paired statistics and resume identities. Exact SID
extrema match exhaustive three-node classes and all four-node classes against
three reference DAGs. The DAG backend also reproduced all 420 archived R DAG
SID values during the local audit. External ASPCR/Java execution requires the
server environment; unit tests do not replace its preflight and output checks.
