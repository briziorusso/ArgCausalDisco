# AIJ matched reruns on the Linux server

Use the existing **`abaenv`** conda environment. This is a run plan, not a record
of completed experiments. The current ASPCR-DAG and FGS commands below exist;
the final section specifies commands for the common-input runner **still to be
implemented**. No experiments were launched when preparing this document.

## Comparison to run

Use 5000 observations and the same ten seeds for every supported method:
`7816 3578 2656 2688 2494 183 7977 3199 316 8266`.

| Dataset | Nodes | True DAG edges | Saved samples currently available |
| --- | ---: | ---: | ---: |
| Cancer | 5 | 4 | 10/10 |
| Earthquake | 5 | 4 | 10/10 |
| Survey | 6 | 6 | 10/10 |
| Asia | 8 | 8 | 10/10 |
| Sachs | 11 | 17 | 0/10 |
| Child | 20 | 25 | 0/10 |

Freeze one input collection before running the full comparison. Reuse the 40
saved samples in `datasets/data_npy/`; recover the original Sachs/Child arrays
from the server if available, otherwise generate those 20 samples once in the
pinned server environment. The latter are a new cohort, not certified copies of
the historical samples. Every learner must read the resulting files; a shared
seed alone is insufficient. Record sample/truth hashes, column order, category
levels, transformations, sample and learner seeds, generator revision and
dependency versions. The loader must fail on a missing or changed input.

| Table name | Implementation to use | Configuration / work needed |
| --- | --- | --- |
| Random | Existing AIJ random baseline in `cd_algorithms/models.py` | Preserve its generator and settings; do not substitute the matched-edge-count `random_edge` baseline. |
| FGS | `scripts/run_fgs_once.py` | Discrete BDeu, equivalent sample size 15, structure prior 1; preserve native endpoints and column order. |
| NOTEARS-MLP | Existing `nt` branch of `run_method` | Preserve architecture, optimisation, threshold and device policy; record all settings. |
| MPC | Existing `mpc` branch of `run_method` | G-squared, alpha 0.05, existing collider/orientation policy. |
| ABAPC (orig) | `abapc.py` with baseline `causalaba.CausalABA` | Current fact generation; path-based solver; `S_weight=false`, `pre_grounding=false`, `disable_reground=false`. |
| ABAPC (nor) | Same baseline solver | `S_weight=false`, `pre_grounding=true`, `disable_reground=true`. |
| ABAPC (bb) | Incremental `causalaba_increm.CausalABA` | Resolve all defaults from the archived bb launch and pin them explicitly. |
| ABAPC (bb-nor) | Same incremental solver | `pre_grounding=false`, `disable_reground=true`; preserve the saved search configuration. |
| ASPforABA | `semantics/src/abapc.py` and external `aspforaba` | Stable extensions; adapt the existing `get_arrow_sets` / `get_best_model` pipeline to saved inputs and raw output records. |
| ASPCR-DAG | `scripts/run_aij_baselines.py --method aspcr` | Native Bayesian CI (`p=0.4`, `alpha=20`), log weights and DAG-only encoding. Cancer/Earthquake/Survey first. |

“Orig” here means the original path-based solver with the current shared
front end, as in `scripts/runners/run_bnlearn_abapc_orig_matched10.py`; it is
not a claim to reproduce an untouched historical release. Keep the historical
50-run summaries separate until replaced by explicitly labelled new runs.

All methods receive the same observations. FGS may reversibly recode categories;
ASPCR retains its own CI procedure. Within the ABAPC solver comparison, save and
verify identical CI queries, results and strength inputs before comparing
encodings. ASPforABA currently removes the weakest facts until extensions exist,
enumerates up to 50,000 extensions and ranks their graphs using the original
score. Preserve and record that policy, deterministic tie handling and any
enumeration truncation; do not silently replace it with another ABAPC solver.

## Implementation order before the full launch

1. **Freeze and validate inputs.** Add an idempotent input exporter and immutable
   manifest. Check every true graph against the archived MPC truth and retain
   node order. Existing `run_matched_baseline_experiments._load_data` can read
   saved arrays but silently regenerates missing ones; strict manifest loading
   must replace that fallback for this cohort.
2. **Add a common runner and method registry.** Extend the saved-input path to
   MPC, NOTEARS and all four ABAPC configurations; the current baseline launcher
   only exposes ASPCR and FGS. Add a subprocess adapter for ASPforABA, pin its
   external revision and resolve its package imports. Its historical use of
   `nx.algorithms.d_separated` also needs compatibility with the environment's
   NetworkX version. Verify it on small known graphs before launching.
3. **Save comparable artifacts and enforce limits.** Save the native graph,
   decoded graph, node order, input hash, settings, elapsed phases, peak memory,
   solver status and code/environment hashes for every attempt. Cache CI traces
   for comparison, but include their computation in end-to-end method timing.
   Reuse the corrected evaluator, including the development branch's handling
   of invalid PDAGs. Resume only when input, settings and implementation hashes
   match; preserve unsuccessful attempts instead of overwriting them.
4. **Pilot and lock resources.** Run seed 7816 on Cancer, Earthquake and Survey
   for all ten methods: 30 jobs. Use a fixed CPU allocation, CPU-only NOTEARS for
   the primary timing comparison, and one timed job at a time. Set the same
   end-to-end time/memory budget before the full run; the current ASPCR limit of
   25,000 seconds covers its solver only and is not such a shared limit.
   The commands below propose four threads, 25,000 seconds and 32 GiB per job;
   these are pilot settings to confirm against the server's available resources.
5. **Complete the small datasets, then scale.** Cancer/Earthquake/Survey give
   300 jobs including the pilot, across ten methods. Pilot the remaining nine
   methods on Asia/Sachs/Child, then complete their ten seeds (up to 270 more
   jobs). ASPCR's current six-node guard remains in place; a larger-dataset
   experiment requires a separately validated change. Unsupported runs and
   timeouts remain explicit, not invented numerical scores.
6. **Evaluate and rebuild from an explicit cohort manifest.** Verify identical
   inputs across methods and preserve invalid-output diagnostics. Evaluate DAG
   and CPDAG scores with corrected F1 and exact NSID_B/NSID_W; normalise distances
   by true DAG edges. Retain defined edge scores when no consistent extension
   exists. Use `n/a` for structurally inapplicable metrics and `--` for unavailable
   measurements. Retain paired two-sided t-tests and BH, with the method family
   declared before inspecting outcomes (45 pairs for ten methods, 36 for nine).
   Missing tests retain their place in that family. Extend the table builder's
   current six-method registry and source selection before importing new runs.

The alpha and semantics studies are separate ablations, not additional methods
to pool into the primary test family. If rerun, use the same frozen observations,
record the changed alpha/semantics explicitly, and keep their paired comparisons
separate. Preserve the current historical panels until those runs are available.

## Linux setup and commands available now

First publish the local AIJ commits from the development checkout when ready:

```bash
git push origin AIJ
```

On the server, replace the repository and external ASPCR paths below. Start in
the existing checkout; these commands do not discard local changes.

```bash
cd /path/to/ArgCausalDisco
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate abaenv
git fetch origin
git switch AIJ  # first checkout, if needed: git switch --track origin/AIJ
git pull --ff-only origin AIJ
python -c 'import sys; print(sys.executable)'
python -m pip install -r requirements-aij-evaluation.txt 'pandas<3'
python -m pip check

export ASPCR_R_DIR=/path/to/aspcr-hyttinen2014uai/R
test -f "$ASPCR_R_DIR/load.R"
# Only if needed: export RSCRIPT=/path/to/Rscript
# Only if needed: export CLINGO_BIN_DIR=/path/to/directory-containing-clingo

mkdir -p results/aij_commondata_n5000_matched10_v3/provenance results/logs
git rev-parse HEAD > results/aij_commondata_n5000_matched10_v3/provenance/code_commit.txt
conda list --explicit > results/aij_commondata_n5000_matched10_v3/provenance/conda-explicit.txt
python -m pip freeze > results/aij_commondata_n5000_matched10_v3/provenance/pip-freeze.txt
lscpu > results/aij_commondata_n5000_matched10_v3/provenance/cpu.txt
free -h > results/aij_commondata_n5000_matched10_v3/provenance/memory.txt
python -m unittest discover -s tests -v
```

FGS still requires the working `pycausal`/Java installation; ASPCR requires the
authors' R dependencies and clingo. The existing preflight actually loads/runs
these dependencies. It does not certify the yet-to-be-added ASPforABA adapter.
Keep `abaenv` fixed after capturing the environment; repeat the capture if its
dependencies must change before launching.

The following commands are usable **now** for the three datasets whose samples
are already archived. They launch 30 runs per method, not the one-seed pilot.
Use a separate result directory so these baseline runs cannot accidentally be
treated as a completed all-method cohort. Reusing them later requires matching
the frozen input/configuration hashes; runtime comparisons additionally require
the common resource and timing protocol.

```bash
export AIJ_BASELINE_RESULTS="$PWD/results/aij_baselines_small_v3"
python scripts/run_aij_baselines.py --method aspcr \
  --datasets cancer earthquake survey --results-dir "$AIJ_BASELINE_RESULTS" --plan
python scripts/run_aij_baselines.py --method fgs \
  --datasets cancer earthquake survey --results-dir "$AIJ_BASELINE_RESULTS" --plan
python scripts/run_aij_baselines.py --method aspcr \
  --datasets cancer earthquake survey --results-dir "$AIJ_BASELINE_RESULTS" --preflight-only
python scripts/run_aij_baselines.py --method fgs \
  --datasets cancer earthquake survey --results-dir "$AIJ_BASELINE_RESULTS" --preflight-only
```

Open a tmux session, activate the environment inside it and run the jobs
sequentially. Detach with Ctrl-b, then d. Run the next method after the first
command finishes; if it fails, inspect its log before continuing.

```bash
tmux new-session -s aij-baselines
# Inside tmux:
cd /path/to/ArgCausalDisco
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate abaenv
export ASPCR_R_DIR=/path/to/aspcr-hyttinen2014uai/R
export AIJ_BASELINE_RESULTS="$PWD/results/aij_baselines_small_v3"
mkdir -p results/logs
set -o pipefail
python -u scripts/run_aij_baselines.py --method aspcr \
  --datasets cancer earthquake survey --results-dir "$AIJ_BASELINE_RESULTS" \
  2>&1 | tee -a results/logs/aij_aspcr_small_v3.log
python -u scripts/run_aij_baselines.py --method fgs \
  --datasets cancer earthquake survey --results-dir "$AIJ_BASELINE_RESULTS" \
  2>&1 | tee -a results/logs/aij_fgs_bdeu_small_v3.log
```

From another shell, inspect progress or reattach:

```bash
tmux list-sessions
tmux attach-session -t aij-baselines
# Or just inspect a log:
tail -n 40 results/logs/aij_aspcr_small_v3.log
```

Repeat the identical launch command to resume; do not run duplicate instances.
The launcher validates ASPCR's traces and evaluates completed cohorts
automatically. `--plan` can be repeated to list outstanding seeds. A nonzero exit
means execution, validation or evaluation is incomplete. New baseline outputs
are not yet automatically included by `build_aij_tables.py`.

To inspect the existing original-ABAPC command without launching it:

```bash
python scripts/runners/run_bnlearn_abapc_orig_matched10.py \
  --python-bin "$(command -v python)" --names cancer earthquake survey \
  --sample-size 5000 --n-runs 10 --test-name gsq --test-alpha 0.05 \
  --threads 4 --no-S-weight --pre-grounding false --disable-reground false \
  --version aij_abapc_orig_n5000_matched10_v3 --print-only
```

The explicit interpreter avoids that wrapper's historical `aba-env` path.
Do not remove `--print-only` for the new common-data study until its loader uses
the manifest. The historical ASPforABA experiment script is likewise not a
substitute for the missing adapter: it generates its own inputs, uses its own
defaults and writes aggregate results.

## Full comparison: command interface to implement

**These commands are the implementation specification, not executable commands
on the current branch.** The three new entry points and `--cohort-manifest` table
option do not exist yet. Implement steps 1–6 above, test their help/preflight and
resume behavior, then use this sequence in a new `aij-all-methods` tmux session
with `conda activate abaenv`. Their preflight must check all adapters and report
resolved settings, external revisions and supported datasets before any jobs.

```bash
export AIJ_COHORT="$PWD/results/aij_commondata_n5000_matched10_v3"
export AIJ_INPUTS="$AIJ_COHORT/inputs/manifest.json"
export AIJ_THREADS=4
export AIJ_TIMEOUT=25000
export AIJ_MEMORY_GIB=32
export ASPFORABA_DIR=/path/to/aspforaba
# ASPCR_R_DIR must also be set as above.

# Recover original server samples before this command, if available.
python scripts/freeze_aij_inputs.py --output "$AIJ_INPUTS" \
  --datasets cancer earthquake survey asia sachs child --sample-size 5000 \
  --seeds 7816 3578 2656 2688 2494 183 7977 3199 316 8266

# Check files/hashes, adapters and external dependencies before launching.
python scripts/run_aij_matched.py --inputs "$AIJ_INPUTS" --results-dir "$AIJ_COHORT" \
  --methods all --datasets cancer earthquake survey --preflight-only

AIJ_COMMON=(--inputs "$AIJ_INPUTS" --results-dir "$AIJ_COHORT"
  --threads "$AIJ_THREADS" --timeout "$AIJ_TIMEOUT" --memory-gib "$AIJ_MEMORY_GIB"
  --device cpu --jobs 1 --resume)
set -o pipefail
python -u scripts/run_aij_matched.py "${AIJ_COMMON[@]}" --methods all \
  --datasets cancer earthquake survey --seeds 7816 \
  2>&1 | tee -a "$AIJ_COHORT/pilot.log"

# After checking pilot validity and locking the common resource budget:
python -u scripts/run_aij_matched.py "${AIJ_COMMON[@]}" --methods all \
  --datasets cancer earthquake survey \
  2>&1 | tee -a "$AIJ_COHORT/small.log"

AIJ_LARGE_METHODS=(random fgs_bdeu nt mpc abapc_orig abapc_nor abapc_bb abapc_bb_nor aspforaba)
python -u scripts/run_aij_matched.py "${AIJ_COMMON[@]}" \
  --methods "${AIJ_LARGE_METHODS[@]}" --datasets asia sachs child --seeds 7816 \
  2>&1 | tee -a "$AIJ_COHORT/large_pilot.log"
# After checking the large-dataset pilot:
python -u scripts/run_aij_matched.py "${AIJ_COMMON[@]}" \
  --methods "${AIJ_LARGE_METHODS[@]}" --datasets asia sachs child \
  2>&1 | tee -a "$AIJ_COHORT/large.log"

python scripts/evaluate_aij_matched.py --cohort-manifest "$AIJ_COHORT/cohort.json"
python scripts/build_aij_tables.py --cohort-manifest "$AIJ_COHORT/cohort.json" \
  --out-dir "$AIJ_COHORT/tables" --paper-dir /path/to/CausalABA-AIJ
```

Each command must stop on input/configuration mismatches; rerunning a pilot as
part of the full cohort skips only identical successful jobs. Changed resource
settings create a new timing cohort rather than silently mixing pilot timings.
The evaluator must report expected/completed/failed/unsupported jobs per method
and dataset, pair counts and excluded seeds before building the tables.

Until that interface is implemented, the following **existing** commands only
rebuild the current archived-result tables, including their provenance:

```bash
python scripts/reevaluate_aij_graphs.py
python scripts/build_aij_tables.py --paper-dir /path/to/CausalABA-AIJ
cd /path/to/CausalABA-AIJ/tables/aij
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
pdflatex -interaction=nonstopmode -halt-on-error preview.tex
```

Commit code/configuration and the paper's TeX/provenance snapshots separately.
Keep large inputs, raw graphs, logs, environment exports and PDFs as experiment
artifacts, outside commits. Preserve historical results and input hashes.
