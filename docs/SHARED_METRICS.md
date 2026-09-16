# Corrected graph metrics in development

`dev-shared-metrics` ports the shared AIJ fixes onto the active development
branch, `dev-llm-full-experiments` at `5727891d`. The publication tables, result
archives, seed lists, and ASPCR launcher remain on `AIJ`.

## Commits

| Development commit | Source / purpose |
| --- | --- |
| `931cff10` | AIJ `aeed90be`: endpoint decoding, complete CPDAGs, edge-state F1, exact SID, dependencies and regression tests. Development's unexpected-backend-error propagation is retained. |
| `8670d241` | AIJ `52858d1f`: shared scale for SID lower/upper-bound plots. |
| `6684b0a2` | AIJ `35ec637a`: preserve FGS endpoints and support discrete BDeu scoring. |
| `f7478115` | Adapt development runners, sweeps and examples; decode MPC-LLM; retain computable PDAG edge scores; update historical metric expectations. |
| `83003e34` | Label new scores `graph_v2` and reject resuming unlabelled or incompatible result versions. |

## Install on the active development branch

Run from the code repository, with the experiment environment activated:

```sh
git fetch origin
git switch dev-llm-full-experiments
git merge --ff-only origin/dev-shared-metrics
python -m pip install -r requirements-aij-evaluation.txt
python -m unittest discover -s tests -v
```

The requirements file keeps its AIJ name so both branches install the same
evaluation dependencies. These supplement an existing experiment environment;
they are not the full learner installation. SID uses `gadjid`, without R/CDT.

If development has advanced, `--ff-only` stops instead of changing its history.
Review and merge the branches normally in that case. To port onto another
development branch without including the MPC-LLM branch history, cherry-pick
the five development commits above in order, then run the same tests. That
separate branch may need conflict resolution; the tested integration target is
`dev-llm-full-experiments`.

## Start a new result version

Keep the existing experiment arguments and change `--version`, for example
from `my_run` to `my_run_graph_v2`. A small smoke run is:

```sh
python experiments.py --source bnlearn --names cancer --models random --n_runs 1 --sample_size 100 --version graph_v2_smoke --results_dir results/graph_v2_smoke
```

Add `--resume` to that command to resume the corrected version. The runner
records the protocol in its manifest, progress CSVs, archived metric records,
and a `metric_protocol_<version>.json` sidecar. WC sweeps also record it in
their JSON summaries; `experiment_llm.py` records it in its result rows and a
sidecar. Choose new output paths for those entry points too.

Old result versions remain historical records. Do not add a protocol label to
old scores or concatenate them with `graph_v2` scores. Recompute evaluation
from the saved raw estimate and true DAG when those are available; rerunning
the learner is unnecessary for metric corrections alone. FGS with a changed
score requires a new learner run.

## Metric contract

- Decode the learner's native representation before binarization. MPC and
  MPC-LLM return transposed causal-learn endpoint matrices; FGS preserves both
  directed and undirected edges.
- DAG evaluation compares directed edges with the true DAG. CPDAG evaluation
  compares edge states with the **complete** true CPDAG. Callers pass
  `evaluation_kind='cpdag'` explicitly, including fully directed CPDAGs.
- Overall F1 uses matched edge states; adjacency F1 ignores orientation and
  arrowhead F1 counts directed edges. Compute F1 as `2 TP / (|prediction| +
  |reference|)`. An empty prediction against nonempty truth scores zero;
  both empty sets give an undefined score. Undefined precision/recall are
  `NaN`. Publication choices such as displaying Sachs arrowhead scores as
  `n/a` belong in the table layer.
- SID is a raw count. CPDAG bounds are exact extrema attained by compatible
  DAGs, verified by witness DAGs. A search limit yields unavailable bounds
  with an explicit status, never approximate extrema. `--eval_timeout`
  controls the CPDAG SID search time budget; zero disables that time limit,
  while the state cap still applies.
- `utils.graph_evaluation.evaluate_estimate` retains native PDAG edge scores
  when no consistent DAG extension exists. DAG metrics and SID are unavailable
  in that case. An extendible but noncompleted PDAG, such as one with prior
  orientations, retains its edge scores and a consistent DAG representative;
  it does not receive CPDAG SID bounds. Its `cpdag_status` identifies this
  distinction; the `cpdag` return slot holds the native PDAG in these cases.
- Unexpected backend failures still raise errors. They are not treated as
  invalid learner outputs.

For saved graphs, use the shared entry point and retain its validity statuses:

```python
from utils.graph_evaluation import evaluate_estimate

evaluation = evaluate_estimate(raw_estimate, true_dag, method, seed=seed)
dag_scores = evaluation['dag_metrics']
cpdag_scores = evaluation['cpdag_metrics']
statuses = evaluation['dag_status'], evaluation['cpdag_status']
```

## Validation

The 23 discoverable tests pass, including exhaustive small-graph SID checks,
FGS adapter tests, plot scales, MPC-LLM endpoint decoding, invalid PDAGs,
clingo-model sweep aggregation, and protocol guards. The two existing
`TestMetricsDAG` cases in `tests.py` also pass with corrected expectations.
All edited entry points compile. A Cancer/Random run saved its graph artifacts
and protocol-labelled scores; resuming it retained one row per result file.
Native Java FGS execution and full experiment sweeps were not run for this port.
