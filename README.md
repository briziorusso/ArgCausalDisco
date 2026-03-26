# Optimal Correction Sets for Argumentative Causal Discovery

This `MCS` branch of ArgCausalDisco contributes an optimal-MCS version of Causal ABA: instead of ABA-PC's heuristic "release low-ranked facts until the encoding becomes satisfiable", this branch treats fact release as a repair problem and computes an optimal correction set (optimal MCS) with weak constraints.
In the manuscript this solver-backed variant is the OptABA-PC method.

## Where the MCS logic lives

- `scripts/wc_opt_strategy_sweep.py` is the experiment driver used to generate the archived runs in [`results/final_mcs_experiments`](results/final_mcs_experiments).
- `causalaba_mus.py` implements the proposed optimum-MCS path through `CausalABA_WC(...)`. It builds the guarded `mus(i)` program and optimizes the weight of released CI facts with clingo weak constraints.
- `causalaba.py` and `causalaba_increm.py` are the ABA-PC baselines compared against the optimum-MCS solver in the sweep.
- The archived `final_mcs_experiments` artifacts are produced by `scripts/wc_opt_strategy_sweep.py` invoking `causalaba_mus.CausalABA_WC(...)`.

## Environment

The code was tested with Python 3.10.
Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

You need `clingo` on `PATH`.
The published sweeps in `results/final_mcs_experiments` do not require `wasp`; that solver is only needed for separate MUS/MCS enumeration workflows.

## How `results/final_mcs_experiments` was created

Running `scripts/wc_opt_strategy_sweep.py` creates a timestamped directory under `results/` named `wc_sweep_*_<timestamp>/` containing:

- `0.Config` with the resolved configuration
- `summary.json` and `metric_ranks.json`
- per-repetition folders `rep*/` with the generated facts and emitted `.lp` programs

The directories now collected under [`results/final_mcs_experiments`](results/final_mcs_experiments) were created that way and then copied into this archive folder.

### Exact archived commands

The following commands reproduce the archived sweep setup.
Defaults not shown on the bnlearn commands resolve to `--strategies bb --objectives lex --encodings inc,base --opt-modes optN --reifications mus`.

```bash
python scripts/wc_opt_strategy_sweep.py --reps 10 --seed 2026 --source bnlearn --bnlearn-dataset cancer --bn-data-path datasets --timeout-sec 120 --graph-eval-timeout 120 --no-condset-weight
python scripts/wc_opt_strategy_sweep.py --reps 10 --seed 2026 --source bnlearn --bnlearn-dataset survey --bn-data-path datasets --timeout-sec 120 --graph-eval-timeout 120 --no-condset-weight
python scripts/wc_opt_strategy_sweep.py --reps 10 --seed 2026 --source bnlearn --bnlearn-dataset asia --bn-data-path datasets --timeout-sec 120 --graph-eval-timeout 120 --no-condset-weight
python scripts/wc_opt_strategy_sweep.py --reps 10 --seed 2026 --source bnlearn --bnlearn-dataset earthquake --bn-data-path datasets --timeout-sec 120 --graph-eval-timeout 120 --no-condset-weight
python scripts/wc_opt_strategy_sweep.py --n-nodes 5 --seed 2026 --reps 10 --strategies bb --objectives lex --encodings inc,base --opt-modes optN --timeout-sec 120 --pct-wrong-facts 0.2 --notes "musonly optonly lexonly solve120 eval120 gsq noweight std worst" --graph-eval-timeout 120 --reif mus --no-condset-weight
python scripts/wc_opt_strategy_sweep.py --n-nodes 8 --seed 2026 --reps 10 --strategies bb --objectives lex --encodings inc --opt-modes optN --timeout-sec 300 --pct-wrong-facts 0.2 --notes "musonly optonly lexonly solve120 eval 300 gsq noweight std worst" --graph-eval-timeout 120 --reif mus --no-condset-weight
```

The core published optimum-MCS setting is therefore:

- reification: `mus`
- objective: `lex`
- clingo opt strategy: `bb`
- clingo opt mode: `optN`
- conditioning-set weighting disabled via `--no-condset-weight`

## Recreate the paper table

Once the run directories are present as children of [`results/final_mcs_experiments`](results/final_mcs_experiments), regenerate the LaTeX table with:

```bash
python results/final_mcs_experiments/recreate_main_results_table.py
```

This writes [`results/final_mcs_experiments/main_results_table.tex`](results/final_mcs_experiments/main_results_table.tex).
The table script expects the five directories used in the manuscript table: `cancer`, `survey`, `asia`, synthetic `ER (5)`, and synthetic `ER (8)`.
The archived `earthquake` sweep is kept in the folder as an additional run, but it is not consumed by the table recreation script since no difference between methods are observed there (both perfect).
