# MUS/MCS Analysis for CausalABA

This document describes how to analyze contradictions in CausalABA using:

- **MUS** (Minimal Unsatisfiable Subsets): minimal groups of `ext_*` facts that still make the encoding UNSAT.
- **MCS** (Minimal Correcting Sets): minimal groups of `ext_*` facts whose removal restores SAT.

## Motivation (why MUS/MCS vs consecutive removal)

ABAPC’s “removal strategy” can restore satisfiability by removing facts until a model exists, but:

- it is **order/heuristic dependent** (e.g., remove lowest-weight facts first),
- it does not tell you which facts form the **logical core** of the contradiction,
- it does not separate **many alternative minimal explanations**.

MUS/MCS analysis instead explains contradictions in terms of minimal logical cores (MUS) and minimal fixes (MCS), which is often what you want for debugging PC outputs, prior knowledge constraints, or scoring/weighting choices.

## Repo structure (manual + automated)

- Manual reproduction of the smallest example:
  - `encodings/test_lps/mock_three_var_manual/`
    - contains a tiny hand-crafted case and `run_tests.sh`

- Automated tests + integration runs:
  - `tests_mus.py`
    - unit tests for parsing and assumption-layer generation
    - integration tests that run ABAPC removal and MUS/MCS on random-PC-generated facts

- MUS program builder + solver wrapper:
  - `causalaba_mus.py`
    - builds an *adorned* ASP program with `mus(i)` assumptions
    - runs `clingo --output=smodels | wasp --mus=mus ...`

- Profiling wrapper (ABAPC vs MUS timing):
  - `scripts/mus_abapc_profile.py`
    - runs the random-size pipeline across node sizes and prints timing summaries
    - can emit the full adorned `.lp` program for external reproduction

## Environment setup

This repo is typically run in the existing conda env `aba-env`.

1) Activate env

```bash
conda activate aba-env
```

2) Install Python deps

```bash
pip install -r requirements.txt
```

3) Ensure solver binaries are available

```bash
clingo --version
wasp --version
```

## Manual walkthrough (mock-three-var)

```bash
cd ArgCausalDisco/encodings/test_lps/mock_three_var_manual
bash run_tests.sh
```

This checks:
- full semantic encoding with all facts is UNSAT,
- commenting out any one of the facts yields SAT,
- the adorned encoding yields a MUS containing all three facts.

## Running the automated tests

Run all MUS-related tests:

```bash
python -m pytest tests_mus.py -xvs
```

Run a single test:

```bash
python -m pytest tests_mus.py::TestMUSAnalysis::test_mock_three_var_manual_vs_mus -v
```

## Emit a complete MUS program and run WASP manually

Both the test harness and profiling script can emit the full adorned MUS program (`--emit-lp`).

### Step 1: Emit `.lp`

Example (profile a single size and write the adorned program):

```bash
python scripts/mus_abapc_profile.py   --node-sizes 7   --seed-base 2004   --solve-timeout 120   --emit-lp results/mus_7_2004.lp
```

Or via the test runner:

```bash
python tests_mus.py   --node-sizes 7   --solve-timeout 120   --emit-lp results/mus_7_{n}.lp
```

### Step 2: Run `clingo | wasp`

```bash
clingo results/mus_7_2004.lp --output=smodels | wasp --mus=mus --mus-algorithm=camus --print-mcses -n 0
```

Notes:
- `-n 0` enumerates all MUS.
- Omitting `-n` uses WASP’s default behavior (prints a single MUS).
- In the Python wrapper + profiling script:
  - `--max-muses ''` omits `-n` (WASP default),
  - `--max-muses 0` becomes `-n 0` (enumerate all),
  - `--max-muses N` becomes `-n N`.

## Profiling ABAPC vs MUS

Run a sweep and auto-emit programs:

```bash
python scripts/mus_abapc_profile.py   --node-sizes 5,6,7,8   --seed-base 2004   --solve-timeout 120   --emit-lp-dir results/mus_programs/
```

Quiet mode is useful for copy/pasteable timing summaries:

```bash
python scripts/mus_abapc_profile.py --node-sizes 8 --solve-timeout 800 --quiet
```

## Interpreting output (quick)

- A printed MUS like:
  - `[MUS #1]: mus(2) mus(7)`
  means “fact #2 and fact #7 are already enough to cause UNSAT”.

- A printed MCS like:
  - `[MCS #3]: mus(7)`
  means “removing fact #7 alone is enough to restore SAT”.

The emitted program preserves the ordering of `ext_*` facts, so `mus(i)` indices map back to those facts deterministically.
