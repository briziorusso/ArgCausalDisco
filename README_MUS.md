# MUS Analysis for CausalABA

This module computes **Minimal Unsatisfiable Subsets (MUS)** of independence/dependence facts that create unsatisfiability when combined with the full CausalABA causal graph constraints.

## Overview

When a set of independence/dependence facts is incompatible with the causal graph structure (UNSAT), MUS analysis identifies the minimal subsets of facts that cause the conflict. This is useful for:

- Diagnosing incompatible fact combinations
- Understanding which facts constrain the search space most strongly
- Explaining conflicts in causal discovery tasks

## Key Files

### `causalaba_mus.py`

Main MUS computation engine using the WASP solver.

**Core Functions:**

- `parse_facts_from_file(facts_location)` — Parse ext_indep/ext_dep facts from a `.lp` file
- `build_semantic_mus_program(n_nodes, facts)` — Build the full CausalABA program with mus/1 assumption layer
- `run_semantic_mus_solver(program_str, gringo_path, wasp_path)` — Execute clingo + WASP pipeline
- `CausalABA_MUS(n_nodes, facts_location, gringo_path, wasp_path)` — Main entry point

**Workflow:**

1. Parse facts from file
2. Build semantic ASP program with:
   - Choice rules `{mus(i)}.` for each fact
   - Conditional activation `fact_i :- mus(i).`
   - Full CausalABA encoding (DAG generation, active paths, colliders, nonblockers, acyclicity)
   - Conflict constraints (dep/indep propagation)
3. Ground with clingo → output smodels format
4. Pipe to WASP with `--mus=mus` flag to compute MUS over mus/1 atoms
5. Parse output and map assumption indices back to facts

**Example:**

```python
from causalaba_mus import CausalABA_MUS

result = CausalABA_MUS(
    n_nodes=3,
    facts_location="facts.lp",
    gringo_path="clingo",
    wasp_path="/path/to/wasp",
    mus_algorithm="camus",
    print_mcses=True,
)

print(f"Found {result['n_mus']} MUS cores")
for i, mus_facts in enumerate(result['mus_facts']):
    print(f"  Core #{i+1}: {mus_facts}")
```

### `tests_mus.py`

Test suite for MUS analysis with three focused tests:

**Test 1: `test_parsing_facts_from_file`**
- Verifies parsing of ext_indep/ext_dep facts
- Handles comments, directives, filtering

**Test 2: `test_adorning_with_mus_assumptions`**
- Tests the assumption layer structure
- Verifies `{mus(i)}.` choice rules generated
- Confirms facts guarded by `mus(i)` atoms
- Ensures no `#show` statements in output

**Test 3: `test_mock_three_var_manual_vs_mus`**
- Main integration test linking manual verification to MUS computation
- **Step 1:** All three facts → UNSAT
- **Step 2:** Remove each fact individually → SAT (different models per removal)
- **Step 3:** MUS discovers all three facts as one core
- Shows that manual removal behavior matches automatic MUS computation

**Run tests:**

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1
python -m pytest tests_mus.py -v
```

Expected output:
```
tests_mus.py::TestMUSAnalysis::test_adorning_with_mus_assumptions PASSED
tests_mus.py::TestMUSAnalysis::test_mock_three_var_manual_vs_mus PASSED
tests_mus.py::TestMUSAnalysis::test_parsing_facts_from_file PASSED
```

Run a single test:

```bash
# Just the parsing test
python -m pytest tests_mus.py::TestMUSAnalysis::test_parsing_facts_from_file -v

# Just the mock-three-var integration test
python -m pytest tests_mus.py::TestMUSAnalysis::test_mock_three_var_manual_vs_mus -v
```

## How It Works

### The mus/1 Assumption Framework

Facts are guarded by choice rules, enabling WASP to explore subsets:

```prolog
% Choice rules for each fact
{mus(1)}.
{mus(2)}.
{mus(3)}.

% Conditional activation
ext_indep(1,2,s0) :- mus(1).
ext_dep(1,2,empty) :- mus(2).
ext_indep(0,1,empty) :- mus(3).

% Full CausalABA encoding (DAG, active paths, conflicts, etc.)
...
```

When WASP runs with `--mus=mus`, it:
1. Tests which subsets of {mus(1), mus(2), mus(3)} make the program UNSAT
2. Returns the minimal ones (cannot remove any atom without becoming SAT)

### MCSes (via CAMUS)

WASP can optionally print **Minimal Correcting Sets (MCSes)** while computing MUSes when using the CAMUS algorithm.
This can be helpful if you want to see (or parse) which assumptions must be removed to restore satisfiability.

In our CausalABA workflow, **MCSes are computed over the same assumption atoms as MUSes**, and we map them back to the corresponding `ext_*` facts.

CLI example (mock-three-var adorned encoding):

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1/encodings/test_lps/mock_three_var_manual
clingo facts_complete_causalaba_adorned.lp --output=smodels | \
    /vol/bitbucket/fr920/wasp/build/release/wasp --mus=mus --mus-algorithm=camus --print-mcses -n 0
```

Typical output:

```text
[MCS #1]: mus(1)
[MCS #2]: mus(2)
[MCS #3]: mus(3)
[MUS #1]: mus(1) mus(3) mus(2)
```

Python API example (return MUS + MCS):

```python
from causalaba_mus import CausalABA_MUS

result = CausalABA_MUS(
        n_nodes=3,
        facts_location="facts.lp",
        gringo_path="clingo",
        wasp_path="/vol/bitbucket/fr920/wasp/build/release/wasp",
        mus_algorithm="camus",
        print_mcses=True,
)

print(result["n_mus"], "MUSes")
print(result.get("n_mcs", 0), "MCSes")
print("Smallest MCS size:", min(map(len, result.get("mcs_facts", [])), default=None))
```

### MCS analysis summary (frequency + examples)

In [tests_mus.py](tests_mus.py), the PC-based test prints an **MCS summary**:

- **PC Fact ranking by MCS frequency**: counts how often each PC-derived `ext_*` fact appears across all MCSes.
    - Intuition: facts that appear in many MCSes are frequently part of a minimal “fix”, so they are good candidates for review/removal.
- **Examples of smallest MCSes**: prints a few smallest MCSes with facts labeled `WRONG`/`CORRECT` (based on ground truth comparison).

This summary is designed to answer: “Which of the facts returned by PC (and input into CausalABA) are most often implicated in minimal corrections?”

### Mock-Three-Var Example

**Facts:**
- `ext_indep(1,2,s0)` — 1 is independent of 2 given {0}
- `ext_dep(1,2,empty)` — 1 depends on 2 unconditionally
- `ext_indep(0,1,empty)` — 0 is independent of 1 unconditionally

**Result:**
- All three facts together: UNSAT
- Remove any one fact: SAT (with different causal structures)
- MUS: `[mus(1) mus(3) mus(2)]` — all three facts form the minimal core

## Manual Verification

Manual tests for the mock-three-var scenario are in:
[encodings/test_lps/mock_three_var_manual/](encodings/test_lps/mock_three_var_manual/)

Run them with:
```bash
cd encodings/test_lps/mock_three_var_manual
bash run_tests.sh
```

Steps:
1. Test full semantic encoding with all facts: UNSAT
2. Test with last fact commented: SAT
3. Test adorned semantic encoding with WASP MUS: one core with all three facts

## Dependencies

The MUS analysis pipeline requires:

- **Python** (3.9+): Core runtime
- **clingo** (5.6.2+): ASP grounder and solver (for compiling and grounding CausalABA programs)
- **WASP** (2.0+): ASP solver with MUS/MCS support (for computing minimal unsatisfiable subsets)
- **Python packages**: pandas, numpy, scikit-learn, causal-learn, gcastle, networkx (see below)

### Python Environment Setup

#### 1. Create a Conda Environment

If you don't have a conda environment yet, create one:

```bash
cd /vol/bitbucket/fr920
conda create -n aba-env python=3.10 -y
conda activate aba-env
```

#### 2. Install Core Dependencies

Install clingo via conda (recommended):

```bash
conda install -c conda-forge clingo=5.6.2 -y
```

Install Python packages:

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1
pip install -r requirements.txt
```

**Note:** Some packages in `requirements.txt` (e.g., `gcastle`, `notears`) have optional dependencies. For basic MUS profiling, you can install only the essentials:

```bash
pip install pandas numpy scikit-learn networkx tqdm
```

#### 3. Build or Locate WASP

WASP is required for MUS computation. You have two options:

**Option A: Use Pre-built WASP (if available)**

If WASP is already built in your workspace:

```bash
# Check if WASP binary exists
ls -la /vol/bitbucket/fr920/wasp/build/release/wasp
```

If the binary exists, set its path when running the profiler:

```bash
python scripts/mus_abapc_profile.py \
    --node-sizes 7 \
    --wasp-path /vol/bitbucket/fr920/wasp/build/release/wasp \
    --emit-lp results/mus_adorned.lp
```

**Option B: Build WASP from Source**

If WASP is not built, clone and build it:

```bash
cd /vol/bitbucket/fr920
git clone https://github.com/potassco/wasp.git  # If not already cloned
cd wasp
mkdir -p build/release
cd build/release
cmake ../.. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

After building, the binary will be at `/vol/bitbucket/fr920/wasp/build/release/wasp`.

**Verify WASP is working:**

```bash
/vol/bitbucket/fr920/wasp/build/release/wasp --version
```

### Optional: Test the Setup

Run a quick smoke test:

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1

# Test clingo
clingo --version

# Test WASP
/vol/bitbucket/fr920/wasp/build/release/wasp --version

# Run a small MUS test
python -m pytest tests_mus.py::TestMUSAnalysis::test_parsing_facts_from_file -v
```

## Notes

- Ensure facts are guarded by assumptions; otherwise MUS may be empty even if program is UNSAT
- Conditioning set membership (in/2) must be explicitly declared for non-empty sets
- No `#show` directives — WASP parses raw output containing `[MUS #i]: ...` lines
- For reproducibility, fact order in files is preserved through indexing

## External MUS Run (Adorned .lp)

You can generate a complete adorned MUS program and execute it externally with clingo + WASP.

### Single Run with Fixed Path

1) Emit the adorned program for a single random graph (example: 7 nodes, seed 2004):

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1
python scripts/mus_abapc_profile.py \
    --node-sizes 7 \
    --seed-base 2004 \
    --solve-timeout 120 \
    --emit-lp results/mus_adorned_7_2004.lp
```

2) Run externally (CAMUS with MCS printing):

```bash
clingo results/mus_adorned_7_2004.lp --output=smodels | \
    /vol/bitbucket/fr920/wasp/build/release/wasp --mus=mus --mus-algorithm=camus --print-mcses -n 0
```

### Sweep with Auto-Naming

To profile multiple node sizes and auto-generate one `.lp` per `(n_nodes, seed)`:

```bash
python scripts/mus_abapc_profile.py \
    --node-sizes 5,6,7,8 \
    --seed-base 2004 \
    --solve-timeout 120 \
    --emit-lp-dir results/mus_programs/
```

This creates:
- `results/mus_programs/mus_5_2004.lp`
- `results/mus_programs/mus_6_2004.lp`
- `results/mus_programs/mus_7_2004.lp`
- `results/mus_programs/mus_8_2004.lp`

With retries on SAT:

```bash
python scripts/mus_abapc_profile.py \
    --node-sizes 5,6,7,8 \
    --seed-base 2004 \
    --solve-timeout 120 \
    --rep_unsat 2 \
    --emit-lp-dir results/mus_programs/
```

When a SAT instance is found, a retry seed is used (2005, 2006, etc.), and the new `.lp` filename reflects it.

**Note:** `--emit-lp` and `--emit-lp-dir` are mutually exclusive; if both are specified, `--emit-lp` takes precedence.

The `mus(i)` indices in the output correspond to the order of the facts in the emitted program.

## Running All MUS Tests Together

To run all MUS-related tests in one shot:

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1
python -m pytest tests_mus.py -xvs
```

Use `-k <substring>` to filter:

```bash
python -m pytest -k mus -v
```

## Quick Reference: Full Workflow

### 1. Setup (One-Time)

```bash
conda create -n aba-env python=3.10 -y
conda activate aba-env
conda install -c conda-forge clingo=5.6.2 -y
cd /vol/bitbucket/fr920/ArgCausalDisco-1
pip install pandas numpy scikit-learn networkx tqdm  # Minimal dependencies
```

Build WASP (if not already built):

```bash
cd /vol/bitbucket/fr920
git clone https://github.com/potassco/wasp.git
cd wasp
mkdir -p build/release && cd build/release
cmake ../.. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 2. Run Tests

```bash
cd /vol/bitbucket/fr920/ArgCausalDisco-1
python -m pytest tests_mus.py -xvs
```

### 3. Profile Single Instance and Emit .lp

```bash
python scripts/mus_abapc_profile.py \
    --node-sizes 7 \
    --seed-base 2004 \
    --solve-timeout 120 \
    --emit-lp results/mus_adorned_7_2004.lp
```

### 4. Run Emitted Program Externally

```bash
clingo results/mus_adorned_7_2004.lp --output=smodels | \
    /vol/bitbucket/fr920/wasp/build/release/wasp \
    --mus=mus --mus-algorithm=camus --print-mcses -n 0
```

### 5. Profile a Sweep with Auto-Naming and Retries

```bash
python scripts/mus_abapc_profile.py \
    --node-sizes 5,6,7,8 \
    --seed-base 2004 \
    --solve-timeout 120 \
    --rep_unsat 2 \
    --emit-lp-dir results/mus_programs/
```

This creates:
- `results/mus_programs/mus_5_2004.lp`, `mus_6_2004.lp`, `mus_7_2004.lp`, `mus_8_2004.lp` (initial seeds)
- Any retry files if SAT instances are found (e.g., `mus_6_2005.lp` if seed 2004 was SAT)

