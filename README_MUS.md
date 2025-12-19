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
    wasp_path="/path/to/wasp"
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

- `clingo` — ASP grounder (5.8.0+)
- `WASP` — ASP solver with MUS support (2.0+)
- `causalaba.py` — CausalABA framework
- Standard Python libraries: os, sys, logging, tempfile, subprocess, re, pathlib

## Notes

- Ensure facts are guarded by assumptions; otherwise MUS may be empty even if program is UNSAT
- Conditioning set membership (in/2) must be explicitly declared for non-empty sets
- No `#show` directives — WASP parses raw output containing `[MUS #i]: ...` lines
- For reproducibility, fact order in files is preserved through indexing

