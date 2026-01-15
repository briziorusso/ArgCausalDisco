"""MUS/MCS support for CausalABA via an external ASP core solver.

This module builds an *adorned* ASP program where each "wrong test" is guarded by
an assumption atom `mus(i)`:

- A choice layer `{mus(i)}.` makes each test optionally enabled.
- The original test constraint/fact is rewritten as `fact :- mus(i).` so it only
  applies when its assumption is selected.

Then we compute:

- **MUSes**: minimal sets of assumptions that still make the program UNSAT.
- **MCSes** (optionally): minimal sets of assumptions to *remove* to regain SAT.

We do this by piping clingo's smodels output into WASP, which implements MUS/MCS
enumeration algorithms:

  clingo adorned.lp --output=smodels | wasp --mus=mus --mus-algorithm=camus --print-mcses -n 0

The main entrypoint is `CausalABA_MUS(...)`, which can also *emit* the adorned
program to disk (for reproducible standalone runs) instead of invoking solvers.

Notes:
- This code expects `clingo` and `wasp` to be available (usually `aba-env` + the
  repo-local WASP build).
- WASP's `-n` behaviour matters: `-n 0` means "enumerate all"; omitting `-n`
  often means "return one" depending on build/config.
"""

__author__ = "Fabrizio Russo"
__email__ = "fabrizio@imperial.ac.uk"
__copyright__ = "Copyright (c) 2025 Fabrizio Russo"

import os
import sys
import logging
import tempfile
import subprocess
import time
import re
from pathlib import Path
from itertools import combinations
from typing import Optional, Tuple, List, Union
from causalaba import compile_and_ground


def parse_facts_from_file(facts_location: str) -> tuple[list[str], dict[int, str]]:
    """Parse `ext_indep` / `ext_dep` facts from a `.lp` / `.asp` text file.
    
    Used by tests/harnesses to load facts from `encodings/test_lps/...`.
    
    The parser is intentionally permissive:
    - Ignores blank lines and comment lines starting with `%`.
    - Ignores ASP directives starting with `#`.
    - Collects only lines containing `ext_indep` or `ext_dep`.
    
    Args:
        facts_location: Path to an ASP file.
    
    Returns:
        `(facts, fact_mapping)` where `facts` is a list of fact strings without the trailing `.`
        and `fact_mapping` maps 1-based indices to the original fact strings including the `.`.
    """
    facts = []
    fact_mapping = {}
    fact_counter = 0
    
    with open(facts_location, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.startswith('#'):
                continue
            if 'ext_indep' in line or 'ext_dep' in line:
                if line.endswith('.'):
                    fact_counter += 1
                    facts.append(line[:-1])  # Remove trailing period
                    fact_mapping[fact_counter] = line  # Keep period for display
    
    logging.debug(f"Parsed {len(facts)} facts from {facts_location}")
    return facts, fact_mapping


def run_mus_solver(
    program_str: str,
    gringo_path: str = "clingo",
    wasp_path: str = "wasp",
    *,
    max_muses: Optional[int] = None,
    mus_algorithm: Optional[str] = None,
    print_mcses: bool = False,
    camus_mcs_threshold: Optional[int] = None,
    camus_mus_threshold: Optional[int] = None,
    return_mcses: bool = False,
    solve_timeout: Optional[float] = None,
) -> Union[List[List[int]], Tuple[List[List[int]], List[List[int]]]]:
    """Run the external `clingo | wasp` pipeline to enumerate MUSes (and optionally MCSes).
    
    The input `program_str` is expected to be an adorned program that contains assumption atoms named `mus/1`.
    
    This executes:
      `clingo <tmp.lp> --output=smodels | wasp --mus=mus ...`
    
    Args:
        program_str: Full adorned ASP program as a string.
        gringo_path: Path/name for the clingo binary (kept as `gringo_path` for historical reasons).
        wasp_path: Path to the WASP binary.
        max_muses: If provided, passes `-n <max_muses>`; use `0` to enumerate all.
        mus_algorithm: Optional WASP `--mus-algorithm` value (e.g. `camus`).
        print_mcses: If True, passes `--print-mcses` and enables CAMUS by default.
        return_mcses: If True, returns `(mus_list, mcs_list)`; otherwise returns only `mus_list`.
        solve_timeout: Optional timeout (seconds) for the overall external run.
    
    Returns:
        Either `mus_list` (list of MUSes, each a list of ints), or `(mus_list, mcs_list)` when
        `return_mcses=True`.
    """
    if print_mcses:
        logging.info("   Solving for MUS and MCS...")
    else:
        logging.info("   Solving for MUS...")
    
    try:
        # Write program to temporary file
        fd, program_file = tempfile.mkstemp(suffix='_mus.lp', text=True)
        os.close(fd)
        with open(program_file, 'w') as f:
            f.write(program_str)
        
        # Build clingo command to ground with smodels output
        clingo_cmd = [gringo_path, program_file, '--output=smodels']
        
        # Build wasp command to compute MUS over mus/1
        # Note: WASP can combine --mus with -n to enumerate MUSes.
        # - If max_muses is None: omit -n flag (WASP default: outputs only 1 MUS but finds all MCS)
        # - If max_muses == 0: use -n 0 (enumerate all MUS)
        # - If max_muses > 0: use -n <max_muses> (limit MUS enumeration)
        wasp_cmd = [wasp_path, '--mus=mus']
        if max_muses is not None:
            wasp_cmd.extend(['-n', str(max_muses)])

        # Optional: select MUS algorithm / print MCSes (CAMUS)
        if print_mcses and mus_algorithm is None:
            mus_algorithm = 'camus'

        if mus_algorithm:
            alg = mus_algorithm
            if mus_algorithm == 'camus':
                # WASP syntax: camus,[mcs_th,[mus_th]]
                if camus_mcs_threshold is not None:
                    alg = f"camus,{camus_mcs_threshold}"
                    if camus_mus_threshold is not None:
                        alg = f"{alg},{camus_mus_threshold}"
            wasp_cmd.extend(['--mus-algorithm', alg])

        if print_mcses:
            wasp_cmd.append('--print-mcses')
        
        logging.debug(f"Clingo command: {' '.join(clingo_cmd)}")
        logging.debug(f"Wasp command: {' '.join(wasp_cmd)}")
        
        # Run clingo and pipe to wasp
        clingo_process = subprocess.Popen(
            clingo_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        wasp_process = subprocess.Popen(
            wasp_cmd,
            stdin=clingo_process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Close clingo's stdout in parent so wasp receives EOF when clingo finishes
        if clingo_process.stdout is not None:
            clingo_process.stdout.close()
        
        # Wait for wasp to complete (with optional timeout)
        try:
            if solve_timeout is not None:
                wasp_output, wasp_error = wasp_process.communicate(timeout=solve_timeout)
            else:
                wasp_output, wasp_error = wasp_process.communicate()
        except subprocess.TimeoutExpired as e:
            # WASP may have already produced partial output (some MUS/MCS) before the timeout.
            # Preserve and parse what we have instead of discarding it.
            logging.error(
                f"Wasp MUS solve timed out after {solve_timeout} seconds. Cancelling processes."
            )

            partial_out = ""
            partial_err = ""
            # In Python 3.12, TimeoutExpired can include partial stdout/stderr.
            try:
                partial_out = e.output or ""
            except Exception:
                partial_out = ""
            try:
                partial_err = e.stderr or ""
            except Exception:
                partial_err = ""

            try:
                wasp_process.kill()
            except Exception:
                pass
            try:
                clingo_process.kill()
            except Exception:
                pass

            # Attempt to collect any remaining buffered output after killing.
            tail_out = ""
            tail_err = ""
            try:
                tail_out, tail_err = wasp_process.communicate(timeout=1)
            except Exception:
                tail_out, tail_err = "", ""
            try:
                clingo_process.wait(timeout=1)
            except Exception:
                pass

            wasp_output = f"{partial_out}{tail_out}"
            wasp_error = f"{partial_err}{tail_err}"

            # Continue below to parse any partial MUS/MCS from wasp_output.
        
        # Wait for clingo to complete and get its stderr
        clingo_process.wait()
        clingo_error = ""
        if clingo_process.stderr is not None:
            clingo_error = clingo_process.stderr.read()
            clingo_process.stderr.close()
        
        if clingo_process.returncode != 0:
            # If WASP terminates early, clingo can receive SIGPIPE while writing to the pipe.
            # Treat that case as non-fatal if WASP succeeded and produced output.
            if clingo_process.returncode == -13 and wasp_process.returncode == 0:
                logging.debug("Clingo exited with SIGPIPE (-13) after WASP finished; ignoring.")
            else:
                logging.error(f"Clingo failed with return code {clingo_process.returncode}: {clingo_error}")
                return []
        elif clingo_error.strip():
            # Clingo succeeded but had warnings/info messages
            logging.debug(f"Clingo stderr (informational): {clingo_error}")
        
        if wasp_process.returncode != 0:
            # If we timed out and killed WASP, returncode will typically be non-zero.
            # We still try to parse any partial output it produced.
            if solve_timeout is None:
                logging.error(f"Wasp failed with return code {wasp_process.returncode}: {wasp_error}")
                return []
            logging.debug(
                f"Wasp exited non-zero (likely due to timeout/kill). Stderr: {wasp_error}"
            )
        
        logging.debug(f"Wasp output:\n{wasp_output}")
        
        # Parse MUS/MCS from output (may be partial if timed out)
        mus_list: List[List[int]] = []
        mcs_list: List[List[int]] = []
        for line in wasp_output.split('\n'):
            if line.startswith('[MUS #'):
                # Format: [MUS #1]: mus(1) mus(3) mus(2)
                match = re.search(r'\[MUS #\d+\]:\s*(.*)', line)
                if match:
                    mus_content = match.group(1).strip()
                    if mus_content:  # Non-empty MUS
                        mus_predicates = re.findall(r'mus\((\d+)\)', mus_content)
                        mus = [int(n) for n in mus_predicates]
                        mus_list.append(mus)
                        # Per-MUS logs are noisy; keep at debug level.
                        logging.debug(f"Found MUS: {mus}")
                    else:
                        logging.debug("Found empty MUS (program is satisfiable)")
            elif line.startswith('[MCS #'):
                # Format: [MCS #1]: mus(1) mus(3)
                match = re.search(r'\[MCS #\d+\]:\s*(.*)', line)
                if match:
                    mcs_content = match.group(1).strip()
                    if mcs_content:
                        mcs_predicates = re.findall(r'mus\((\d+)\)', mcs_content)
                        mcs = [int(n) for n in mcs_predicates]
                        mcs_list.append(mcs)
                        logging.debug(f"Found MCS: {mcs}")

        if solve_timeout is not None and (wasp_process.returncode != 0):
            # This is the timeout path: surface partial progress if any.
            logging.warning(
                f"WASP timed out; returning partial results: {len(mus_list)} MUS, {len(mcs_list)} MCS"
            )
        
        # Cleanup
        try:
            os.remove(program_file)
        except:
            pass
        
        if return_mcses:
            return mus_list, mcs_list

        return mus_list
        
    except FileNotFoundError as e:
        logging.error(f"Error running MUS solver: {e}")
        logging.error("Make sure clingo and wasp are installed and in your PATH")
        return []
    except Exception as e:
        logging.error(f"Unexpected error running MUS solver: {e}")
        return []


def build_mus_program(
    n_nodes: int,
    facts: list[str],
    facts_location: str = "",
    *,
    deadline: Optional[float] = None,
    timing_recorder: dict | None = None,
) -> str:
    """Build the full adorned program used for MUS/MCS enumeration.
    
    This uses `compile_and_ground(...)` to obtain the base CausalABA encoding for `n_nodes`, then
    adds an assumption layer over the provided wrong-test facts.
    
    The resulting program contains:
    - a `{mus(i)}.` choice rule for each fact (1-based indexing), and
    - a guarded form `fact :- mus(i).` so each test can be toggled via assumptions.
    
    Args:
        n_nodes: Number of variables/nodes for the instance.
        facts: Facts/constraints (typically derived from PC output).
        facts_location: Optional path used to derive a "specific rules" file.
        deadline: Optional absolute perf_counter deadline for the build phase.
        timing_recorder: Optional dict to accumulate build/compile timings.
    
    Returns:
        The complete adorned ASP program as a string.
    
    Raises:
        TimeoutError: If `deadline` is exceeded during program construction.
    """
    logging.debug("Building MUS program using compile_and_ground...")
    
    # Determine specific rules file location
    if facts_location:
        # Default: same directory as facts file
        facts_path = Path(facts_location)
        specific_rules_file = str(facts_path.parent / f"{facts_path.stem}_specific.lp")
    else:
        # Fallback to temporary file
        fd, specific_rules_file = tempfile.mkstemp(suffix='_specific.lp', text=True)
        os.close(fd)
    
    # Try to load existing specific rules
    specific_rules_text = None
    if os.path.exists(specific_rules_file):
        try:
            logging.debug(f"Found existing specific rules at {specific_rules_file}")
            with open(specific_rules_file, 'r') as f:
                specific_rules_text = f.read()
            logging.info(f"Loaded existing specific rules ({len(specific_rules_text)} chars)")
        except Exception as e:
            logging.warning(f"Failed to load existing specific rules: {e}")
            specific_rules_text = None
    
    # Generate specific rules if not loaded
    if specific_rules_text is None:
        # Parse facts to get indep_facts and dep_facts dicts
        indep_facts = {}
        dep_facts = {}
        
        if facts_location:
            with open(facts_location, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('%') or line.startswith('#'):
                        continue
                    
                    # Parse ext_indep or ext_dep
                    match = re.match(r'(ext_indep|ext_dep)\((\d+),(\d+),([^\)]+)\)', line)
                    if not match:
                        continue
                    
                    fact_type, x, y, s = match.groups()
                    x, y = int(x), int(y)
                    
                    # Convert s to tuple format
                    if s == 'empty':
                        s_tuple = ()
                    else:
                        # Parse s0, sy1y2, etc. to extract indices
                        match_indices = re.match(r's((?:0|[1-9]\d*)*)', s)
                        if match_indices:
                            indices_str = match_indices.group(1)
                            if indices_str:
                                s_tuple = tuple(int(indices_str[i]) for i in range(len(indices_str)))
                            else:
                                s_tuple = ()
                        else:
                            s_tuple = ()
                    
                    facts_dict = indep_facts if fact_type == 'ext_indep' else dep_facts
                    if (x, y) not in facts_dict:
                        facts_dict[(x, y)] = set()
                    facts_dict[(x, y)].add(s_tuple)
        
            # Call compile_and_ground with dump_specific
            logging.debug(f"Calling compile_and_ground to dump specific rules to {specific_rules_file}")
            ctl = compile_and_ground(
                n_nodes,
                facts_location=facts_location,
                skeleton_rules_reduction=True,  # Enable skeleton-rules optimization for MUS
                weak_constraints=False,
                indep_facts=indep_facts,
                dep_facts=dep_facts,
                opt_mode='optN',
                out_n=0,
                show=['arrow'],
                pre_grounding=False,
                ext_flag=False,
                prior_knowledge=None,
                max_path_length=None,
                max_conditioning_size=None,
                collider_tree_depth=None,
                cycle_length=None,
                dump_specific=specific_rules_file,
                deadline=deadline,
                timing_recorder=timing_recorder,
            )
            
            # Load the dumped specific rules
            with open(specific_rules_file, 'r') as f:
                specific_rules_text = f.read()
            
            logging.debug(f"Generated and loaded {len(specific_rules_text)} chars of specific rules")
    
    # Load base causalaba.lp (skip #program directive)
    causalaba_lp = Path(__file__).resolve().parent / 'encodings' / 'causalaba.lp'
    base_program = ""
    with open(causalaba_lp, 'r') as f:
        for line in f:
            if line.strip().startswith('#program main'):
                continue
            # Replace n_vars with actual value
            line = line.replace('n_vars', str(n_nodes - 1))
            base_program += line
    
    # Build the complete program
    program_lines = [
        base_program,
        "",
        "% ===== Specific Rules Generated by compile_and_ground =====",
        specific_rules_text,
        "",
        "% ===== Assumption Layer for MUS =====",
    ]
    
    # Add choice rules for mus/1 assumptions
    for i in range(1, len(facts) + 1):
        program_lines.append(f"{{mus({i})}}.")
    
    program_lines.append("")
    program_lines.append("% ===== Adorned Facts =====")
    
    # Add adorned facts: fact:- mus(i) (no spaces around :- for WASP compatibility)
    for i, fact in enumerate(facts, 1):
        program_lines.append(f"{fact}:-mus({i}).")
    
    program = '\n'.join(program_lines)
    logging.debug(f"Built complete MUS program: {len(program)} chars")
    
    return program


def CausalABA_MUS(
    n_nodes: int,
    facts_location: str = "",
    gringo_path: str = "clingo",
    wasp_path: str = "wasp",
    *,
    max_muses: Optional[int] = None,
    mus_algorithm: Optional[str] = None,
    print_mcses: bool = False,
    camus_mcs_threshold: Optional[int] = None,
    camus_mus_threshold: Optional[int] = None,
    solve_timeout: Optional[float] = None,
    emit_lp: Optional[str] = None,
    timing_recorder: dict | None = None,
) -> dict:
    """High-level MUS/MCS analysis over `ext_indep`/`ext_dep` wrong-test facts.
    
    Workflow:
    1) Parse wrong-test facts from `facts_location`.
    2) Build the adorned program via `build_mus_program(...)`.
    3) Optionally write the program to `emit_lp` (for reproducibility).
    4) Run `run_mus_solver(...)` to enumerate MUSes (and optionally MCSes).
    
    Reproducible external run (after emitting):
    
      clingo adorned.lp --output=smodels | wasp \
          --mus=mus --mus-algorithm=camus --print-mcses -n 0
    
    Returns:
        A dict containing `mus_list`, `mus_facts`, `n_mus`, `fact_mapping`, and (when enabled)
        `mcs_list`, `mcs_facts`, `n_mcs`, plus timing fields.
    """
    logging.info("Running CausalABA MUS...")
    
    if not facts_location:
        logging.error("facts_location is required for MUS analysis")
        return {'mus_list': [], 'mus_facts': [], 'n_mus': 0, 'fact_mapping': {}}
    
    # Step 1: Parse facts from file
    facts, fact_mapping = parse_facts_from_file(facts_location)
    
    if not facts:
        logging.error(f"No ext_indep/ext_dep facts found in {facts_location}")
        return {'mus_list': [], 'mus_facts': [], 'n_mus': 0, 'fact_mapping': {}}
    
    logging.debug(f"Found {len(facts)} facts to analyze for MUS")
    if len(fact_mapping) <= 20:
        for idx, fact in fact_mapping.items():
            logging.debug(f"  Fact {idx}: {fact}")
    else:
        preview = [fact_mapping[i] for i in sorted(fact_mapping.keys())[:5]]
        logging.debug(f"  Preview facts: {preview} (showing 5 of {len(fact_mapping)})")
    
    # Step 2: Build MUS program with assumption layer on top of CausalABA encoding
    # Treat solve_timeout as an end-to-end budget (build + solve)
    deadline = (time.perf_counter() + float(solve_timeout)) if solve_timeout is not None else None
    build_start = time.perf_counter()
    try:
        program = build_mus_program(n_nodes, facts, facts_location, deadline=deadline, timing_recorder=timing_recorder)
        build_time = time.perf_counter() - build_start
        logging.info(f"   MUS Build time: {build_time:.3f}s (nodes={n_nodes})")
        
        # Optionally save the complete MUS program for debugging
        if emit_lp:
            with open(emit_lp, 'w') as f:
                f.write(program)
            logging.info(f"Emitted complete MUS program to {emit_lp}")
    except TimeoutError:
        build_time = time.perf_counter() - build_start
        logging.error("MUS program build exceeded the timeout budget.")
        logging.info(f"   MUS Build time: {build_time:.3f}s (nodes={n_nodes})")
        # Return empty results to indicate timeout (consistent with solver timeout behavior)
        return {
            'mus_list': [],
            'mus_facts': [],
            'n_mus': 0,
            'fact_mapping': fact_mapping,
            'mcs_list': [],
            'mcs_facts': [],
            'n_mcs': 0,
            'mus_build_time': build_time,
            'mus_solve_time': 0.0,
            'mus_total_time': build_time,
        }

    # Compute remaining time for the solver stage
    remaining_timeout: Optional[float] = None
    if deadline is not None:
        remaining_timeout = max(0.0, deadline - time.perf_counter())
        if remaining_timeout == 0.0:
            logging.error("No time remaining for MUS solve after build phase.")
            return {
                'mus_list': [],
                'mus_facts': [],
                'n_mus': 0,
                'fact_mapping': fact_mapping,
                'mcs_list': [],
                'mcs_facts': [],
                'n_mcs': 0,
            }
    
    # Step 3: Run MUS solver (optionally with CAMUS/MCS printing)
    mus_list: List[List[int]] = []
    mcs_list: List[List[int]] = []

    solve_start = time.perf_counter()
    if print_mcses or mus_algorithm:
        mus_mcs = run_mus_solver(
            program,
            gringo_path,
            wasp_path,
            max_muses=max_muses,
            mus_algorithm=mus_algorithm,
            print_mcses=print_mcses,
            camus_mcs_threshold=camus_mcs_threshold,
            camus_mus_threshold=camus_mus_threshold,
            return_mcses=True,
            solve_timeout=remaining_timeout,
        )
        if isinstance(mus_mcs, tuple):
            mus_list, mcs_list = mus_mcs
        else:
            # Defensive fallback: older code paths may return only mus_list.
            mus_list = mus_mcs
            mcs_list = []
    else:
        mus_only = run_mus_solver(
            program,
            gringo_path,
            wasp_path,
            max_muses=max_muses,
            solve_timeout=remaining_timeout,
        )
        if isinstance(mus_only, list):
            mus_list = mus_only
        else:
            mus_list, mcs_list = mus_only
    solve_time = time.perf_counter() - solve_start
    total_time = build_time + solve_time
    logging.info(f"MUS Solve time: {solve_time:.3f}s; Total: {total_time:.3f}s (nodes={n_nodes})")
    
    # Step 4: Map MUS indices back to actual facts
    mus_facts = []
    for mus in mus_list:
        mus_fact_list = [fact_mapping.get(idx, f"mus({idx})") for idx in mus]
        mus_facts.append(mus_fact_list)
    
    if mus_list:
        sizes = [len(m) for m in mus_list]
        logging.info(
            f"   MUS facts resolved: {len(mus_list)} cores (min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f})"
        )
        logging.debug(f"   MUS facts (resolved): {mus_facts}")
    else:
        logging.info("   MUS facts (resolved): []")
    
    result = {
        'mus_list': mus_list,
        'mus_facts': mus_facts,
        'n_mus': len(mus_list),
        'fact_mapping': fact_mapping,
        'mcs_list': mcs_list,
        'mcs_facts': [[fact_mapping.get(idx, f"mus({idx})") for idx in mcs] for mcs in mcs_list],
        'n_mcs': len(mcs_list),
        'mus_build_time': build_time,
        'mus_solve_time': solve_time,
        'mus_total_time': total_time,
    }
    
    return result
