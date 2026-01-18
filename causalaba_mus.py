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
import hashlib
import json
from pathlib import Path
from itertools import combinations
from typing import Optional, Tuple, List, Union
from causalaba import compile_and_ground


def _normalize_ext_fact_key(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith('#external'):
        s = s[len('#external'):].strip()
    if s.endswith('.'):
        s = s[:-1]
    return s.strip()


def parse_weights_from_wc_file(facts_wc_location: str) -> dict[str, int]:
    """Parse weights from a weak-constraint file.

    Expected formats include (with or without trailing '.' after the bracket):

      :~ ext_indep(...). [-123]
      :~ ext_dep(...). [-123].

    Returns a dict mapping normalized fact strings (without trailing '.') to positive weights.
    """
    weights: dict[str, int] = {}
    if not facts_wc_location:
        return weights

    with open(facts_wc_location, 'r') as f:
        for raw in f:
            line = (raw or "").strip()
            if not line or line.startswith('%'):
                continue
            # Capture the body before the bracket and the first integer inside the bracket.
            m = re.match(r"^:~\s*(.*?)\s*\[\s*([-+]?\d+)", line)
            if not m:
                continue
            fact_part = _normalize_ext_fact_key(m.group(1))
            try:
                w = abs(int(m.group(2)))
            except Exception:
                continue
            if fact_part:
                weights[fact_part] = w
    return weights


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

            # Accept facts declared as externals (CausalABA-compatible):
            #   #external ext_indep(...).
            #   #external ext_dep(...).
            if line.startswith('#external'):
                line = line[len('#external'):].strip()

            # Skip other directives.
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
    optimum_mcs_algorithm: Optional[str] = None,
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
    if optimum_mcs_algorithm:
        logging.info("   Solving for MUS and optimum MCS...")
    elif print_mcses:
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
        if (print_mcses or optimum_mcs_algorithm) and mus_algorithm is None:
            mus_algorithm = 'camus'

        if mus_algorithm:
            alg = mus_algorithm
            if mus_algorithm == 'camus':
                # WASP syntax: camus,[mcs_th,[mus_th]]
                # Note: thresholds are optional; passing 0 can disable MUS enumeration
                # (e.g., camus,0,0 => compute 0 MUS). Treat non-positive values as "unset".
                if camus_mcs_threshold is not None and camus_mcs_threshold > 0:
                    alg = f"camus,{camus_mcs_threshold}"
                    if camus_mus_threshold is not None and camus_mus_threshold > 0:
                        alg = f"{alg},{camus_mus_threshold}"
            wasp_cmd.extend(['--mus-algorithm', alg])

        # MCS modes:
        # - Standard enumeration: --print-mcses (requires CAMUS)
        # - Optimum MCS enumeration: --optimum-mcs-algorithm=<camus|emax>
        #   (requires objective-literal facts in the program).
        if optimum_mcs_algorithm:
            wasp_cmd.append(f"--optimum-mcs-algorithm={optimum_mcs_algorithm}")
        elif print_mcses:
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
        timed_out = False
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
            timed_out = True

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
        
        if (not timed_out) and clingo_process.returncode != 0:
            # If WASP terminates early (successfully or with an error), clingo can receive SIGPIPE
            # while writing to the pipe. Treat SIGPIPE as non-fatal and rely on WASP's return code.
            if clingo_process.returncode == -13:
                logging.debug("Clingo exited with SIGPIPE (-13) after WASP closed stdin; ignoring.")
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
        for line in (wasp_output or "").split('\n'):
            s = line.strip()
            if s.startswith('[MUS #'):
                # Format: [MUS #1]: mus(1) mus(3) mus(2)
                match = re.search(r'\[MUS #\d+\]:\s*(.*)', s)
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
            elif s.startswith('[MCS #'):
                # Formats:
                # - Standard: [MCS #1]: mus(1) mus(3)
                # - Optimum:  [MCS #1, Cost 123]: mus(1) mus(3)
                match = re.search(r'\[MCS #\d+(?:,\s*Cost\s+[^\]]+)?\]:\s*(.*)', s)
                if match:
                    mcs_content = match.group(1).strip()
                    if mcs_content:
                        mcs_predicates = re.findall(r'mus\((\d+)\)', mcs_content)
                        mcs = [int(n) for n in mcs_predicates]
                        # In optimum-MCS mode, WASP prints a sequence of improving cuts.
                        # The last printed MCS corresponds to the optimum; keep only that.
                        if optimum_mcs_algorithm:
                            mcs_list = [mcs]
                        else:
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
    base_program_path: Optional[str] = None,
    deadline: Optional[float] = None,
    timing_recorder: dict | None = None,
    weights: Optional[list[int]] = None,
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
    logging.debug("Building MUS program...")

    def _strip_show_directives(text: str) -> str:
        # IMPORTANT: If the program contains any `#show` directive, clingo's symbol table
        # for `--output=smodels` can omit non-shown symbols. WASP relies on those symbols
        # to identify atoms over the predicate passed to `--mus=...` (e.g., mus/1).
        # Incremental/base dumps often include `#show arrow/2.`; strip all `#show` lines
        # so mus/1 and other internal symbols remain visible to WASP.
        lines: list[str] = []
        for raw in (text or "").splitlines():
            if raw.lstrip().startswith('#show'):
                continue
            lines.append(raw)
        return "\n".join(lines)

    def _strip_block_edge_skeleton_reduction(text: str) -> str:
        # ABAPC_INC debug dumps may include skeleton-reduction artifacts in the form of
        # `block_edge/2` facts and constraints mentioning `block_edge(...)`.
        # Baseline MUS programs are built with skeleton_rules_reduction=False; to keep
        # MUS/MCS/OptMCS analysis comparable, strip these restrictions from the loaded
        # base program.
        lines: list[str] = []
        for raw in (text or "").splitlines():
            s = raw.lstrip()
            if not s or s.startswith('%'):
                lines.append(raw)
                continue
            # Drop materialized block_edge facts.
            if s.startswith('block_edge('):
                continue
            # Drop any constraint/rule that mentions block_edge/2.
            if 'block_edge(' in raw:
                continue
            lines.append(raw)
        return "\n".join(lines)

    def _parse_ext_fact_line(line: str) -> tuple[str, int, int, tuple[int, ...]] | None:
        """Parse an ext fact like `ext_indep(0,1,s1y2)` (optionally with trailing '.')"""
        raw = (line or "").strip()
        if not raw:
            return None
        if raw.startswith('#external'):
            raw = raw[len('#external'):].strip()
        if raw.startswith('#') or raw.startswith('%'):
            return None
        if raw.endswith('.'):
            raw = raw[:-1]

        m = re.fullmatch(r'(ext_indep|ext_dep)\((\d+)\s*,\s*(\d+)\s*,\s*([^\)\s]+)\)', raw)
        if not m:
            return None
        fact_type, x_str, y_str, s_sym = m.groups()
        x, y = int(x_str), int(y_str)

        # Convert s-sym (e.g., empty, s3, s1y2y4) to a tuple[int,...]
        if s_sym == 'empty':
            s_tuple: tuple[int, ...] = ()
        elif s_sym.startswith('s'):
            tail = s_sym[1:]
            if not tail:
                s_tuple = ()
            else:
                parts = [p for p in tail.split('y') if p]
                try:
                    s_tuple = tuple(int(p) for p in parts)
                except Exception:
                    return None
        else:
            return None

        return fact_type, x, y, s_tuple

    def _required_in_atoms_from_facts(facts_list: list[str]) -> set[str]:
        req: set[str] = set()
        for fact in facts_list or []:
            parsed = _parse_ext_fact_line(fact)
            if not parsed:
                continue
            _fact_type, _x, _y, s_tuple = parsed
            if not s_tuple:
                continue
            s_sym = 's' + 'y'.join(str(i) for i in s_tuple)
            for i in s_tuple:
                req.add(f"in({i},{s_sym}).")
        return req

    def _specific_rules_missing_required_in(specific_text: str, facts_list: list[str]) -> list[str]:
        required = _required_in_atoms_from_facts(facts_list)
        if not required:
            return []
        present = set(re.findall(r"\bin\(\d+,[a-zA-Z0-9y_]+\)\.", specific_text or ""))
        missing = sorted(required - present)
        return missing
    
    # Base program selection:
    # - default: causalaba.lp + specific rules via compile_and_ground
    # - override: a pre-emitted base program (e.g., a grounded dump from ABAPC_INC)
    if base_program_path:
        with open(str(base_program_path), 'r') as f:
            raw_base_program = f.read()

        def _sha256_text(s: str) -> str:
            try:
                return hashlib.sha256((s or "").encode("utf-8", errors="replace")).hexdigest()
            except Exception:
                return ""

        # Normalize base program dumps so they can be used as clingo CLI input.
        # In particular, ABAPC_INC may emit a *source* dump that still contains
        # `#program main(...)` blocks and uses the symbolic token `n_vars`.
        # The default path (non-override) strips '#program main' and replaces
        # `n_vars` with (n_nodes-1); do the same here for compatibility.
        base_lines: list[str] = []
        for line in (raw_base_program or "").splitlines():
            if line.strip().startswith('#program main'):
                continue
            base_lines.append(line.replace('n_vars', str(n_nodes - 1)))
        base_program = _strip_show_directives("\n".join(base_lines))
        base_program = _strip_block_edge_skeleton_reduction(base_program)

        raw_hash = _sha256_text(raw_base_program)
        norm_hash = _sha256_text(base_program)
        try:
            logging.info(
                "Loaded base program from %s (raw_sha256=%s, normalized_sha256=%s, raw_chars=%d, normalized_chars=%d)",
                str(base_program_path),
                raw_hash[:12],
                norm_hash[:12],
                len(raw_base_program or ""),
                len(base_program or ""),
            )
        except Exception:
            pass
        program_lines = [
            base_program,
            "",
            f"% ===== Base program loaded from: {base_program_path} =====",
            f"% ===== Base program sha256 (raw): {raw_hash} =====",
            f"% ===== Base program sha256 (normalized): {norm_hash} =====",
            "% ===== Assumption Layer for MUS =====",
        ]
    else:
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
                # Validate cached specific rules against the current fact set.
                missing_in = _specific_rules_missing_required_in(specific_rules_text, facts)
                if missing_in:
                    logging.warning(
                        f"Cached specific rules file is missing {len(missing_in)} required in(...) atoms; regenerating: {specific_rules_file}"
                    )
                    specific_rules_text = None
                else:
                    logging.info(f"   Loaded existing specific rules ({len(specific_rules_text)} chars)")
            except Exception as e:
                logging.warning(f"Failed to load existing specific rules: {e}")
                specific_rules_text = None

        # Generate specific rules if not loaded
        if specific_rules_text is None:
            # Parse facts to get indep_facts and dep_facts dicts
            indep_facts = {}
            dep_facts = {}

            # Build conditioning-set dictionaries from the already-parsed facts list.
            # This is more robust than re-parsing a file (which may contain directives, spacing, or be moved).
            for fact in facts or []:
                parsed = _parse_ext_fact_line(fact)
                if not parsed:
                    continue
                fact_type, x, y, s_tuple = parsed
                facts_dict = indep_facts if fact_type == 'ext_indep' else dep_facts
                facts_dict.setdefault((x, y), set()).add(s_tuple)

            # Call compile_and_ground with dump_specific (once, after collecting all facts)
            logging.debug(f"Calling compile_and_ground to dump specific rules to {specific_rules_file}")
            compile_and_ground(
                n_nodes,
                # IMPORTANT: For MUS/MCS enumeration we do not want ext_* atoms to be externals
                # (declared via the facts file). The adorned program defines them via `fact :- mus(i).`
                # so loading a facts file that contains `#external ext_*...` would create a conflict
                # (atom both external and defined) and can change semantics.
                facts_location="",
                skeleton_rules_reduction=False,
                weak_constraints=False,
                indep_facts=indep_facts,
                dep_facts=dep_facts,
                opt_mode='optN',
                out_n=1,
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
                specific_rules_text = _strip_show_directives(f.read())

            # Defensive: ensure we emitted all required in(...) atoms for the fact set.
            missing_in = _specific_rules_missing_required_in(specific_rules_text, facts)
            if missing_in:
                preview = " ".join(missing_in[:5])
                raise RuntimeError(
                    f"Generated specific rules missing {len(missing_in)} required in(...) atoms; preview: {preview}"
                )

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
        base_program = _strip_show_directives(base_program)

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

    if weights is not None:
        if len(weights) != len(facts):
            raise ValueError(
                f"weights length mismatch: got {len(weights)} weights for {len(facts)} facts"
            )
        program_lines.append("")
        program_lines.append("% ===== Objective Literals for Optimum MCS =====")
        program_lines.append("% __optimum_mcs_objective_literal__(W, mus(I)).")
        for i, w in enumerate(weights, 1):
            # WASP expects an integer weight.
            wi = int(w)
            program_lines.append(f"__optimum_mcs_objective_literal__({wi}, mus({i})).")
    
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
    optimum_mcs_algorithm: Optional[str] = None,
    facts_wc_location: str = "",
    camus_mcs_threshold: Optional[int] = None,
    camus_mus_threshold: Optional[int] = None,
    solve_timeout: Optional[float] = None,
    emit_lp: Optional[str] = None,
    timing_recorder: dict | None = None,
    base_program_path: Optional[str] = None,
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

    if optimum_mcs_algorithm:
        # Fail fast if the selected WASP build does not support optimum-MCS.
        try:
            proc = subprocess.run(
                [wasp_path, '--help'],
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except Exception as e:
            raise RuntimeError(
                f"optimum_mcs_algorithm requested but failed to probe WASP help for '{wasp_path}': {e}"
            )
        if '--optimum-mcs-algorithm' not in help_text:
            raise RuntimeError(
                "optimum_mcs_algorithm requested but this WASP build does not support --optimum-mcs-algorithm. "
                "Upgrade/rebuild WASP with optimum-MCS support, or disable optimum_mcs_algorithm."
            )
    
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
        weights: Optional[list[int]] = None
        if optimum_mcs_algorithm:
            if not facts_wc_location:
                raise ValueError(
                    "facts_wc_location is required when optimum_mcs_algorithm is enabled"
                )
            wmap = parse_weights_from_wc_file(facts_wc_location)
            missing: list[str] = []
            weights = []
            for fact in facts:
                key = _normalize_ext_fact_key(fact)
                w = wmap.get(key)
                if w is None:
                    missing.append(key)
                    weights.append(1)
                else:
                    # WASP optimum-MCS requires strictly positive weights.
                    weights.append(max(1, int(w)))
            if missing:
                preview = ", ".join(missing[:5])
                raise RuntimeError(
                    f"Missing weights for {len(missing)} facts in {facts_wc_location}; preview: {preview}"
                )

        program = build_mus_program(
            n_nodes,
            facts,
            facts_location,
            base_program_path=base_program_path,
            deadline=deadline,
            timing_recorder=timing_recorder,
            weights=weights,
        )
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
    
    # Step 3: Run MUS solver (optionally with CAMUS/MCS printing or optimum MCS)
    mus_list: List[List[int]] = []
    mcs_list: List[List[int]] = []

    solve_start = time.perf_counter()
    if print_mcses or mus_algorithm or optimum_mcs_algorithm:
        mus_mcs = run_mus_solver(
            program,
            gringo_path,
            wasp_path,
            max_muses=max_muses,
            mus_algorithm=mus_algorithm,
            print_mcses=print_mcses,
            camus_mcs_threshold=camus_mcs_threshold,
            camus_mus_threshold=camus_mus_threshold,
            optimum_mcs_algorithm=optimum_mcs_algorithm,
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


def _append_clingo_optimum_mcs_objective(program: str) -> str:
    """Append a weak-constraint objective usable by clingo.

    The program is expected to contain:
      __optimum_mcs_objective_literal__(W, mus(I)).

    We add the weak constraint that penalizes disabling assumptions:
      :~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]
    """
    lines = [program.rstrip(), ""]
    lines.append(":~ not mus(X), __optimum_mcs_objective_literal__(C, mus(X)). [C@1,X]")
    lines.append("")
    return "\n".join(lines)


def _append_show_mus(program: str) -> str:
    # For JSON parsing, keep output small and stable.
    lines = [program.rstrip(), "", "#show mus/1.", ""]
    return "\n".join(lines)


def _run_clingo_optimize_mus(
    program_str: str,
    *,
    n_facts: int,
    gringo_path: str = "clingo",
    solve_timeout: Optional[float] = None,
) -> dict:
    """Run clingo optimization on a program that shows mus/1.

    Returns a dict with:
      - selected_mus: sorted list[int]
      - costs: list[int] (clingo cost vector)
      - timed_out: bool
      - status: str (SATISFIABLE/UNSATISFIABLE/UNKNOWN)
    """
    fd, program_file = tempfile.mkstemp(suffix='_wc.lp', text=True)
    os.close(fd)
    try:
        with open(program_file, 'w') as f:
            f.write(program_str)

        cmd = [
            gringo_path,
            program_file,
            '--opt-mode=optN',
            '--outf=2',
            '-n',
            '1',
        ]

        timed_out = False
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=solve_timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                'selected_mus': [],
                'costs': [],
                'timed_out': True,
                'status': 'UNKNOWN',
            }

        stdout = proc.stdout or ""
        # clingo JSON output is a single JSON object.
        try:
            data = json.loads(stdout)
        except Exception:
            # Fall back to UNKNOWN.
            return {
                'selected_mus': [],
                'costs': [],
                'timed_out': False,
                'status': 'UNKNOWN',
            }

        status = str(data.get('Result', 'UNKNOWN'))
        calls = data.get('Call', []) or []
        witnesses = []
        costs: list[int] = []
        if calls:
            witnesses = (calls[-1].get('Witnesses', []) or [])
        if witnesses:
            w = witnesses[-1]
            vals = w.get('Value', []) or []
            costs = [int(x) for x in (w.get('Costs', []) or []) if isinstance(x, (int, float, str))]
            selected = []
            for atom in vals:
                m = re.match(r"^mus\((\d+)\)$", str(atom).strip())
                if m:
                    try:
                        idx = int(m.group(1))
                    except Exception:
                        continue
                    if 1 <= idx <= int(n_facts):
                        selected.append(idx)
            selected = sorted(set(selected))
            return {
                'selected_mus': selected,
                'costs': costs,
                'timed_out': timed_out,
                'status': status,
            }

        return {
            'selected_mus': [],
            'costs': [],
            'timed_out': False,
            'status': status,
        }
    finally:
        try:
            os.remove(program_file)
        except Exception:
            pass


def CausalABA_WC(
    n_nodes: int,
    facts_location: str,
    *,
    gringo_path: str = "clingo",
    facts_wc_location: str,
    solve_timeout: Optional[float] = None,
    emit_lp: Optional[str] = None,
    timing_recorder: dict | None = None,
    base_program_path: Optional[str] = None,
) -> dict:
    """Compute an optimum cut using clingo weak constraints only.

    This is the "pure clingo" equivalent of the emitted `*_optmcs_*_wc.lp` program:
    - Assumption layer `{mus(i)}.`
    - Facts guarded by `:-mus(i).`
    - Objective literals + weak constraint minimizing the weight of disabled mus(i)

    Returns a dict containing:
      - cut_facts: list[str] (facts removed; each includes trailing '.')
      - cut_indices: list[int]
      - cut_weight: int
      - selected_mus: list[int]
      - timed_out: bool
      - solve_time: float
    """
    if not facts_location:
        raise ValueError("facts_location is required")
    if not facts_wc_location:
        raise ValueError("facts_wc_location is required")

    facts, fact_mapping = parse_facts_from_file(facts_location)
    if not facts:
        return {
            'cut_indices': [],
            'cut_facts': [],
            'cut_weight': 0,
            'selected_mus': [],
            'timed_out': False,
            'solve_time': 0.0,
            'fact_mapping': fact_mapping,
        }

    wmap = parse_weights_from_wc_file(facts_wc_location)
    weights: list[int] = []
    missing: list[str] = []
    for fact in facts:
        key = _normalize_ext_fact_key(fact)
        w = wmap.get(key)
        if w is None:
            missing.append(key)
            weights.append(1)
        else:
            weights.append(max(1, int(w)))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"Missing weights for {len(missing)} facts in {facts_wc_location}; preview: {preview}"
        )

    # Treat solve_timeout as an end-to-end budget (build + solve)
    deadline = (time.perf_counter() + float(solve_timeout)) if solve_timeout is not None else None
    build_start = time.perf_counter()
    program = build_mus_program(
        n_nodes,
        facts,
        facts_location,
        base_program_path=base_program_path,
        deadline=deadline,
        timing_recorder=timing_recorder,
        weights=weights,
    )
    program = _append_clingo_optimum_mcs_objective(program)
    program = _append_show_mus(program)

    if emit_lp:
        with open(emit_lp, 'w') as f:
            f.write(program)
        logging.info(f"Emitted complete WC optimization program to {emit_lp}")

    build_time = time.perf_counter() - build_start
    remaining_timeout: Optional[float] = None
    if deadline is not None:
        remaining_timeout = max(0.0, deadline - time.perf_counter())

    solve_start = time.perf_counter()
    out = _run_clingo_optimize_mus(
        program,
        n_facts=len(facts),
        gringo_path=gringo_path,
        solve_timeout=remaining_timeout,
    )
    solve_time = time.perf_counter() - solve_start

    selected_mus = out.get('selected_mus', []) or []
    selected_set = set(int(x) for x in selected_mus)
    cut_indices = [i for i in range(1, len(facts) + 1) if i not in selected_set]
    cut_facts = [fact_mapping.get(i, f"mus({i}).") for i in cut_indices]
    cut_weight = int(sum(weights[i - 1] for i in cut_indices)) if cut_indices else 0

    return {
        'cut_indices': cut_indices,
        'cut_facts': cut_facts,
        'cut_weight': cut_weight,
        'selected_mus': selected_mus,
        'timed_out': bool(out.get('timed_out', False)),
        'status': out.get('status', 'UNKNOWN'),
        'costs': out.get('costs', []),
        'wc_build_time': float(build_time),
        'solve_time': float(solve_time),
        'total_time': float(build_time + solve_time),
        'fact_mapping': fact_mapping,
    }
