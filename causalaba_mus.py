"""MUS (Minimal Unsatisfiable Subset) analysis for CausalABA

This module computes MUS using the full CausalABA encoding with active paths
and d-separation. MUS identifies minimal subsets of independence/dependence facts
that create unsatisfiability.

Copyright 2025 Fabrizio Russo, Department of Computing, Imperial College London

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

__author__ = "Fabrizio Russo"
__email__ = "fabrizio@imperial.ac.uk"
__copyright__ = "Copyright (c) 2025 Fabrizio Russo"

import os
import sys
import logging
import tempfile
import subprocess
import re
from pathlib import Path
from itertools import combinations
from causalaba import compile_and_ground


def parse_facts_from_file(facts_location: str) -> tuple[list[str], dict[int, str]]:
    """
    Parse independence and dependence facts from a file.
    
    Recognizes patterns:
    - ext_indep(X,Y,S).
    - ext_dep(X,Y,S).
    
    Args:
        facts_location: Path to the facts file
    
    Returns:
        Tuple of (list of fact strings, dict mapping indices to fact strings)
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
    
    logging.info(f"Parsed {len(facts)} facts from {facts_location}")
    return facts, fact_mapping


def run_mus_solver(program_str: str, gringo_path: str = "clingo", 
                   wasp_path: str = "wasp") -> list:
    """
    Run MUS solver on ASP program using gringo and wasp.
    
    Args:
        program_str: Complete ASP program as string
        gringo_path: Path to clingo executable
        wasp_path: Path to wasp executable
    
    Returns:
        List of MUS, each MUS is a list of assumption indices (a(i) atoms)
    """
    logging.info("Running MUS solver...")
    
    try:
        # Write program to temporary file
        fd, program_file = tempfile.mkstemp(suffix='_mus.lp', text=True)
        os.close(fd)
        with open(program_file, 'w') as f:
            f.write(program_str)
        
        # Build clingo command to ground with smodels output
        clingo_cmd = [gringo_path, program_file, '--output=smodels']
        
        # Build wasp command to compute MUS over mus/1
        wasp_cmd = [wasp_path, '--mus=mus', '-n', '0']
        
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
        clingo_process.stdout.close()
        
        # Wait for wasp to complete
        wasp_output, wasp_error = wasp_process.communicate()
        
        # Wait for clingo to complete and get its stderr
        clingo_process.wait()
        clingo_error = clingo_process.stderr.read()
        clingo_process.stderr.close()
        
        if clingo_process.returncode != 0:
            logging.error(f"Clingo failed with return code {clingo_process.returncode}: {clingo_error}")
            return []
        elif clingo_error.strip():
            # Clingo succeeded but had warnings/info messages
            logging.debug(f"Clingo stderr (informational): {clingo_error}")
        
        if wasp_process.returncode != 0:
            logging.error(f"Wasp failed with return code {wasp_process.returncode}: {wasp_error}")
            return []
        
        logging.debug(f"Wasp output:\n{wasp_output}")
        
        # Parse MUS from output
        mus_list = []
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
        
        logging.info(f"Total MUS found: {len(mus_list)}")
        
        # Cleanup
        try:
            os.remove(program_file)
        except:
            pass
        
        return mus_list
        
    except FileNotFoundError as e:
        logging.error(f"Error running MUS solver: {e}")
        logging.error("Make sure clingo and wasp are installed and in your PATH")
        return []
    except Exception as e:
        logging.error(f"Unexpected error running MUS solver: {e}")
        return []


def build_mus_program(n_nodes: int, facts: list[str], facts_location: str = "") -> str:
    """
    Build an ASP program for MUS computation using CausalABA's compile_and_ground.
    
    This function:
    1. Calls compile_and_ground with dump_specific to get the generated specific rules
    2. Loads the base causalaba.lp encoding
    3. Combines with adorned facts and MUS assumption layer
    
    Args:
        n_nodes: Number of nodes in the causal graph
        facts: List of fact strings (without period, e.g., "ext_indep(1,2,s0)")
        facts_location: Path to facts file (used by compile_and_ground to extract indep/dep facts)
    
    Returns:
        Ungrounded ASP program text (will be grounded by run_mus_solver)
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
                skeleton_rules_reduction=False,
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
                dump_specific=specific_rules_file
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
    
    # Add adorned facts: fact :- mus(i)
    for i, fact in enumerate(facts, 1):
        program_lines.append(f"{fact}:-mus({i}).")
    
    program = '\n'.join(program_lines)
    logging.debug(f"Built complete MUS program: {len(program)} chars")
    
    return program


def CausalABA_MUS(n_nodes: int, facts_location: str = "",
                  gringo_path: str = "clingo", wasp_path: str = "wasp") -> dict:
    """
    Run CausalABA with MUS (Minimal Unsatisfiable Subset) analysis.
    
    This function builds an ASP program that includes the full CausalABA
    encoding with active paths, then uses wasp to compute MUS over the assumption
    atoms mus(i), which correspond directly to the input facts.
    
    Args:
        n_nodes: Number of nodes in the causal graph
        facts_location: Path to the facts file containing ext_indep/ext_dep statements
        gringo_path: Path to clingo executable (default: "clingo")
        wasp_path: Path to wasp executable
    
    Returns:
        Dictionary with:
            - 'mus_list': List of MUS (each MUS contains fact indices 1, 2, ...)
            - 'mus_facts': List of MUS with actual fact strings
            - 'n_mus': Number of MUS found
            - 'fact_mapping': Mapping from fact indices to fact strings
    """
    logging.info("Running CausalABA MUS analysis")
    
    if not facts_location:
        logging.error("facts_location is required for MUS analysis")
        return {'mus_list': [], 'mus_facts': [], 'n_mus': 0, 'fact_mapping': {}}
    
    # Step 1: Parse facts from file
    facts, fact_mapping = parse_facts_from_file(facts_location)
    
    if not facts:
        logging.error(f"No ext_indep/ext_dep facts found in {facts_location}")
        return {'mus_list': [], 'mus_facts': [], 'n_mus': 0, 'fact_mapping': {}}
    
    logging.info(f"Found {len(facts)} facts to analyze for MUS")
    if len(fact_mapping) <= 20:
        for idx, fact in fact_mapping.items():
            logging.debug(f"  Fact {idx}: {fact}")
    else:
        preview = [fact_mapping[i] for i in sorted(fact_mapping.keys())[:5]]
        logging.debug(f"  Preview facts: {preview} (showing 5 of {len(fact_mapping)})")
    
    # Step 2: Build MUS program with assumption layer on top of CausalABA encoding
    program = build_mus_program(n_nodes, facts, facts_location)
    
    # Step 3: Run MUS solver
    mus_list = run_mus_solver(program, gringo_path, wasp_path)
    
    # Step 4: Map MUS indices back to actual facts
    mus_facts = []
    for mus in mus_list:
        mus_fact_list = [fact_mapping.get(idx, f"mus({idx})") for idx in mus]
        mus_facts.append(mus_fact_list)
    
    if mus_list:
        sizes = [len(m) for m in mus_list]
        logging.info(
            f"MUS facts resolved: {len(mus_list)} cores (min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f})"
        )
        logging.debug(f"MUS facts (resolved): {mus_facts}")
    else:
        logging.info("MUS facts (resolved): []")
    
    result = {
        'mus_list': mus_list,
        'mus_facts': mus_facts,
        'n_mus': len(mus_list),
        'fact_mapping': fact_mapping
    }
    
    return result
