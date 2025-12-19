"""MUS (Minimal Unsatisfiable Subset) tests for CausalABA.

This module tests MUS functionality using the full CausalABA encoding with
active paths and d-separation semantics. MUS identifies minimal subsets of
independence/dependence facts that create unsatisfiability when combined with
the causal graph constraints. Facts are guarded by mus/1 assumption atoms to
enable WASP to compute minimal cores.

Copyright 2025 Fabrizio Russo, Department of Computing, Imperial College London
Licensed under the Apache License, Version 2.0
"""

import os
import sys
import logging
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
if not hasattr(pd, '__version__'):
    pd.__version__ = '2.2.3'

from causalaba import CausalABA
from causalaba_mus import CausalABA_MUS


def logger_setup(scenario="test_mus"):
    """Setup logging for tests."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        stream=sys.stdout,
        force=True,
    )


class TestMUSAnalysis(unittest.TestCase):
    """Tests for MUS analysis of CausalABA using the mus/1 assumption framework.
    
    The mus/1 assumption layer allows WASP to compute minimal unsatisfiable
    subsets of facts that violate causal graph constraints. Facts are guarded
    by choice rules {mus(i)}. and conditionally activated.
    """

    def test_parsing_facts_from_file(self):
        """Test parsing ext_indep/ext_dep facts from a file."""
        logger_setup()
        logging.info("===============Running test_parsing_facts_from_file===============")
        
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        
        with open(facts_file, 'w') as f:
            f.write("% Comment line\n")
            f.write("ext_indep(0,1,empty).\n")
            f.write("ext_dep(1,2,s0).\n")
            f.write("ext_indep(2,0,s1).\n")
            f.write("% Another comment\n")
            f.write("#show something.\n")
        
        from causalaba_mus import parse_facts_from_file
        facts, fact_mapping = parse_facts_from_file(facts_file)
        
        logging.info(f"Parsed {len(facts)} facts:")
        for idx, fact_str in fact_mapping.items():
            logging.info(f"  Fact {idx}: {fact_str}")
        
        self.assertEqual(len(facts), 3, "Expected 3 facts parsed")
        self.assertIn("ext_indep(0,1,empty).", list(fact_mapping.values())[0])
        
        os.remove(facts_file)

    def test_adorning_with_mus_assumptions(self):
        """Test that facts are correctly adorned with mus/1 assumptions.
        
        The adorning process wraps each fact in a conditional rule guarded by
        a mus(i) assumption, allowing WASP to compute MUS over the assumptions.
        
        This test verifies:
        1. mus(i) choice rules are generated
        2. Facts are guarded by mus(i) atoms
        3. No #show statements pollute the output
        """
        logger_setup()
        logging.info("===============Running test_adorning_with_mus_assumptions===============")
        
        from causalaba_mus import build_mus_program
        
        n_nodes = 3
        facts = ["ext_indep(1,2,s0)", "ext_dep(1,2,empty)"]
        
        program = build_mus_program(n_nodes, facts)
        logging.info(f"Generated program with {len(program.split(chr(10)))} lines")
        
        # Verify mus choice rules exist (allow varying whitespace)
        self.assertIn("{mus(1)}", program, "Expected mus(1) choice rule")
        self.assertIn("{mus(2)}", program, "Expected mus(2) choice rule")
        
        # Verify facts are guarded by mus(i) (allow varying whitespace)
        self.assertRegex(program, r'ext_indep\s*\(\s*1\s*,\s*2\s*,\s*s0\s*\)\s*:-\s*mus\s*\(\s*1\s*\)', "Expected fact 1 guarded by mus(1)")
        self.assertRegex(program, r'ext_dep\s*\(\s*1\s*,\s*2\s*,\s*empty\s*\)\s*:-\s*mus\s*\(\s*2\s*\)', "Expected fact 2 guarded by mus(2)")
        
        # Verify no #show statements (would interfere with MUS output)
        self.assertNotIn("#show", program, "Program should not contain #show directives")
        
        logging.info("✓ Adorning structure verified")

    def test_mock_three_var_manual_vs_mus(self):
        """Test mock_three_var: manually removing facts vs MUS finding them all.
        
        This test verifies that:
        1. All three facts together cause UNSAT
        2. Removing any single fact causes SAT (with different models)
        3. MUS discovers all three facts as a single minimal core
        
        This directly links the manual removal behavior to the MUS algorithm.
        """
        logger_setup()
        logging.info("===============Running test_mock_three_var_manual_vs_mus===============")
        
        n_nodes = 3
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        
        with open(facts_file, 'w') as f:
            f.write("ext_indep(1,2,s0).\n")
            f.write("ext_dep(1,2,empty).\n")
            f.write("ext_indep(0,1,empty).\n")
        
        logging.info("Mock-three-var facts:")
        logging.info("  1. ext_indep(1,2,s0)")
        logging.info("  2. ext_dep(1,2,empty)")
        logging.info("  3. ext_indep(0,1,empty)")
        logging.info("")
        
        # Step 1: Verify UNSAT with all facts
        logging.info("Step 1: Testing with all three facts")
        models, _ = CausalABA(n_nodes, facts_file, print_models=False)
        logging.info(f"  → {len(models)} models (expected 0 for UNSAT)")
        self.assertEqual(len(models), 0, "Expected UNSAT with all three facts")
        logging.info("")
        
        # Step 2: Verify SAT after removing each fact individually
        logging.info("Step 2: Testing with each fact removed individually")
        for removed_fact_num in [1, 2, 3]:
            fd_temp, facts_file_removed = tempfile.mkstemp(suffix='.lp', text=True)
            os.close(fd_temp)
            
            with open(facts_file_removed, 'w') as f:
                if removed_fact_num != 1:
                    f.write("ext_indep(1,2,s0).\n")
                if removed_fact_num != 2:
                    f.write("ext_dep(1,2,empty).\n")
                if removed_fact_num != 3:
                    f.write("ext_indep(0,1,empty).\n")
            
            models_removed, _ = CausalABA(n_nodes, facts_file_removed, print_models=False)
            logging.info(f"  Removed fact {removed_fact_num}: {len(models_removed)} models")
            if len(models_removed) > 0:
                first_model = models_removed[0]
                arrow_syms = sorted([str(s) for s in first_model])
                logging.info(f"    First model arrows: {', '.join(arrow_syms) if arrow_syms else '(none)'}")
            self.assertGreater(len(models_removed), 0, f"Expected SAT after removing fact {removed_fact_num}")
            
            os.remove(facts_file_removed)
        
        logging.info("")
        
        # Step 3: Compute MUS - should find all three facts
        logging.info("Step 3: Running MUS analysis")
        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_file,
            gringo_path="clingo",
            wasp_path="/vol/bitbucket/fr920/wasp/build/release/wasp"
        )
        
        logging.info(f"  MUS cores found: {mus_result['n_mus']}")
        for i, mus_facts in enumerate(mus_result['mus_facts']):
            logging.info(f"    MUS #{i+1}: {len(mus_facts)} facts")
            for fact in mus_facts:
                logging.info(f"      - {fact}")
        
        self.assertEqual(mus_result['n_mus'], 1, "Expected exactly one MUS")
        self.assertEqual(len(mus_result['mus_facts'][0]), 3, "Expected all three facts in the MUS")
        
        os.remove(facts_file)

    def test_mus_catches_wrong_facts_on_four_nodes(self):
        """Link wrong tests to MUSes on a larger 4-node case.

        We craft two explicit contradictions:
        - ext_indep(0,1,empty) vs ext_dep(0,1,empty)
        - ext_indep(2,3,empty) vs ext_dep(2,3,empty)

        Each pair is independently inconsistent due to the base constraint
        ":- dep(X,Y,S), indep(X,Y,S), ...". MUS should return two minimal
        cores, each containing the two conflicting facts. This demonstrates
        how MUS pinpoints wrong tests in bigger problems.
        """
        logger_setup()
        logging.info("===============Running test_mus_catches_wrong_facts_on_four_nodes===============")

        n_nodes = 4
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)

        with open(facts_file, 'w') as f:
            # Contradiction A (pair 0-1)
            f.write("ext_indep(0,1,empty).\n")
            f.write("ext_dep(0,1,empty).\n")
            # Contradiction B (pair 2-3)
            f.write("ext_indep(2,3,empty).\n")
            f.write("ext_dep(2,3,empty).\n")
            # Some additional non-contradictory facts to make the instance bigger
            f.write("ext_dep(0,2,empty).\n")
            f.write("ext_indep(1,2,s0).\n")

        # Run MUS analysis
        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_file,
            gringo_path="clingo",
            wasp_path="/vol/bitbucket/fr920/wasp/build/release/wasp"
        )

        logging.info(f"MUS cores found: {mus_result['n_mus']}")
        for i, mus_facts in enumerate(mus_result['mus_facts']):
            logging.info(f"  MUS #{i+1}: {len(mus_facts)} facts")
            for fact in mus_facts:
                logging.info(f"    - {fact}")

        # Expect at least two MUSes (one for each contradictory pair)
        self.assertGreaterEqual(mus_result['n_mus'], 2, "Expected at least two MUS cores")

        # Define the two wrong test sets
        wrong_A = {"ext_indep(0,1,empty).", "ext_dep(0,1,empty)."}
        wrong_B = {"ext_indep(2,3,empty).", "ext_dep(2,3,empty)."}

        # Check that each wrong set appears as a MUS of size 2
        mus_sets = [set(mf) for mf in mus_result['mus_facts']]
        has_A = any(ms == wrong_A for ms in mus_sets)
        has_B = any(ms == wrong_B for ms in mus_sets)
        self.assertTrue(has_A, "Contradictory pair (0,1) should be identified as a MUS")
        self.assertTrue(has_B, "Contradictory pair (2,3) should be identified as a MUS")

        os.remove(facts_file)


if __name__ == '__main__':
    start = datetime.now()
    logger_setup()
    unittest.main(verbosity=2)
    logging.info(f"Total test time={str(datetime.now()-start)}")
