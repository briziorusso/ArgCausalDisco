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
from collections import Counter
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

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

    def test_mus_links_wrong_tests_four_node_abapc(self):
        """Follow test_abapc_four_node_example and append MUS analysis.

        We reuse the exact fact construction from test_abapc_four_node_example:
        facts come from PC output versus ground truth (no synthetic contradictions),
        then we run:
        - CausalABA with all facts (expect UNSAT or at least allow removal),
        - CausalABA with search_for_models='first' (removal strategy),
        - MUS over the fact set plus the ground-truth counterparts of wrong facts
          to highlight which tests are inconsistent.
        """
        logger_setup()
        logging.info("===============Running test_mus_links_wrong_tests_four_node_abapc===============")

        import networkx as nx
        import numpy as np
        import types
        # Stub notears to avoid heavy optional dependency required by cd_algorithms.models
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')
            class _DummyMLP:
                pass
            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")
            notears_nonlinear_module.NotearsMLP = _DummyMLP
            notears_nonlinear_module.notears_nonlinear = _dummy_notears_nonlinear
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module
        from utils.graph_utils import find_all_d_separations_sets, extract_test_elements_from_symbol, initial_strength
        from utils.data_utils import simulate_data_and_run_PC
        from utils.helpers import random_stability

        # ArgCD four-node DAG (same structure and seed as test_abapc_four_node_example)
        alpha = 0.05
        seed = 2376
        B_true = np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
            ]
        )
        n_nodes = B_true.shape[0]
        G_true = nx.DiGraph(
            pd.DataFrame(
                B_true,
                columns=[f"X{i+1}" for i in range(B_true.shape[1])],
                index=[f"X{i+1}" for i in range(B_true.shape[1])],
            )
        )
        relabel_dict = {f"X{i+1}": i for i in range(n_nodes)}
        G_true1 = nx.relabel_nodes(G_true, relabel_dict)

        true_seplist = find_all_d_separations_sets(G_true)

        random_stability(seed)
        data, cg = simulate_data_and_run_PC(G_true, alpha, seed=seed, uc_rule=5, stable=True)

        facts = []  # (fact_str, I, is_correct)
        facts_ext = []
        wrong_ext = []
        count_wrong = 0

        for test in true_seplist:
            X, S, Y, dep_type = extract_test_elements_from_symbol(test)
            test_PC = set([t for t in cg.sepset[X, Y] if set(t[0]) == S])
            if len(test_PC) != 1:
                continue
            p = list(test_PC)[0][1]
            dep_type_PC = "indep" if p > alpha else "dep"
            I = initial_strength(p, len(S), alpha, 0.5, n_nodes)
            if dep_type == dep_type_PC:
                fact_str = test
                is_correct = True
            elif dep_type == "indep":
                count_wrong += 1
                fact_str = test.replace("indep", "dep")
                is_correct = False
            else:  # dep
                count_wrong += 1
                fact_str = test.replace("dep", "indep")
                is_correct = False

            facts.append((fact_str, I, is_correct))
            ext_line = f"ext_{fact_str}"
            facts_ext.append(ext_line)
            if not is_correct:
                wrong_ext.append(ext_line)

        logging.info(f"Seed: {seed}")
        logging.info(f"True DAG: {G_true1.edges}")
        logging.info(f"Number of total independence statements: {len(true_seplist)}")
        logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
        logging.info(f"Number of wrong facts: {count_wrong} ({(count_wrong/len(facts))*100 if facts else 0:.2f}%)")
        logging.info(f"Fully directed edges from PC: {cg.find_fully_directed()}")
        logging.info(f"Undirected edges from PC: {[(x,y) for (x,y) in cg.find_undirected() if x < y]}")

        # Write facts to temp files (base + I + wc) for CausalABA
        fd, facts_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd)
        facts_I_file = facts_file.replace('.lp', '_I.lp')
        facts_wc_file = facts_file.replace('.lp', '_wc.lp')

        with open(facts_file, 'w') as f:
            for s in facts_ext:
                line = s if s.endswith('.') else s + '.'
                f.write(f"#external {line}\n")

        with open(facts_I_file, 'w') as f:
            for fact, s in zip(facts, facts_ext):
                line = s if s.endswith('.') else s + '.'
                I = fact[1]
                f.write(f"{line} I={I}, NA\n")

        with open(facts_wc_file, 'w') as f:
            for fact, s in zip(facts, facts_ext):
                line = s if s.endswith('.') else s + '.'
                I = fact[1]
                f.write(f":~ {line} [-{int(I*1e14)*2}]\n")

        # Step 1: Run with all facts (may be UNSAT). If UNSAT, removal should fix it.
        logging.info("Step 1: Testing with all facts")
        models_all, _ = CausalABA(n_nodes, facts_file, weak_constraints=True, print_models=False)
        logging.info(f"  → {len(models_all)} models")

        # Step 2: Apply removal strategy to reach SAT (or keep SAT)
        logging.info("Step 2: Applying removal strategy (search_for_models='first')")
        models_after, multiple, stats, remove_n = CausalABA(
            n_nodes,
            facts_file,
            weak_constraints=True,
            search_for_models='first',
            print_models=False,
            return_statistics=True,
        )
        logging.info(f"  → Facts removed: {remove_n}")
        logging.info(f"  → Models found after removal: {len(models_after)}")

        self.assertGreaterEqual(len(models_after), 0)
        if len(models_all) == 0:
            self.assertGreater(remove_n, 0, "Expected removal of X>0 tests to reach SAT when UNSAT")
            self.assertGreater(len(models_after), 0, "Expected SAT after removing X tests")

        # Step 3: Run MUS on PC facts
        logging.info("Step 3: Running MUS analysis")
        fd_mus, facts_mus_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd_mus)
        with open(facts_mus_file, 'w') as f:
            for s in facts_ext:
                line = s if s.endswith('.') else s + '.'
                f.write(f"{line}\n")

        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_mus_file,
            gringo_path="clingo",
            wasp_path="/vol/bitbucket/fr920/wasp/build/release/wasp",
            mus_algorithm="camus",
            print_mcses=True,
        )

        logging.info(f"MUS cores found: {mus_result['n_mus']}")
        mus_sets = [set(mf) for mf in mus_result['mus_facts']]
        if mus_sets:
            sizes = [len(ms) for ms in mus_sets]
            fact_freq = Counter(f for ms in mus_sets for f in ms)
            common_facts = set.intersection(*mus_sets)
            common_preview = sorted(common_facts)[:10]
            if len(common_facts) > 10:
                common_preview.append(f"... (+{len(common_facts) - 10} more)")
            top_common = ', '.join(f"{f} ({c}/{mus_result['n_mus']})" for f, c in fact_freq.most_common(5))
            logging.info(
                f"  MUS sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}"
            )
            logging.info(
                f"  Facts present in all MUS ({len(common_facts)}): {', '.join(common_preview) if common_preview else '(none)'}"
            )
            logging.info(f"  Top frequent facts: {top_common if top_common else '(none)'}")
            
            # Analyze wrong vs correct fact frequencies in MUS
            wrong_set = set(w + '.' if not w.endswith('.') else w for w in wrong_ext)
            correct_facts = [f for f in fact_freq if f not in wrong_set]
            wrong_facts = [f for f in fact_freq if f in wrong_set]
            
            logging.info(f"  Wrong facts in MUS: {len(wrong_facts)} unique, correct facts: {len(correct_facts)} unique")
            
            if wrong_facts:
                wrong_freq = [(f, fact_freq[f], fact_freq[f]/mus_result['n_mus']*100) for f in wrong_facts]
                wrong_freq.sort(key=lambda x: x[1], reverse=True)
                top_wrong = ', '.join(f"{f} ({c}/{mus_result['n_mus']} = {p:.1f}%)" for f, c, p in wrong_freq[:5])
                logging.info(f"  Top wrong facts in MUS: {top_wrong}")
                
                # Show average frequency
                avg_wrong_freq = sum(fact_freq[f] for f in wrong_facts) / len(wrong_facts)
                avg_correct_freq = sum(fact_freq[f] for f in correct_facts) / len(correct_facts) if correct_facts else 0
                logging.info(f"  Avg appearances: wrong facts {avg_wrong_freq:.1f}, correct facts {avg_correct_freq:.1f}")

            # Ordered ranking of facts by MUS frequency, explicitly labeled WRONG/CORRECT
            n_mus = max(int(mus_result['n_mus'] or 0), 1)
            ordered = [(f, fact_freq[f], (f in wrong_set)) for f in fact_freq]
            ordered.sort(key=lambda t: (-t[1], t[0]))
            logging.info("  Fact ranking by MUS frequency (count / n_mus):")
            for fact, count, is_wrong in ordered[: min(30, len(ordered))]:
                label = "WRONG" if is_wrong else "CORRECT"
                pct = (count / n_mus) * 100.0
                logging.info(f"    - {label}: {fact} ({count}/{n_mus} = {pct:.1f}%)")

            # Print 3 smallest MUSes as examples
            sorted_muses = sorted(enumerate(mus_result['mus_facts'], 1), key=lambda x: (len(x[1]), x[0]))
            logging.info("  Examples of smallest MUSes:")
            for idx, mus_facts in sorted_muses[:3]:
                logging.info(f"    MUS #{idx} (size {len(mus_facts)}):")
                for fact in mus_facts:
                    label = "WRONG" if fact in wrong_set else "CORRECT"
                    logging.info(f"      - {label}: {fact}")
        else:
            logging.info("  No MUS cores found")

        # --------------------------
        # MCS analysis (CAMUS)
        # --------------------------
        logging.info(f"MCSes found: {mus_result.get('n_mcs', 0)}")
        mcs_sets = [set(mf) for mf in mus_result.get('mcs_facts', [])]
        if mcs_sets:
            # Only analyze PC-derived facts (filter out ground-truth counterparts)
            pc_facts_set = set(s + '.' if not s.endswith('.') else s for s in facts_ext)
            wrong_set = set(w + '.' if not w.endswith('.') else w for w in wrong_ext)
            
            # Filter MCS to only include PC facts
            mcs_sets_pc_only = [ms.intersection(pc_facts_set) for ms in mcs_sets]
            fact_freq = Counter(f for ms in mcs_sets_pc_only for f in ms)
            
            # Compute stats on PC facts only
            sizes = [len(ms) for ms in mcs_sets_pc_only]
            common_facts = set.intersection(*mcs_sets_pc_only) if mcs_sets_pc_only else set()
            common_preview = sorted(common_facts)[:10]
            if len(common_facts) > 10:
                common_preview.append(f"... (+{len(common_facts) - 10} more)")
            top_common = ', '.join(f"{f} ({fact_freq[f]}/{mus_result.get('n_mcs', 0)})" for f in sorted(fact_freq, key=lambda x: -fact_freq[x])[:5])
            
            logging.info(
                f"  MCS sizes (PC facts only): min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}"
            )
            logging.info(
                f"  PC facts present in all MCSes ({len(common_facts)}): {', '.join(common_preview) if common_preview else '(none)'}"
            )
            logging.info(f"  Top frequent PC facts: {top_common if top_common else '(none)'}")

            # Analyze wrong vs correct fact frequencies in MCS (PC facts only)
            correct_facts = [f for f in fact_freq if f not in wrong_set]
            wrong_facts = [f for f in fact_freq if f in wrong_set]
            logging.info(f"  Wrong PC facts in MCS: {len(wrong_facts)} unique, correct PC facts: {len(correct_facts)} unique")

            if wrong_facts and mus_result.get('n_mcs', 0) > 0:
                wrong_freq = [(f, fact_freq[f], fact_freq[f]/mus_result['n_mcs']*100) for f in wrong_facts]
                wrong_freq.sort(key=lambda x: x[1], reverse=True)
                top_wrong = ', '.join(f"{f} ({c}/{mus_result['n_mcs']} = {p:.1f}%)" for f, c, p in wrong_freq[:5])
                logging.info(f"  Top wrong PC facts in MCS: {top_wrong}")

            # Ordered ranking of PC facts by MCS frequency, explicitly labeled WRONG/CORRECT
            n_mcs = max(int(mus_result.get('n_mcs', 0) or 0), 1)
            ordered = [(f, fact_freq[f], (f in wrong_set)) for f in fact_freq]
            ordered.sort(key=lambda t: (-t[1], t[0]))
            logging.info("  PC Fact ranking by MCS frequency (count / n_mcs):")
            for fact, count, is_wrong in ordered[: min(30, len(ordered))]:
                label = "WRONG" if is_wrong else "CORRECT"
                pct = (count / n_mcs) * 100.0
                logging.info(f"    - {label}: {fact} ({count}/{n_mcs} = {pct:.1f}%)")

            # Print 3 smallest MCSes as examples
            sorted_mcses = sorted(enumerate(mus_result.get('mcs_facts', []), 1), key=lambda x: (len(x[1]), x[0]))
            logging.info("  Examples of smallest MCSes:")
            for idx, mcs_facts in sorted_mcses[:3]:
                logging.info(f"    MCS #{idx} (size {len(mcs_facts)}):")
                for fact in mcs_facts:
                    label = "WRONG" if fact in wrong_set else "CORRECT"
                    logging.info(f"      - {label}: {fact}")
        else:
            logging.info("  No MCSes found")

        wrong_set = set(w + '.' if not w.endswith('.') else w for w in wrong_ext)
        if mus_result['n_mus'] > 0:
            for ms in mus_sets:
                self.assertTrue(len(ms.intersection(wrong_set)) >= 1, "Each MUS should include at least one wrong/flipped test")

        os.remove(facts_file)
        os.remove(facts_I_file)
        os.remove(facts_wc_file)
        os.remove(facts_mus_file)

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
            wasp_path="/vol/bitbucket/fr920/wasp/build/release/wasp",
            mus_algorithm="camus",
            print_mcses=True,
        )
        
        logging.info(f"  MUS cores found: {mus_result['n_mus']}")
        for i, mus_facts in enumerate(mus_result['mus_facts']):
            logging.info(f"    MUS #{i+1}: {len(mus_facts)} facts")
            for fact in mus_facts:
                logging.info(f"      - {fact}")
        
        self.assertEqual(mus_result['n_mus'], 1, "Expected exactly one MUS")
        self.assertEqual(len(mus_result['mus_facts'][0]), 3, "Expected all three facts in the MUS")

        # With CAMUS + --print-mcses, we should also see the 3 singleton MCSes.
        self.assertEqual(mus_result.get('n_mcs', 0), 3, "Expected three singleton MCSes")
        mcs_sets = [set(m) for m in mus_result.get('mcs_list', [])]
        self.assertIn({1}, mcs_sets)
        self.assertIn({2}, mcs_sets)
        self.assertIn({3}, mcs_sets)

        logging.info(f"  MCSes found: {mus_result.get('n_mcs', 0)}")
        for i, mcs_facts in enumerate(mus_result.get('mcs_facts', [])):
            logging.info(f"    MCS #{i+1}: {len(mcs_facts)} facts")
            for fact in mcs_facts:
                logging.info(f"      - {fact}")
        
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
            wasp_path="/vol/bitbucket/fr920/wasp/build/release/wasp",
            mus_algorithm="camus",
            print_mcses=True,
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

        # Additional section: print MCSes (more actionable than just MUS)
        logging.info(f"MCSes found: {mus_result.get('n_mcs', 0)}")
        for i, mcs_facts in enumerate(mus_result.get('mcs_facts', [])):
            logging.info(f"  MCS #{i+1}: {len(mcs_facts)} facts")
            for fact in mcs_facts:
                logging.info(f"    - {fact}")

        os.remove(facts_file)


if __name__ == '__main__':
    start = datetime.now()
    logger_setup()
    unittest.main(verbosity=2)
    logging.info(f"Total test time={str(datetime.now()-start)}")
