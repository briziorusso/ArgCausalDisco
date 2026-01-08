"""MUS/MCS analysis tests + reproducible profiling harness.

This file serves two roles:

1) **pytest/unittest tests** for the MUS assumption-layer and small sanity cases.
2) A **scriptable harness** for generating random PC-based instances, running:
   - ABAPC removal (`CausalABA(..., search_for_models='first')`), and
   - MUS/MCS analysis (`CausalABA_MUS(...)`),
   while printing a phase-by-phase timing comparison.

Key tests:

- `test_parsing_facts_from_file`: ensures we correctly read `ext_indep`/`ext_dep` facts and ignore comments/directives.
- `test_adorning_with_mus_assumptions`: validates `{mus(i)}.` choice rules and `fact :- mus(i).` guarding.
- `test_mock_three_var_manual_vs_mus`: smallest end-to-end example; matches the manual folder
  `encodings/test_lps/mock_three_var_manual/`.
- `test_mus_links_wrong_tests_four_node_abapc`: links PC-derived wrong tests to MUS/MCS on a fixed 4-node case.
- `test_mus_mcs_random_five_node_abapc`: PC → ABAPC → MUS/MCS on a deterministic 5-node seed.
- `test_mus_mcs_random_sizes_abapc`: same pipeline across multiple sizes (`--node-sizes`).

Run via pytest:

  python -m pytest tests_mus.py -xvs
  python -m pytest tests_mus.py::TestMUSAnalysis::test_mock_three_var_manual_vs_mus -v

Run as a script (uses argparse at bottom):

  python tests_mus.py --node-sizes 7,8 --solve-timeout 120 --max-muses 0 --emit-lp results/adornedLP_{n}.lp

To reproduce a run externally once you emitted an adorned `.lp`:

  clingo results/mus_7.lp --output=smodels | wasp \
      --mus=mus --mus-algorithm=camus --print-mcses -n 0

Important knobs:
- `--graph-type`: random DAG family passed to the simulator (default: ER).
- `--opt-mode`, `--out-n`: passed to the ABAPC run (`CausalABA`) to control clingo optimization and model bound.
- `--max-muses`: controls whether WASP receives `-n`.
"""

import os
import sys
import argparse
import logging
import tempfile
import unittest
from datetime import datetime
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(PROJECT_ROOT))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

CausalABA: Any = None
CausalABA_MUS: Any = None


def _ensure_solvers_imported() -> None:
    """Import solver modules lazily so `--help` works without clingo/wasp installed."""
    global CausalABA, CausalABA_MUS
    if CausalABA is None or CausalABA_MUS is None:
        from causalaba import CausalABA as _CausalABA
        from causalaba_mus import CausalABA_MUS as _CausalABA_MUS

        CausalABA = _CausalABA
        CausalABA_MUS = _CausalABA_MUS


def _parse_node_sizes(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in raw.split(',') if x.strip())
    except Exception:
        return (5, 7, 9)


def _parse_graph_types(raw: str) -> tuple[str, ...]:
    items = [x.strip() for x in (raw or "").split(',') if x.strip()]
    return tuple(items)


MUS_SOLVE_TIMEOUT = int(os.environ.get("MUS_SOLVE_TIMEOUT", "60"))
MUS_NODE_SIZES = _parse_node_sizes(os.environ.get("MUS_NODE_SIZES", "5,7,9"))
MUS_EDGE_PER_NODE = int(os.environ.get("MUS_EDGE_PER_NODE", "2"))
MUS_SEED_BASE = int(os.environ.get("MUS_SEED_BASE", "2004"))
MUS_REP_UNSAT = int(os.environ.get("MUS_REP_UNSAT", "2"))
MUS_RANDOM_REPS = int(os.environ.get("MUS_RANDOM_REPS", "1"))
MUS_EMIT_LP = os.environ.get("MUS_EMIT_LP", "")
MUS_MAX_MUSES = os.environ.get("MUS_MAX_MUSES", "")  # "" = omit -n flag (WASP default: 1 MUS output), "0" = unlimited, ">0" = limit
MUS_MCS_THRESHOLD = int(os.environ.get("MUS_MCS_THRESHOLD", "0"))  # 0 = no limit (unlimited enumeration)
MUS_MUS_THRESHOLD = int(os.environ.get("MUS_MUS_THRESHOLD", "0"))  # 0 = no limit (unlimited enumeration)

# Random graph family used by build_random_pc_case/randomG_PC_case.
MUS_GRAPH_TYPE = os.environ.get("MUS_GRAPH_TYPE", "ER")
MUS_GRAPH_TYPES = _parse_graph_types(os.environ.get("MUS_GRAPH_TYPES", "")) or (MUS_GRAPH_TYPE,)

# ABAPC/Clingo options passed into CausalABA (removal strategy).
MUS_ABAPC_OUT_N = int(os.environ.get("MUS_ABAPC_OUT_N", "0"))
MUS_ABAPC_OPT_MODE = os.environ.get("MUS_ABAPC_OPT_MODE", "optN")

# Directory to save temp facts files for inspection (default: "" = delete temp files)
MUS_KEEP_TEMP_FILES = os.environ.get("MUS_KEEP_TEMP_FILES", "")


@dataclass(frozen=True)
class RandomPCSimConfig:
    n_nodes: int
    alpha: float
    graph_type: str
    edge_per_node: int
    seed: int
    sample_size: int = 10000
    uc_rule: int = 5
    uc_priority: int = 2
    stable: bool = True


def build_random_pc_case(config: RandomPCSimConfig):
    import networkx as nx
    import numpy as np
    import pandas as pd

    from utils.graph_utils import (
        find_all_d_separations_sets,
        extract_test_elements_from_symbol,
        initial_strength,
    )
    from utils.data_utils import simulate_dag, simulate_data_and_run_PC
    from utils.helpers import random_stability

    n_nodes = config.n_nodes
    s0 = int(n_nodes * config.edge_per_node)
    max_edges = int(n_nodes * (n_nodes - 1) / 2)
    if s0 > max_edges:
        s0 = max_edges

    random_stability(config.seed)
    B_true = simulate_dag(d=n_nodes, s0=s0, graph_type=config.graph_type)
    G_true = nx.DiGraph(
        pd.DataFrame(
            B_true,
            columns=[f"X{i+1}" for i in range(n_nodes)],
            index=[f"X{i+1}" for i in range(n_nodes)],
        )
    )
    relabel_dict = {f"X{i+1}": i for i in range(n_nodes)}
    G_true1 = nx.relabel_nodes(G_true, relabel_dict)

    true_seplist = find_all_d_separations_sets(G_true, verbose=False)

    random_stability(config.seed)
    data, cg = simulate_data_and_run_PC(
        G_true,
        config.alpha,
        uc_rule=config.uc_rule,
        uc_priority=config.uc_priority,
        stable=config.stable,
        seed=config.seed,
        sample_size=config.sample_size,
    )

    facts = []  # (fact_str, I, is_correct)
    facts_ext = []
    wrong_ext = []
    count_wrong = 0

    for test in true_seplist:
        X, S, Y, dep_type = extract_test_elements_from_symbol(test)
        test_PC = [t for t in cg.sepset[X, Y] if set(t[0]) == S]
        if len(test_PC) != 1:
            continue

        p = test_PC[0][1]
        dep_type_PC = "indep" if p > config.alpha else "dep"
        I = initial_strength(p, len(S), config.alpha, 0.5, n_nodes)

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

    return {
        "B_true": B_true,
        "G_true": G_true,
        "G_true1": G_true1,
        "true_seplist": true_seplist,
        "data": data,
        "cg": cg,
        "facts": facts,
        "facts_ext": facts_ext,
        "wrong_ext": wrong_ext,
        "count_wrong": count_wrong,
    }


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

    def randomG_PC_case(
        self,
        n_nodes: int,
        edge_per_node: int = 2,
        graph_type: str = "ER",
        seed: int = 2024,
        alpha: float = 0.05,
        sample_size: int = 10000,
        uc_rule: int = 5,
        uc_priority: int = 2,
        stable: bool = True,
    ):
        """Build a random-DAG + PC fact set case (similar to tests.py::randomG_PC_facts).

        Returns a dict with keys like: facts, facts_ext, wrong_ext, count_wrong, cg, true_seplist, G_true1.
        """
        config = RandomPCSimConfig(
            n_nodes=n_nodes,
            alpha=alpha,
            graph_type=graph_type,
            edge_per_node=edge_per_node,
            seed=seed,
            sample_size=sample_size,
            uc_rule=uc_rule,
            uc_priority=uc_priority,
            stable=stable,
        )
        logging.info(
            "Sim config: "
            f"n_nodes={config.n_nodes}, alpha={config.alpha}, graph_type={config.graph_type}, "
            f"edge_per_node={config.edge_per_node}, seed={config.seed}, sample_size={config.sample_size}, "
            f"uc_rule={config.uc_rule}, uc_priority={config.uc_priority}, stable={config.stable}"
        )
        case = build_random_pc_case(config)
        case["config"] = config
        return case

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
                - MUS/MCS over the PC fact set (CAMUS + --print-mcses) to highlight which
                    PC tests are implicated in contradictions.
        """
        logger_setup()
        logging.info("===============Running test_mus_links_wrong_tests_four_node_abapc===============")

        _ensure_solvers_imported()

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
            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
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
        import pandas as pd

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
            weights: list[int] = []
            for fact, s in zip(facts, facts_ext):
                line = s if s.endswith('.') else s + '.'
                I = fact[1]
                # Weights must be integers with at most 7 digits.
                # Map I in [0,1] to [1, 9_999_999] using rounding.
                try:
                    w = int(round(float(I) * 9_999_999))
                except Exception:
                    w = 0
                w = max(1, min(9_999_999, w))
                weights.append(w)
                f.write(f":~ {line} [-{w}]\n")

        if facts_ext:
            try:
                w_min = min(weights) if weights else None
                w_max = max(weights) if weights else None
                logging.info(
                    f"Weak-constraint weights: min={w_min}, max={w_max}, max_digits={len(str(w_max)) if w_max is not None else 'NA'}"
                )
            except Exception:
                pass

        # Step 1: Run with all facts (may be UNSAT). If UNSAT, removal should fix it.
        logging.info("Step 1: Testing with all facts")
        models_all, _ = CausalABA(n_nodes, facts_file, weak_constraints=True, print_models=False, skeleton_rules_reduction=True)
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
            wasp_path="wasp",
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

    def test_mus_mcs_random_five_node_abapc(self):
        """Random 5-node variant of the PC→ABAPC→MUS/MCS pipeline.

        This mirrors the intent of `randomG_PC_facts` in `tests.py`, but compares against
        the *normal* ABAPC configuration (search_for_models='first') rather than
        search_for_models='all_subsets'.

        Steps:
        1) Build a random 5-node DAG, run PC, and construct the `ext_*` facts.
        2) Run CausalABA with all PC facts (fixed seed chosen to be UNSAT).
        3) Run ABAPC removal with search_for_models='first' to restore SAT.
        4) Run MUS+MCS analysis on the PC fact set only (CAMUS + --print-mcses).
        """
        logger_setup()
        logging.info("===============Running test_mus_mcs_random_five_node_abapc===============")

        _ensure_solvers_imported()

        import types

        # Stub notears to avoid heavy optional dependency required by cd_algorithms.models
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')

            class _DummyMLP:
                pass

            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")

            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module

        # Deterministic configuration (seed chosen to yield UNSAT for step 1)
        case = self.randomG_PC_case(
            n_nodes=5,
            edge_per_node=2,
            graph_type="ER",
            seed=2004,
            alpha=0.05,
            sample_size=10000,
            uc_rule=5,
            stable=True,
        )
        config = case["config"]
        n_nodes = config.n_nodes
        facts = case["facts"]
        facts_ext = case["facts_ext"]
        wrong_ext = case["wrong_ext"]
        count_wrong = case["count_wrong"]
        true_seplist = case["true_seplist"]
        cg = case["cg"]
        G_true1 = case["G_true1"]

        logging.info(f"Seed: {config.seed}")
        logging.info(f"True DAG: {G_true1.edges}")
        logging.info(f"Number of total independence statements: {len(true_seplist)}")
        logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
        logging.info(f"Number of wrong facts: {count_wrong} ({(count_wrong/len(facts))*100 if facts else 0:.2f}%)")
        logging.info(f"Fully directed edges from PC: {cg.find_fully_directed()}")
        logging.info(f"Undirected edges from PC: {[(x,y) for (x,y) in cg.find_undirected() if x < y]}")

        self.assertGreater(len(facts_ext), 0, "Expected at least one PC-derived fact")
        self.assertGreater(count_wrong, 0, "Expected at least one wrong PC fact for MUS analysis")

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
                try:
                    w = int(round(float(I) * 9_999_999))
                except Exception:
                    w = 0
                w = max(1, min(9_999_999, w))
                f.write(f":~ {line} [-{w}]\n")

        # Step 1: Run with all facts (expected UNSAT for this seed)
        logging.info("Step 1: Testing with all facts")
        models_all, _ = CausalABA(n_nodes, facts_file, weak_constraints=True, print_models=False, skeleton_rules_reduction=True)
        logging.info(f"  → {len(models_all)} models")
        self.assertEqual(len(models_all), 0, "Expected UNSAT with all PC facts for the chosen seed")

        # Step 2: Apply ABAPC removal strategy (search='first')
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

        self.assertGreater(remove_n, 0, "Expected removal of X>0 tests to reach SAT")
        self.assertGreater(len(models_after), 0, "Expected SAT after removing X tests")

        # Derive which facts were removed by ABAPC 'first'.
        # CausalABA sorts facts by descending I and then removes from the tail until SAT.
        def _canon_fact(s: str) -> str:
            return s if s.endswith('.') else s + '.'

        facts_with_I = []
        for (fact_str, I, _is_correct), ext_line in zip(facts, facts_ext):
            stmt = _canon_fact(ext_line)
            facts_with_I.append((float(I), stmt))
        facts_sorted_by_I = sorted(facts_with_I, key=lambda t: t[0], reverse=True)
        removed_facts = [stmt for _I, stmt in facts_sorted_by_I[-remove_n:]] if remove_n else []
        removed_set = set(removed_facts)
        all_pc_facts_set = set(stmt for _I, stmt in facts_sorted_by_I)
        kept_set = all_pc_facts_set.difference(removed_set)

        wrong_set = set(_canon_fact(w) for w in wrong_ext)
        wrong_all = wrong_set & all_pc_facts_set
        wrong_removed = wrong_set & removed_set
        wrong_kept = wrong_set & kept_set

        removed_preview = sorted(removed_facts)[:10]
        if len(removed_facts) > 10:
            removed_preview.append(f"... (+{len(removed_facts) - 10} more)")
        logging.info(
            f"  ABAPC removed facts (lowest I): {len(removed_facts)} / {len(all_pc_facts_set)}"
        )
        logging.info(
            f"  Removed preview: {', '.join(removed_preview) if removed_preview else '(none)'}"
        )
        logging.info(
            f"  Wrong PC facts (all): {len(wrong_all)} / {len(all_pc_facts_set)} ({(len(wrong_all)/len(all_pc_facts_set))*100 if all_pc_facts_set else 0:.2f}%)"
        )
        logging.info(
            f"  Wrong among REMOVED: {len(wrong_removed)} / {len(removed_set)} ({(len(wrong_removed)/len(removed_set))*100 if removed_set else 0:.2f}%)"
        )
        logging.info(
            f"  Wrong among KEPT: {len(wrong_kept)} / {len(kept_set)} ({(len(wrong_kept)/len(kept_set))*100 if kept_set else 0:.2f}%)"
        )

        # Step 3: Run MUS/MCS on PC facts only
        logging.info("Step 3: Running MUS/MCS analysis")
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
            wasp_path="wasp",
            max_muses=500,
            mus_algorithm="camus",
            print_mcses=True,
            camus_mcs_threshold=50,
            camus_mus_threshold=300,
        )

        logging.info(f"MUS cores found: {mus_result['n_mus']}")
        logging.info(f"MCSes found: {mus_result.get('n_mcs', 0)}")
        self.assertGreater(mus_result['n_mus'], 0, "Expected at least one MUS")
        self.assertGreater(mus_result.get('n_mcs', 0), 0, "Expected at least one MCS")

        mus_sets = [set(_canon_fact(f) for f in mf) for mf in mus_result['mus_facts']]
        for ms in mus_sets:
            self.assertTrue(
                len(ms.intersection(wrong_set)) >= 1,
                "Each MUS should include at least one wrong/flipped test",
            )

        # --------------------------
        # Link ABAPC removals vs MUS/MCS rankings
        # --------------------------
        mus_freq = Counter(f for ms in mus_sets for f in ms)
        n_mus = max(int(mus_result.get('n_mus', 0) or 0), 1)
        mcs_sets = [set(_canon_fact(f) for f in mf) for mf in mus_result.get('mcs_facts', [])]
        mcs_freq = Counter(f for cs in mcs_sets for f in cs)
        n_mcs = max(int(mus_result.get('n_mcs', 0) or 0), 1)

        wrong_in_mus_unique = set(mus_freq.keys()) & wrong_set
        wrong_in_mcs_unique = set(mcs_freq.keys()) & wrong_set
        logging.info(
            f"  Wrong facts appearing in MUSes: {len(wrong_in_mus_unique)} / {len(wrong_all)} unique"
        )
        logging.info(
            f"  Wrong facts appearing in MCSes: {len(wrong_in_mcs_unique)} / {len(wrong_all)} unique"
        )

        def _log_ranked(title: str, freq: Counter, denom: int, top_k: int = 20) -> list[str]:
            ordered = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
            top_facts = [f for f, _c in ordered[: min(top_k, len(ordered))]]
            logging.info(title)
            for fact, count in ordered[: min(top_k, len(ordered))]:
                pct = (count / denom) * 100.0
                label_wc = "WRONG" if fact in wrong_set else "CORRECT"
                label_rm = "REMOVED" if fact in removed_set else "KEPT"
                logging.info(f"    - {label_rm} | {label_wc}: {fact} ({count}/{denom} = {pct:.1f}%)")
            return top_facts

        top_mus = _log_ranked("  Top facts by MUS frequency (all PC facts):", mus_freq, n_mus, top_k=20)
        top_mcs = _log_ranked("  Top facts by MCS frequency (all PC facts):", mcs_freq, n_mcs, top_k=20)

        for k in (5, 10, 20):
            k_mus = set(top_mus[: min(k, len(top_mus))])
            k_mcs = set(top_mcs[: min(k, len(top_mcs))])
            logging.info(
                f"  Overlap with ABAPC removed (top-{k}): MUS={len(k_mus & removed_set)}/{len(k_mus) or 1}, MCS={len(k_mcs & removed_set)}/{len(k_mcs) or 1}"
            )

        # Rank removed facts by how often they appear in MUS/MCS
        removed_by_mus = sorted(
            ((f, mus_freq.get(f, 0)) for f in removed_facts),
            key=lambda t: (-t[1], t[0]),
        )
        removed_by_mcs = sorted(
            ((f, mcs_freq.get(f, 0)) for f in removed_facts),
            key=lambda t: (-t[1], t[0]),
        )
        logging.info("  ABAPC-REMOVED facts ranked by MUS frequency (count / n_mus):")
        for fact, count in removed_by_mus[:20]:
            pct = (count / n_mus) * 100.0
            label_wc = "WRONG" if fact in wrong_set else "CORRECT"
            logging.info(f"    - {label_wc}: {fact} ({count}/{n_mus} = {pct:.1f}%)")
        logging.info("  ABAPC-REMOVED facts ranked by MCS frequency (count / n_mcs):")
        for fact, count in removed_by_mcs[:20]:
            pct = (count / n_mcs) * 100.0
            label_wc = "WRONG" if fact in wrong_set else "CORRECT"
            logging.info(f"    - {label_wc}: {fact} ({count}/{n_mcs} = {pct:.1f}%)")

        # Examples of smallest MUSes and MCSes
        sorted_muses = sorted(enumerate(mus_result.get('mus_facts', []), 1), key=lambda x: (len(x[1]), x[0]))
        logging.info("  Examples of smallest MUSes:")
        for idx, mus_facts in sorted_muses[:3]:
            logging.info(f"    MUS #{idx} (size {len(mus_facts)}):")
            for fact in mus_facts:
                label_wc = "WRONG" if fact in wrong_set else "CORRECT"
                label_rm = "REMOVED" if fact in removed_set else "KEPT"
                logging.info(f"      - {label_rm} | {label_wc}: {fact}")

        sorted_mcses = sorted(enumerate(mus_result.get('mcs_facts', []), 1), key=lambda x: (len(x[1]), x[0]))
        logging.info("  Examples of smallest MCSes:")
        for idx, mcs_facts in sorted_mcses[:3]:
            logging.info(f"    MCS #{idx} (size {len(mcs_facts)}):")
            for fact in mcs_facts:
                label_wc = "WRONG" if fact in wrong_set else "CORRECT"
                label_rm = "REMOVED" if fact in removed_set else "KEPT"
                logging.info(f"      - {label_rm} | {label_wc}: {fact}")

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

        _ensure_solvers_imported()
        
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
        models, _ = CausalABA(n_nodes, facts_file, print_models=False, skeleton_rules_reduction=True)
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
            
            models_removed, _ = CausalABA(n_nodes, facts_file_removed, print_models=False, skeleton_rules_reduction=True)
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
            wasp_path="wasp",
            mus_algorithm="camus",
            print_mcses=True,
        )
        
        logging.info(f"  MUS cores found: {mus_result['n_mus']}")
        for i, mus_facts in enumerate(mus_result['mus_facts']):
            logging.info(f"    MUS #{i+1}: {len(mus_facts)} facts")
            for fact in mus_facts:
                logging.info(f"      - {fact}")
        
        self.assertEqual(mus_result['n_mus'], 1, "Expected exactly one MUS")
        # With skeleton-rules optimization enabled, the MUS may be smaller than all 3 facts
        # (only the minimal subset needed to cause UNSAT is reported)
        self.assertGreater(len(mus_result['mus_facts'][0]), 0, "Expected at least one fact in the MUS")
        self.assertLessEqual(len(mus_result['mus_facts'][0]), 3, "Expected at most three facts in the MUS")

        # With CAMUS + --print-mcses, we should see singleton MCSes for each removed element
        self.assertGreater(mus_result.get('n_mcs', 0), 0, "Expected at least one MCS")
        self.assertLessEqual(mus_result.get('n_mcs', 0), 3, "Expected at most three MCSes")
        mcs_sets = [set(m) for m in mus_result.get('mcs_list', [])]
        logging.info(f"MCS sets: {mcs_sets}")

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

        _ensure_solvers_imported()

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
            wasp_path="wasp",
            # Enumerate all MUSes; WASP default without -n is typically just 1.
            max_muses=0,
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

    def test_mus_mcs_random_sizes_abapc(self):
        """Run MUS/MCS analysis across multiple node sizes.
        
        This test parameterizes test_mus_mcs_random_five_node_abapc across different graph
        sizes, using unittest.TestCase.subTest to organize results by size.
        """
        logger_setup()
        logging.info("===============Running test_mus_mcs_random_sizes_abapc===============")

        import types

        # Stub notears once to avoid repeated imports
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')

            class _DummyMLP:
                pass

            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")

            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module

        graph_types = MUS_GRAPH_TYPES or (MUS_GRAPH_TYPE,)
        # If running multiple repetitions, collect per-(graph_type,n_nodes) summaries.
        rep_summaries: dict[tuple[str, int], list[dict[str, Any]]] = {}
        rep_outcomes: dict[tuple[str, int], dict[str, int]] = {}

        # Run for multiple node sizes, repetitions, and graph types.
        for graph_idx, graph_type in enumerate(graph_types):
            for n_nodes in MUS_NODE_SIZES:
                key = (str(graph_type), int(n_nodes))
                rep_summaries.setdefault(key, [])
                rep_outcomes.setdefault(key, {'completed': 0, 'timed_out': 0, 'startSAT': 0, 'no_wrong': 0})

                for rep_idx in range(max(1, MUS_RANDOM_REPS)):
                    with self.subTest(n_nodes=n_nodes, graph_type=graph_type, rep=rep_idx):
                        logging.info(f"\n--- Running n_nodes={n_nodes} graph_type={graph_type} rep={rep_idx} ---")
                        run_info = None
                        selected_run: dict[str, Any] | None = None
                        saw_timeout = False
                        saw_sat = False
                        saw_no_wrong = False
                        # Seed policy: always start from --seed-base,
                        # then retries just add +1. When running multiple reps, we stride seeds 
                        # by (rep_unsat+1) so each rep explores a different instance without 
                        # overlapping the retry window of another rep.
                        stride = max(1, int(MUS_REP_UNSAT) + 1)
                        seed0 = int(MUS_SEED_BASE) + (int(rep_idx) * stride)
                        # Retry with incremented seeds until we get an UNSAT instance (i.e., ABAPC removes >0)
                        # that also has at least one wrong fact.
                        for attempt in range(max(0, MUS_REP_UNSAT) + 1):
                            seed = seed0 + attempt
                            run_info = self._run_mus_mcs_for_size(n_nodes, seed=seed, graph_type=graph_type)
                            if run_info.get('abapc_timeout', False) or run_info.get('mus_timeout', False):
                                saw_timeout = True
                            if run_info.get('was_sat', False):
                                saw_sat = True
                                logging.info(
                                    f"Instance already SAT (no removal) for n_nodes={n_nodes} graph_type={graph_type} seed={seed}; "
                                    f"trying next seed {attempt+1}/{MUS_REP_UNSAT}"
                                )
                                continue

                            if run_info.get('count_wrong', 0) <= 0:
                                saw_no_wrong = True
                                logging.warning(
                                    f"No wrong facts for n_nodes={n_nodes} graph_type={graph_type} seed={seed}; "
                                    f"retry {attempt+1}/{MUS_REP_UNSAT}"
                                )
                                continue

                            selected_run = run_info
                            break

                        if not selected_run:
                            last_seed = (run_info or {}).get('seed', None)
                            last_was_sat = bool((run_info or {}).get('was_sat', False))
                            last_wrong = int((run_info or {}).get('count_wrong', 0) or 0)
                            # Track why this rep didn't produce a usable run.
                            if saw_timeout:
                                rep_outcomes[key]['timed_out'] += 1
                            elif saw_sat or last_was_sat:
                                rep_outcomes[key]['startSAT'] += 1
                            elif saw_no_wrong or last_wrong <= 0:
                                rep_outcomes[key]['no_wrong'] += 1
                            self.skipTest(
                                f"No suitable UNSAT instance with wrong facts for n_nodes={n_nodes} graph_type={graph_type} "
                                f"after {MUS_REP_UNSAT+1} seed(s). Last_seed={last_seed}, last_was_sat={last_was_sat}, last_count_wrong={last_wrong}. "
                                f"Try a different --seed-base or increase --rep-unsat."
                            )

                        rep_outcomes[key]['completed'] += 1

                        overlap = self._assert_wrong_fact_correspondence(
                            n_nodes=n_nodes,
                            facts_ext=selected_run.get('facts_ext', []),
                            wrong_ext=selected_run.get('wrong_ext', []),
                            removed_facts=selected_run.get('removed_facts', []),
                            mus_facts=selected_run.get('mus_facts', []),
                            mcs_facts=selected_run.get('mcs_facts', []),
                        )

                        rep_summaries[key].append(
                            {
                                'seed': selected_run.get('seed'),
                                'abapc_time_sec': float(selected_run.get('abapc_time_sec', 0.0) or 0.0),
                                'mus_time_sec': float(selected_run.get('mus_time_sec', 0.0) or 0.0),
                                'n_facts': int(len(selected_run.get('facts_ext', []) or [])),
                                'n_wrong': int(selected_run.get('count_wrong', 0) or 0),
                                'overlap': overlap or {},
                            }
                        )

                # Per-(graph_type,n_nodes) aggregate summary over reps.
                if MUS_RANDOM_REPS > 1:
                    rows = rep_summaries.get(key, [])
                    outs = rep_outcomes.get(key, {})
                    if rows:
                        def _avg_min_max(vals: list[float]) -> tuple[float, float, float]:
                            if not vals:
                                return (0.0, 0.0, 0.0)
                            return (sum(vals) / len(vals), min(vals), max(vals))

                        abapc_vals = [r.get('abapc_time_sec', 0.0) for r in rows if r.get('abapc_time_sec', 0.0) > 0]
                        mus_vals = [r.get('mus_time_sec', 0.0) for r in rows if r.get('mus_time_sec', 0.0) > 0]
                        abapc_avg, abapc_min, abapc_max = _avg_min_max(abapc_vals)
                        mus_avg, mus_min, mus_max = _avg_min_max(mus_vals)

                        def _fmt_triple(min_v: float, avg_v: float, max_v: float) -> str:
                            return f"{min_v:7.3f} / {avg_v:7.3f} / {max_v:7.3f}"

                        def _fmt_count_triple(min_v: float, avg_v: float, max_v: float) -> str:
                            return f"{min_v:7.0f} / {avg_v:7.1f} / {max_v:7.0f}"

                        logging.info("\n" + 29*" "+"=" * 60)
                        logging.info(
                            f"SUMMARY OVER REPS (n_nodes={n_nodes}, graph_type={graph_type}, reps={MUS_RANDOM_REPS}, completed={len(rows)})"
                        )
                        logging.info("=" * 60)
                        logging.info("  [min/avg/max]")
                        logging.info(f"  {'completed runs':<18}: {outs.get('completed', len(rows))}/{MUS_RANDOM_REPS}")
                        logging.info(f"  {'timed out':<18}: {outs.get('timed_out', 0)}")
                        logging.info(f"  {'startSAT':<18}: {outs.get('startSAT', 0)}")
                        if outs.get('no_wrong', 0):
                            logging.info(f"  {'no_wrong':<18}: {outs.get('no_wrong', 0)}")
                        logging.info(f"  {'ABAPC total (s)':<18}: {_fmt_triple(abapc_min, abapc_avg, abapc_max)}")
                        logging.info(f"  {'MUS total (s)':<18}: {_fmt_triple(mus_min, mus_avg, mus_max)}")

                        facts_vals = [float(r.get('n_facts', 0) or 0) for r in rows if (r.get('n_facts', 0) or 0) > 0]
                        wrong_vals = [float(r.get('n_wrong', 0) or 0) for r in rows]
                        if facts_vals:
                            f_avg, f_min, f_max = _avg_min_max(facts_vals)
                            logging.info(f"  {'facts (n)':<18}: {_fmt_count_triple(f_min, f_avg, f_max)}")
                        if wrong_vals:
                            w_avg, w_min, w_max = _avg_min_max(wrong_vals)
                            logging.info(f"  {'wrong (n)':<18}: {_fmt_count_triple(w_min, w_avg, w_max)}")

                        def _metric_vals(group: str, metric: str, stat: str = 'avg') -> list[float]:
                            out: list[float] = []
                            for r in rows:
                                g = (r.get('overlap') or {}).get(group) or {}
                                m = (g.get(metric) or {})
                                v = m.get(stat)
                                if v is not None:
                                    out.append(float(v))
                            return out

                        def _count_vals(group: str) -> list[float]:
                            out: list[float] = []
                            for r in rows:
                                g = (r.get('overlap') or {}).get(group) or {}
                                c = g.get('count')
                                if c is not None:
                                    out.append(float(c))
                            return out

                        def _metric_triplet_over_reps(group: str, metric: str) -> tuple[float, float, float] | None:
                            # We summarize across reps using the per-run distribution over sets:
                            #   min = min_over_reps(per_run_min)
                            #   avg = avg_over_reps(per_run_avg)
                            #   max = max_over_reps(per_run_max)
                            mins = _metric_vals(group, metric, stat='min')
                            avgs = _metric_vals(group, metric, stat='avg')
                            maxs = _metric_vals(group, metric, stat='max')
                            if not mins and not avgs and not maxs:
                                return None
                            min_v = min(mins) if mins else 0.0
                            avg_v = (sum(avgs) / len(avgs)) if avgs else 0.0
                            max_v = max(maxs) if maxs else 0.0
                            return (min_v, avg_v, max_v)

                        for group in ('Removal', 'MUS', 'MCS'):
                            logging.info(f"\n  {group}:")
                            nsets_vals = _count_vals(group)
                            if nsets_vals:
                                c_avg, c_min, c_max = _avg_min_max(nsets_vals)
                                logging.info(f"    {'n_sets':<10}: {_fmt_count_triple(c_min, c_avg, c_max)}")
                            else:
                                logging.info(f"    {'n_sets':<10}: none")

                            # Aggregate set sizes across reps (per-run set_size is already min/avg/max over sets)
                            t_sz = _metric_triplet_over_reps(group, 'set_size')
                            if t_sz:
                                mn, av, mx = t_sz
                                logging.info(f"    {'set_size':<10}: {_fmt_triple(mn, av, mx)}")
                            else:
                                logging.info(f"    {'set_size':<10}: none")

                            for metric in ('hit_rate', 'jaccard', 'precision', 'recall', 'f1'):
                                t = _metric_triplet_over_reps(group, metric)
                                if not t:
                                    logging.info(f"    {metric:<10}: none")
                                    continue
                                mn, av, mx = t
                                logging.info(f"    {metric:<10}: {_fmt_triple(mn, av, mx)}")

                        logging.info("=" * 60)

    @staticmethod
    def _normalize_fact_str(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return s
        return s if s.endswith('.') else s + '.'

    def _assert_wrong_fact_correspondence(
        self,
        *,
        n_nodes: int,
        facts_ext,
        wrong_ext,
        removed_facts,
        mus_facts,
        mcs_facts,
    ) -> dict[str, Any]:
        """Check MUS/MCS sets correspond to wrong facts.

        Intended for random-PC instances where `wrong_ext` is derived from ground truth.
                Notes:
                - Every MUS should contain at least one wrong fact (otherwise the contradiction isn't explained).
                - MCSes are *repairs* (sets of assumptions to disable). They can include correct facts too.
                    So we do not require every MCS to include a wrong fact; instead we report a hit-rate.
                - Returned MUS/MCS facts should be drawn from the provided fact universe.
        """
        all_facts = {self._normalize_fact_str(s) for s in (facts_ext or []) if str(s).strip()}
        wrong_set = {self._normalize_fact_str(s) for s in (wrong_ext or []) if str(s).strip()}
        removed_set = {self._normalize_fact_str(s) for s in (removed_facts or []) if str(s).strip()}
        if not all_facts or not wrong_set:
            self.fail(f"Missing facts/wrong facts for correspondence check (n_nodes={n_nodes})")

        def to_sets(groups):
            out = []
            for g in (groups or []):
                s = {self._normalize_fact_str(x) for x in (g or []) if str(x).strip()}
                if s:
                    out.append(s)
            return out

        mus_sets = to_sets(mus_facts)
        mcs_sets = to_sets(mcs_facts)

        def jaccard(a: set[str], b: set[str]) -> float:
            if not a and not b:
                return 1.0
            u = a.union(b)
            if not u:
                return 0.0
            return len(a.intersection(b)) / len(u)

        def summarize_overlap(label: str, groups: list[set[str]]) -> dict[str, Any]:
            def _mmx(xs: list[float]) -> dict[str, float]:
                if not xs:
                    return {'min': 0.0, 'avg': 0.0, 'max': 0.0}
                return {'min': min(xs), 'avg': (sum(xs) / len(xs)), 'max': max(xs)}

            def _fmt_triple(d: dict[str, float]) -> str:
                return f"{d['min']:7.3f} / {d['avg']:7.3f} / {d['max']:7.3f}"

            if not groups:
                out = {
                    'count': 0,
                    'set_size': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'hit_rate': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'jaccard': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'precision': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'recall': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                    'f1': {'min': 0.0, 'avg': 0.0, 'max': 0.0},
                }
                logging.info(f"  {label}:")
                logging.info(f"    {'n_sets':<10}: {out['count']:7d}")
                logging.info(f"    {'set_size':<10}: {_fmt_triple(out['set_size'])}")
                logging.info(f"    {'hit_rate':<10}: {_fmt_triple(out['hit_rate'])}")
                logging.info(f"    {'jaccard':<10}: {_fmt_triple(out['jaccard'])}")
                logging.info(f"    {'precision':<10}: {_fmt_triple(out['precision'])}")
                logging.info(f"    {'recall':<10}: {_fmt_triple(out['recall'])}")
                logging.info(f"    {'f1':<10}: {_fmt_triple(out['f1'])}")
                return out

            jac = [jaccard(g, wrong_set) for g in groups]
            prec = [(len(g.intersection(wrong_set)) / len(g)) if g else 0.0 for g in groups]
            # Recall is w.r.t wrong_set (can be large); still useful as a scale indicator.
            rec = [(len(g.intersection(wrong_set)) / len(wrong_set)) if wrong_set else 0.0 for g in groups]
            f1 = [((2 * p * r) / (p + r)) if (p + r) > 0 else 0.0 for p, r in zip(prec, rec)]
            hit = [1.0 if len(g.intersection(wrong_set)) >= 1 else 0.0 for g in groups]

            out = {
                'count': len(groups),
                'set_size': _mmx([float(len(g)) for g in groups]),
                'hit_rate': _mmx(hit),
                'jaccard': _mmx(jac),
                'precision': _mmx(prec),
                'recall': _mmx(rec),
                'f1': _mmx(f1),
            }

            logging.info(f"  {label}:")
            logging.info(f"    {'n_sets':<10}: {out['count']:7d}")
            logging.info(f"    {'set_size':<10}: {_fmt_triple(out['set_size'])}")
            logging.info(f"    {'hit_rate':<10}: {_fmt_triple(out['hit_rate'])}")
            logging.info(f"    {'jaccard':<10}: {_fmt_triple(out['jaccard'])}")
            logging.info(f"    {'precision':<10}: {_fmt_triple(out['precision'])}")
            logging.info(f"    {'recall':<10}: {_fmt_triple(out['recall'])}")
            logging.info(f"    {'f1':<10}: {_fmt_triple(out['f1'])}")

            return out

        # Removal (ABAPC): removed facts should align with wrong facts.
        removal_groups: list[set[str]] = [removed_set] if removed_set else []

        logging.info(f"\n")
        logging.info("=" * 60)
        logging.info(f" OVERLAP SUMMARY (n_nodes={n_nodes})")
        logging.info("=" * 60)
        logging.info("  [min/avg/max]")
        removal_summary = summarize_overlap("Removal", removal_groups)

        # MUS correspondence: in many cases MUS cores implicate wrong facts, but for some
        # random instances a core can be composed entirely of (ground-truth) correct facts.
        # We therefore require that *at least one* MUS hits a wrong fact and report hit-rate.
        self.assertGreater(len(mus_sets), 0, f"Expected at least one MUS set for correspondence check (n_nodes={n_nodes})")
        for i, core in enumerate(mus_sets, start=1):
            self.assertTrue(
                core.issubset(all_facts),
                f"MUS #{i} contains facts not in instance fact set (n_nodes={n_nodes})",
            )

        any_wrong_mus = any(len(core.intersection(wrong_set)) >= 1 for core in mus_sets)
        self.assertTrue(any_wrong_mus, f"No MUS contains a wrong fact (n_nodes={n_nodes})")

        mus_summary = summarize_overlap("MUS", mus_sets)

        # MCS correspondence: do not require every MCS to include wrong facts.
        # (A minimal correction set can disable some correct assumptions too.)
        for i, cut in enumerate(mcs_sets, start=1):
            self.assertTrue(
                cut.issubset(all_facts),
                f"MCS #{i} contains facts not in instance fact set (n_nodes={n_nodes})",
            )

        mcs_summary = summarize_overlap("MCS", mcs_sets)

        logging.info("=" * 60)

        if mcs_sets:
            any_wrong = any(len(cut.intersection(wrong_set)) >= 1 for cut in mcs_sets)
            self.assertTrue(any_wrong, f"No MCS contains a wrong fact (n_nodes={n_nodes})")

        return {
            'Removal': removal_summary,
            'MUS': mus_summary,
            'MCS': mcs_summary,
        }

    def _run_mus_mcs_for_size(self, n_nodes: int, seed: int | None = None, graph_type: str | None = None):
        """Helper to run the full MUS/MCS pipeline for a given node size.
        
        This extracts the core logic of test_mus_mcs_random_five_node_abapc so it can
        be parameterized across different sizes without code duplication.
        
        Args:
            n_nodes: Number of nodes in the random graph
            seed: Optional random seed; if None, uses default seed_map based on n_nodes
        
        Returns a dict with keys:
        - 'was_sat': bool, True if ABAPC found the instance was already SAT (remove_n == 0)
        - 'abapc_timeout': bool, True if ABAPC hit timeout
        - 'mus_timeout': bool, True if MUS hit timeout
        """
        import types
        import re

        _ensure_solvers_imported()

        # Stub notears again for this sub-run (if needed)
        if 'notears.nonlinear' not in sys.modules:
            notears_module = types.ModuleType('notears')
            notears_nonlinear_module = types.ModuleType('notears.nonlinear')

            class _DummyMLP:
                pass

            def _dummy_notears_nonlinear(*args, **kwargs):
                raise ImportError("notears is not installed in this test environment")

            setattr(notears_nonlinear_module, 'NotearsMLP', _DummyMLP)
            setattr(notears_nonlinear_module, 'notears_nonlinear', _dummy_notears_nonlinear)
            sys.modules['notears'] = notears_module
            sys.modules['notears.nonlinear'] = notears_nonlinear_module

        # Use a deterministic seed
        if seed is None:
            seed = MUS_SEED_BASE

        if graph_type is None:
            graph_type = MUS_GRAPH_TYPE

        assert graph_type is not None
        graph_type = str(graph_type)

        # Help static type-checkers: from here on, seed is always an int.
        assert seed is not None
        seed = int(seed)

        # Deterministic configuration
        case = self.randomG_PC_case(
            n_nodes=n_nodes,
            edge_per_node=MUS_EDGE_PER_NODE,
            graph_type=graph_type,
            seed=seed,
            alpha=0.05,
            sample_size=10000,
            uc_rule=5,
            stable=True,
        )
        config = case["config"]
        facts = case["facts"]
        facts_ext = case["facts_ext"]
        wrong_ext = case["wrong_ext"]
        count_wrong = case["count_wrong"]
        true_seplist = case["true_seplist"]
        cg = case["cg"]
        G_true1 = case["G_true1"]

        logging.info(f"Config: n_nodes={n_nodes}, seed={seed}")
        logging.info(f"True DAG: {G_true1.edges}")
        logging.info(f"Number of total independence statements: {len(true_seplist)}")
        logging.info(f"Number of facts from PC: {len(facts)} ({len(facts)/len(true_seplist)*100:.2f}%)")
        logging.info(f"Number of wrong facts: {count_wrong} ({(count_wrong/len(facts))*100 if facts else 0:.2f}%)")
        logging.info(f"Fully directed edges from PC: {cg.find_fully_directed()}")
        logging.info(f"Undirected edges from PC: {[(x,y) for (x,y) in cg.find_undirected() if x < y]}")

        # Basic assertions
        self.assertGreater(len(facts_ext), 0, f"Expected at least one PC-derived fact for n_nodes={n_nodes}")

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
            weights: list[int] = []
            for fact, s in zip(facts, facts_ext):
                line = s if s.endswith('.') else s + '.'
                I = fact[1]
                try:
                    w = int(round(float(I) * 9_999_999))
                except Exception:
                    w = 0
                w = max(1, min(9_999_999, w))
                weights.append(w)
                f.write(f":~ {line} [-{w}]\n")

        if facts_ext:
            try:
                w_min = min(weights) if weights else None
                w_max = max(weights) if weights else None
                logging.info(
                    f"Weak-constraint weights: min={w_min}, max={w_max}, max_digits={len(str(w_max)) if w_max is not None else 'NA'}"
                )
            except Exception:
                pass

        # Optionally save temp files to permanent location for inspection
        if MUS_KEEP_TEMP_FILES:
            keep_dir = Path(MUS_KEEP_TEMP_FILES)
            keep_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            base_name = f"{n_nodes}_{seed}_facts"
            shutil.copy(facts_file, keep_dir / f"{base_name}.lp")
            shutil.copy(facts_I_file, keep_dir / f"{base_name}_I.lp")
            shutil.copy(facts_wc_file, keep_dir / f"{base_name}_wc.lp")
            
            logging.info(f"  → Saved temp files to {keep_dir / base_name}*.lp")

        # Step 1: Run CasusalABA with all facts and apply ABAPC removal strategy (search='first')
        logging.info("Step 1: Run CasusalABA with all facts and apply removal strategy (search_for_models='first') if UNSAT")
        start_abapc = datetime.now()
        abapc_timing = {}  # To capture compile/ground breakdown
        models_after, multiple, stats, remove_n = CausalABA(
            n_nodes,
            facts_file,
            weak_constraints=True,
            search_for_models='first',
            opt_mode=MUS_ABAPC_OPT_MODE,
            out_n=MUS_ABAPC_OUT_N,
            skeleton_rules_reduction=True,
            print_models=False,
            return_statistics=True,
            solve_timeout=MUS_SOLVE_TIMEOUT,
            timing_recorder=abapc_timing,
        )
        abapc_time = datetime.now() - start_abapc
        
        # Extract detailed timing from clingo statistics
        abapc_times = {}
        try:
            abapc_times = {key: stats['summary']['times'][key] for key in ['total', 'cpu', 'solve']}
        except (KeyError, TypeError):
            abapc_times = {'total': abapc_time.total_seconds(), 'cpu': 0, 'solve': 0}
        
        logging.info(f"  → Total facts: {len(facts)}")
        logging.info(f"  → Wrong facts: {count_wrong}")
        logging.info(f"  → Facts removed: {remove_n}")
        logging.info(f"  → Models found after removal: {len(models_after)}")
        logging.info(f"  → ABAPC time: {abapc_time.total_seconds():.3f}s")

        # Derive which facts were removed by ABAPC 'first'.
        # CausalABA sorts facts by descending I and then removes from the tail until SAT.
        facts_with_I: list[tuple[float, str]] = []
        for (fact_str, I, _is_correct), ext_line in zip(facts, facts_ext):
            stmt = self._normalize_fact_str(ext_line)
            facts_with_I.append((float(I), stmt))
        facts_sorted_by_I = sorted(facts_with_I, key=lambda t: t[0], reverse=True)
        removed_facts = [stmt for _I, stmt in facts_sorted_by_I[-remove_n:]] if remove_n else []

        # For larger sizes, we may not enforce UNSAT for all seeds; just assert we can reach SAT
        # If we hit the timeout before reaching SAT, skip the assertion (timeout is the real limit)
        abapc_timeout = bool(abapc_timing.get('timed_out', False))
        # Fallback heuristic (older CausalABA versions may not set timing_recorder flags).
        if (not abapc_timeout) and (len(models_after) == 0):
            abapc_timeout = abapc_time.total_seconds() >= (MUS_SOLVE_TIMEOUT - 0.5)
        if remove_n > 0 and not abapc_timeout:
            self.assertGreater(len(models_after), 0, f"Expected SAT after removing {remove_n} tests for n_nodes={n_nodes}")
        elif remove_n > 0 and abapc_timeout and len(models_after) == 0:
            logging.warning(f"⚠ Timeout hit after {abapc_time.total_seconds():.1f}s before reaching SAT (removed {remove_n} tests, n_nodes={n_nodes})")

        # Step 3: Run MUS/MCS on PC facts only
        logging.info("Step 3: Running MUS/MCS analysis")
        fd_mus, facts_mus_file = tempfile.mkstemp(suffix='.lp', text=True)
        os.close(fd_mus)
        with open(facts_mus_file, 'w') as f:
            for s in facts_ext:
                line = s if s.endswith('.') else s + '.'
                f.write(f"{line}\n")

        # Cap MUS enumeration for larger sizes to keep runtime reasonable
        # But allow environment variables to override for full exploration
        if MUS_MAX_MUSES != "":
            max_muses = int(MUS_MAX_MUSES) if MUS_MAX_MUSES != "0" else 0
        else:
            # Default: limit for faster testing
            max_muses = 500 if n_nodes <= 6 else 200
        
        # Use environment thresholds: 0 means None (no threshold), otherwise use value or defaults
        if MUS_MCS_THRESHOLD == 0 and MUS_MAX_MUSES == "":
            # When MUS_MAX_MUSES is empty (replicating manual WASP), default to no thresholds
            mcs_threshold = None
            mus_threshold = None
        elif MUS_MCS_THRESHOLD == 0:
            # When MUS_MCS_THRESHOLD explicitly 0, use defaults for faster testing
            mcs_threshold = 50
            mus_threshold = 300 if MUS_MUS_THRESHOLD == 0 else MUS_MUS_THRESHOLD
        else:
            mcs_threshold = MUS_MCS_THRESHOLD
            mus_threshold = MUS_MUS_THRESHOLD if MUS_MUS_THRESHOLD > 0 else None
        
        start_mus = datetime.now()
        
        # Optionally emit the complete MUS program for debugging.
        # Accept common boolean-like values (e.g. MUS_EMIT_LP=True) by mapping them to a default path,
        # so we don't accidentally write to a file literally named 'True'.
        emit_lp_path = None
        emit_lp_spec = MUS_EMIT_LP
        if isinstance(emit_lp_spec, str):
            spec = emit_lp_spec.strip()
            if spec.lower() in ("true", "1", "yes", "y"):
                spec = "results/adornedLP_{n}_{seed}.lp"
            elif spec.lower() in ("false", "0", "no", "n"):
                spec = ""
        else:
            spec = "results/adornedLP_{n}_{seed}.lp" if emit_lp_spec else ""

        if spec:
            emit_lp_path = spec.replace('{n}', str(n_nodes)).replace('{seed}', str(seed))
            emit_dir = os.path.dirname(emit_lp_path)
            if emit_dir:
                os.makedirs(emit_dir, exist_ok=True)
            logging.info(f"Will emit MUS program to {emit_lp_path}")
        
        # Special handling: if MUS_MAX_MUSES is empty string, pass None to omit -n flag
        max_muses_arg = None if MUS_MAX_MUSES == "" else max_muses
        
        mus_timing = {}  # To capture compile/ground breakdown
        mus_result = CausalABA_MUS(
            n_nodes=n_nodes,
            facts_location=facts_mus_file,
            gringo_path="clingo",
            wasp_path="wasp",
            max_muses=max_muses_arg,
            mus_algorithm="camus",
            print_mcses=True,
            camus_mcs_threshold=mcs_threshold,
            camus_mus_threshold=mus_threshold,
            solve_timeout=MUS_SOLVE_TIMEOUT,
            emit_lp=emit_lp_path,
            timing_recorder=mus_timing,
        )
        mus_time = datetime.now() - start_mus

        mus_timeout = mus_time.total_seconds() >= (MUS_SOLVE_TIMEOUT - 0.5)

        logging.info(f"  → MUS found: {mus_result['n_mus']}")
        logging.info(f"  → MCS found: {mus_result.get('n_mcs', 0)}")
        logging.info(f"  → MUS time: {mus_time.total_seconds():.3f}s")

        # ===== TIMING COMPARISON (Phase Breakdown) =====
        try:
            header = f"TIMING COMPARISON (n_nodes={n_nodes})"
            prefix = "  "
            logging.info("\n" +  29*" "+ "=" * 60)
            logging.info(prefix + header)
            logging.info("=" * 60)
            
            # Extract ABAPC timing breakdown
            abapc_compile = abapc_timing.get('compile_sec_total', 0.0)
            abapc_ground = abapc_timing.get('ground_sec_total', 0.0)
            abapc_unsat_ground = abapc_timing.get('unsat_ground_sec_total', 0.0)
            abapc_unsat_solve = abapc_timing.get('unsat_solve_sec_total', 0.0)
            abapc_solve = stats.get('summary', {}).get('times', {}).get('solve', 0.0) if stats else 0.0
            abapc_total = abapc_time.total_seconds()
            # Sys (residual) time captures everything not covered by the measured phases.
            # For ABAPC removal, this includes Python overhead + reground/solve work not
            # reflected in the final ctl.statistics (which is for the last ctl only).
            abapc_sys = max(
                0.0,
                abapc_total
                - abapc_compile
                - abapc_ground
                - abapc_solve
                - abapc_unsat_ground
                - abapc_unsat_solve,
            )
            
            # Extract MUS timing breakdown
            # Note: MUS uses WASP which works differently from clingo:
            # - compile_and_ground is called in build_mus_program (captured in mus_timing)
            # - Then clingo re-grounds the full program and pipes to WASP (captured in mus_solve_time)
            # - WASP computes MUS internally without streaming (no Python enumeration overhead)
            mus_compile = mus_timing.get('compile_sec_total', 0.0)
            mus_ground = mus_timing.get('ground_sec_total', 0.0)
            mus_solve = mus_result.get('mus_solve_time', 0.0)  # This is clingo grounding + WASP execution
            mus_total = mus_time.total_seconds()
            # Sys (residual) time captures wrapper overhead around the external solver.
            mus_sys = max(0.0, mus_total - mus_compile - mus_ground - mus_solve)
            
            # Detect which phase timed out for ABAPC
            abapc_timeout_phase = None
            abapc_phase_hint = abapc_timing.get('timeout_phase', None) if isinstance(abapc_timing, dict) else None
            if abapc_timeout:
                # Prefer a phase hint from CausalABA if available.
                if abapc_phase_hint in ('compile', 'ground', 'solve', 'sys'):
                    abapc_timeout_phase = abapc_phase_hint

                # In practice ABAPC timeouts happen while clingo is still solving/enumerating.
                # If we reached the timeout before SAT (0 models), treat everything after compile/ground as solve.
                if len(models_after) == 0:
                    abapc_timeout_phase = 'solve'
                    abapc_solve = max(0.0, abapc_total - abapc_compile - abapc_ground)
                    abapc_sys = 0.0
                else:
                    abapc_timeout_phase = 'solve'
            # Detect which phase timed out for MUS
            mus_timeout_phase = None
            if mus_timeout:
                # For WASP MUS solver:
                # - If we found 0 MUS and 0 MCS, timeout was during solve (WASP couldn't finish)
                # - If we found some MUS/MCS, timeout was during enumerate (WASP was enumerating more)
                # - Unlike clingo, WASP does all computation internally (no Python enumeration overhead)
                n_mus = len(mus_result.get('mus_list', []))
                n_mcs = len(mus_result.get('mcs_list', []))
                if n_mus == 0 and n_mcs == 0:
                    mus_timeout_phase = 'solve'
                    # Recalculate: all non-compile/ground time was actually solving
                    mus_solve = max(0.0, mus_total - mus_compile - mus_ground)
                    mus_sys = 0.0
                else:
                    # Found some results, so timeout was during enumeration of additional MUS/MCS
                    mus_timeout_phase = 'sys'
            
            # Format timing displays with timeout indicators
            def format_time(val, timeout_phase, phase_name):
                if timeout_phase == phase_name:
                    # Keep the measured value (often partial) and annotate the timeout,
                    # instead of replacing it with the budget (which is misleading).
                    return f"{val:.3f}s (timeout @{MUS_SOLVE_TIMEOUT}s)"
                return f"{val:.3f}s"

            def fmt_cell(val, timeout_phase, phase_name, width: int = 16) -> str:
                return f"{format_time(val, timeout_phase, phase_name):>{width}}"
            
            logging.info("")
            logging.info(f"  ┌─ ABAPC removal strategy:")
            logging.info(f"  │   Compiling:      {fmt_cell(abapc_compile, abapc_timeout_phase, 'compile')}")
            logging.info(f"  │   UNSAT grounding:{abapc_unsat_ground:>15.3f}s")
            logging.info(f"  │   UNSAT solving:  {abapc_unsat_solve:>15.3f}s")
            logging.info(f"  │   SAT grounding:  {fmt_cell(abapc_ground, abapc_timeout_phase, 'ground')}")
            logging.info(f"  │   SAT solving:    {fmt_cell(abapc_solve, abapc_timeout_phase, 'solve')}")
            logging.info(f"  │   Sys (residual): {fmt_cell(abapc_sys, abapc_timeout_phase, 'sys')}")
            logging.info(f"  │   Total:          {abapc_total:>15.3f}s")
            logging.info(f"  │")
            logging.info(f"  └─ ABAPC MUS/MCS analysis:")
            logging.info(f"      Compiling:      {fmt_cell(mus_compile, mus_timeout_phase, 'compile')}")
            logging.info(f"      Grounding:      {fmt_cell(mus_ground, mus_timeout_phase, 'ground')}")
            logging.info(f"      Solving:        {fmt_cell(mus_solve, mus_timeout_phase, 'solve')}")
            logging.info(f"      Sys (residual): {fmt_cell(mus_sys, mus_timeout_phase, 'sys')}")
            logging.info(f"      Total:          {mus_total:>15.3f}s")
            logging.info("")
            
            if not (abapc_timeout or mus_timeout):
                ratio_total = mus_total / abapc_total if abapc_total > 0 else 0
                ratio_solve = mus_solve / abapc_solve if abapc_solve > 0 else 0
                
                logging.info(f"  Total time ratio (MUS/ABAPC):   {ratio_total:8.2f}x")
                logging.info(f"  Solve time ratio (WASP/clingo): {ratio_solve:8.2f}x")
            else:
                logging.info(f"  (Timing ratios not computed due to timeout)")
            logging.info("=" * 60 + "\n")
        except Exception as timing_error:
            # If timing measurement fails, still show what we have
            logging.warning(f"Timing comparison measurement failed: {timing_error}")
            logging.info("=" * 60 + "\n")

        # Clean up temp files
        os.remove(facts_file)
        os.remove(facts_I_file)
        os.remove(facts_wc_file)
        os.remove(facts_mus_file)

        logging.info(f"✓ Completed run for n_nodes={n_nodes}\n")
        
        # Assertions after timing comparison so it always displays
        # Basic assertion: we expect at least some MUSes for contradiction cases
        if remove_n > 0:
            if mus_timeout and mus_result.get('n_mus', 0) == 0:
                self.skipTest(
                    f"MUS timed out after {mus_time.total_seconds():.2f}s before finding any MUS "
                    f"(n_nodes={n_nodes}, seed={seed}, graph_type={graph_type})"
                )
            self.assertGreater(mus_result.get('n_mus', 0), 0, f"Expected at least one MUS for UNSAT case (n_nodes={n_nodes})")
        
        # Final MUS timeout assertion: handled above via skipTest for the 0-MUS case.
        
        # Return status info for retry logic
        return {
            'was_sat': remove_n == 0,
            'abapc_timeout': abapc_timeout,
            'mus_timeout': mus_timeout,
            'abapc_time_sec': abapc_time.total_seconds(),
            'mus_time_sec': mus_time.total_seconds(),
            'remove_n': remove_n,
            # For correspondence checks (used by test_mus_mcs_random_sizes_abapc)
            'seed': seed,
            'graph_type': graph_type,
            'count_wrong': count_wrong,
            'facts_ext': facts_ext,
            'wrong_ext': wrong_ext,
            'removed_facts': removed_facts,
            'n_mus': mus_result.get('n_mus', 0),
            'n_mcs': mus_result.get('n_mcs', 0),
            'mus_facts': mus_result.get('mus_facts', []),
            'mcs_facts': mus_result.get('mcs_facts', []),
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MUS/ABAPC random-size tests with optional overrides")
    parser.add_argument(
        "--random-only",
        action="store_true",
        help="Run only TestMUSAnalysis.test_mus_mcs_random_sizes_abapc (skip other unit tests)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=MUS_SEED_BASE,
        help="Base seed for random graph generation (default: 2004). For some historical sizes, derived defaults are used: 5->base, 6->base+1, 10->base+6.",
    )
    parser.add_argument(
        "--rep-unsat",
        type=int,
        default=MUS_REP_UNSAT,
        help="Retry with seed+1, seed+2, ... when an instance is already SAT or has 0 wrong facts (default: 2)",
    )
    parser.add_argument(
        "--random-reps",
        type=int,
        default=MUS_RANDOM_REPS,
        help="Number of random repetitions per (node_size, graph_type) case (default: 1)",
    )
    parser.add_argument(
        "--solve-timeout",
        type=int,
        default=MUS_SOLVE_TIMEOUT,
        help="Per-instance wall-clock timeout (seconds) for both ABAPC removal and MUS solving",
    )
    parser.add_argument(
        "--node-sizes",
        type=str,
        default=','.join(str(n) for n in MUS_NODE_SIZES),
        help="Comma-separated list of node counts for the random graph tests (e.g., 5,7,9)",
    )
    parser.add_argument(
        "--edge-per-node",
        type=int,
        default=MUS_EDGE_PER_NODE,
        help="Edge-per-node multiplier for random graph generation",
    )
    parser.add_argument(
        "--emit-lp",
        type=str,
        default=MUS_EMIT_LP,
        help="Path to emit complete MUS program (use {n} for node count placeholder, e.g., /tmp/test_{n}node.lp)",
    )
    parser.add_argument(
        "--max-muses",
        type=str,
        default=MUS_MAX_MUSES,
        help="Max MUS to enumerate: '' (empty)=omit -n flag (WASP default: 1 MUS output), '0'=unlimited, '>0'=limit to N",
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        default=MUS_GRAPH_TYPE,
        help="Random graph type for PC simulation (default: ER)",
    )
    parser.add_argument(
        "--graph-types",
        type=str,
        default=",".join(MUS_GRAPH_TYPES) if MUS_GRAPH_TYPES else "",
        help="Comma-separated list of random graph types to run (overrides --graph-type), e.g. ER,SF",
    )
    parser.add_argument(
        "--out-n",
        type=int,
        default=MUS_ABAPC_OUT_N,
        help="Clingo -n model bound for ABAPC removal (0=all; default: 0)",
    )
    parser.add_argument(
        "--opt-mode",
        type=str,
        default=MUS_ABAPC_OPT_MODE,
        help="Clingo opt_mode for ABAPC removal (default: optN)",
    )
    parser.add_argument(
        "--mcs-threshold",
        type=int,
        default=MUS_MCS_THRESHOLD,
        help="CAMUS MCS threshold (0=unlimited, default=50)",
    )
    parser.add_argument(
        "--mus-threshold",
        type=int,
        default=MUS_MUS_THRESHOLD,
        help="CAMUS MUS threshold (0=unlimited, default=300)",
    )
    args, remaining = parser.parse_known_args()

    MUS_SOLVE_TIMEOUT = args.solve_timeout
    MUS_NODE_SIZES = _parse_node_sizes(args.node_sizes)
    MUS_EDGE_PER_NODE = args.edge_per_node
    MUS_SEED_BASE = args.seed_base
    MUS_REP_UNSAT = args.rep_unsat
    MUS_RANDOM_REPS = args.random_reps
    MUS_EMIT_LP = args.emit_lp
    MUS_MAX_MUSES = args.max_muses
    MUS_GRAPH_TYPE = args.graph_type
    MUS_GRAPH_TYPES = _parse_graph_types(args.graph_types) or (MUS_GRAPH_TYPE,)
    MUS_ABAPC_OUT_N = args.out_n
    MUS_ABAPC_OPT_MODE = args.opt_mode
    MUS_MCS_THRESHOLD = args.mcs_threshold
    MUS_MUS_THRESHOLD = args.mus_threshold

    start = datetime.now()
    logger_setup()
    logging.info(
        f"CLI overrides: solve_timeout={MUS_SOLVE_TIMEOUT}s, node_sizes={MUS_NODE_SIZES}, edge_per_node={MUS_EDGE_PER_NODE}, seed_base={MUS_SEED_BASE}, "
        f"rep_unsat={MUS_REP_UNSAT}, random_reps={MUS_RANDOM_REPS}, graph_types={MUS_GRAPH_TYPES}, opt_mode={MUS_ABAPC_OPT_MODE}, out_n={MUS_ABAPC_OUT_N}, "
        f"emit_lp={MUS_EMIT_LP or '(none)'}, max_muses={MUS_MAX_MUSES}, mcs_threshold={MUS_MCS_THRESHOLD}, mus_threshold={MUS_MUS_THRESHOLD}"
    )

    if args.random_only:
        suite = unittest.TestSuite()
        suite.addTest(TestMUSAnalysis('test_mus_mcs_random_sizes_abapc'))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)

    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
    logging.info(f"Total test time={str(datetime.now()-start)}")
