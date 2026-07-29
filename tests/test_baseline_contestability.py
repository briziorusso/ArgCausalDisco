from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_mpc_test_enforcement import audit_dag_against_facts, summarise_audit
from scripts.build_baseline_contestability_tables import build_abapc_optimality_summary
from scripts.run_aspcr_contestability import (
    _parse_solver_json,
    build_base_program,
    build_constraint_program,
)


class MPCEnforcementAuditTests(unittest.TestCase):
    def test_counts_independence_and_dependence_contradictions(self) -> None:
        dag = np.zeros((3, 3), dtype=int)
        dag[0, 1] = 1
        dag[1, 2] = 1
        facts = [
            (False, 0, 2, frozenset()),       # active chain: satisfied dependence
            (True, 0, 2, frozenset({1})),     # blocked chain: satisfied independence
            (True, 0, 2, frozenset()),        # contradicted independence
            (False, 0, 2, frozenset({1})),    # contradicted dependence
        ]
        counts = audit_dag_against_facts(dag, facts)
        self.assertEqual(counts["total_tests"], 4)
        self.assertEqual(counts["contradicted_tests"], 2)
        self.assertEqual(counts["contradicted_independence_tests"], 1)
        self.assertEqual(counts["contradicted_dependence_tests"], 1)

    def test_invalid_output_counts_as_enforcement_failure_not_imputed_mean(self) -> None:
        records = pd.DataFrame([
            {
                "dataset": "cancer", "seed": 1, "endpoint_valid": True,
                "all_tests_enforced": True, "total_tests": 10,
                "contradicted_tests": 0, "contradiction_rate": 0.0,
                "contradicted_independence_tests": 0,
                "contradicted_dependence_tests": 0,
            },
            {
                "dataset": "cancer", "seed": 2, "endpoint_valid": False,
                "all_tests_enforced": False, "total_tests": 10,
                "contradicted_tests": np.nan, "contradiction_rate": np.nan,
                "contradicted_independence_tests": np.nan,
                "contradicted_dependence_tests": np.nan,
            },
        ])
        summary = summarise_audit(records).iloc[0]
        self.assertEqual(summary["outputs"], 2)
        self.assertEqual(summary["valid_outputs"], 1)
        self.assertEqual(summary["fully_enforcing_outputs"], 1)
        self.assertEqual(summary["enforcement_failure_outputs"], 1)
        self.assertEqual(summary["contradicted_tests_mean_valid"], 0)


class ASPCRContestabilityTests(unittest.TestCase):
    def test_base_program_connects_requested_transformation(self) -> None:
        text = build_base_program(3, [(1, 0, 4)])
        self.assertIn("node(1..3).", text)
        self.assertIn("ismember(4,3).", text)
        self.assertIn("marginalize(0,0,0,3,4).", text)
        self.assertIn("condition(0,1,1,0,4).", text)

    def test_constraint_program_has_relation_and_stable_identifier(self) -> None:
        constraints = pd.DataFrame([
            {
                "constraint_id": 7, "x": 1, "y": 3, "cset": 2,
                "jset": 0, "mset": 4, "asp_weight": 99,
                "test_independent": True,
            },
            {
                "constraint_id": 8, "x": 2, "y": 3, "cset": 0,
                "jset": 0, "mset": 1, "asp_weight": 101,
                "test_independent": False,
            },
        ])
        text = build_constraint_program(constraints)
        self.assertIn("indep(1,3,2,0,4,99).", text)
        self.assertIn("constraint(7,1,3,2,0,4,99).", text)
        self.assertIn("dep(2,3,0,0,1,101).", text)

    def test_solver_json_requires_proved_optimum_for_cost(self) -> None:
        payload = {
            "Call": [{"Witnesses": [{"Value": ["fail(1,2,0,0,4,5)"], "Costs": [5]}]}],
            "Result": "OPTIMUM FOUND",
            "Models": {"Optimum": "yes", "Costs": [5]},
            "Time": {"Total": 0.1},
        }
        result = _parse_solver_json(
            json.dumps(payload), forced_id=3, target_fail="fail(1,3,0,0,2,7)"
        )
        self.assertTrue(result["optimality_proven"])
        self.assertEqual(result["objective_cost"], 5)
        self.assertTrue(result["target_enforced"])
        self.assertEqual(result["failed_fact_count"], 1)


class ABAPCOptimalitySummaryTests(unittest.TestCase):
    def test_compares_heuristic_cost_with_proved_optimum(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "summary.json"
            summary_path.write_text(json.dumps({
                "baseline_results": [
                    {"rep": 1, "weight_total": 100, "weight_accepted": 60},
                    {"rep": 2, "weight_total": 100, "weight_accepted": 75},
                ]
            }))
            manifest = {
                "baseline_instances": [
                    {
                        "analyzable": True, "optimality_proven": True,
                        "dataset_key": "cancer", "summary_path": str(summary_path),
                        "rep": 1, "seed": 2026, "expected_baseline_cost": 25,
                    },
                    {
                        "analyzable": True, "optimality_proven": True,
                        "dataset_key": "cancer", "summary_path": str(summary_path),
                        "rep": 2, "seed": 2027, "expected_baseline_cost": 25,
                    },
                ]
            }
            row = build_abapc_optimality_summary(manifest).iloc[0]
            self.assertEqual(row["proved_pairs"], 2)
            self.assertEqual(row["aba_strictly_suboptimal"], 1)
            self.assertEqual(row["full_trace_enforced"], 0)
            self.assertAlmostEqual(row["relative_cost_reduction_median"], 0.1875)


if __name__ == "__main__":
    unittest.main()
