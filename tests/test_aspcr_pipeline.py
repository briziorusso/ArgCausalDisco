from __future__ import annotations

import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_matched_baseline_experiments as runner


class ASPCRArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.data_path = self.data_dir / "data_cancer_2026.csv"
        pd.DataFrame(np.zeros((4, 5))).to_csv(self.data_path, index=False, header=False)
        self.paths = runner._aspcr_output_paths(
            self.data_path,
            algorithm="log-weights",
            model_space="dag_sufficient",
        )
        self.paths["directed"].parent.mkdir(parents=True)
        self.W = np.zeros((5, 5), dtype=int)
        self.W[0, 1] = 1
        self.W[1, 2] = 1
        self.W[0, 3] = 1
        self._write_valid_artifacts()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _constraint_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        index = 0
        for x in range(1, 6):
            for y in range(x + 1, 6):
                available = [node for node in range(1, 6) if node not in {x, y}]
                for size in range(len(available) + 1):
                    for conditioning in itertools.combinations(available, size):
                        index += 1
                        tested = index % 2 == 0
                        truth = tested if index % 3 else not tested
                        final = tested if index % 5 else not tested
                        correct = tested == truth
                        retained = tested == final
                        rows.append({
                            "constraint_id": index,
                            "x": x,
                            "y": y,
                            "conditioning_set": ";".join(map(str, conditioning)),
                            "intervention_set": "",
                            "cset": 0,
                            "jset": 0,
                            "mset": 0,
                            "test_independent": tested,
                            "truth_independent": truth,
                            "final_independent": final,
                            "tested_relation": "independent" if tested else "dependent",
                            "truth_relation": "independent" if truth else "dependent",
                            "final_graph_relation": "independent" if final else "dependent",
                            "test_correct": correct,
                            "retained": retained,
                            "retained_true": retained and correct,
                            "retained_false": retained and not correct,
                            "raw_weight": index / 1000,
                            "asp_weight": index,
                            "probability_independent": 0.5,
                        })
        return rows

    def _write_valid_artifacts(self) -> None:
        zero = np.zeros((5, 5), dtype=int)
        pd.DataFrame(self.W.T).to_csv(self.paths["directed"], index=False, header=False)
        pd.DataFrame(zero).to_csv(self.paths["bidirected"], index=False, header=False)
        pd.DataFrame(zero).to_csv(self.paths["tailtail"], index=False, header=False)
        constraints = pd.DataFrame(self._constraint_rows())
        constraints.to_csv(self.paths["constraints"], index=False)
        correct = constraints["test_correct"].astype(bool)
        retained = constraints["retained"].astype(bool)
        retained_true = retained & correct
        retained_false = retained & ~correct
        fact_true = int(correct.sum())
        fact_false = int((~correct).sum())
        fact_retained = int(retained.sum())
        fact_retained_true = int(retained_true.sum())
        fact_retained_false = int(retained_false.sum())
        precision = fact_retained_true / fact_retained
        recall = fact_retained_true / fact_true
        f1 = 2 * precision * recall / (precision + recall)
        objective = int(constraints.loc[~retained, "asp_weight"].sum())
        diagnostics = pd.DataFrame([{
            "algorithm": "log-weights",
            "model_space": "dag_sufficient",
            "encoding": "new_wmaxsat_acyclic_sufficient.pl",
            "test": "bayes",
            "weight": "log",
            "prior_independence": 0.4,
            "alpha": 20.0,
            "expected_constraints": 80,
            "fact_total": 80,
            "fact_true": fact_true,
            "fact_false": fact_false,
            "fact_retained": fact_retained,
            "fact_retained_true": fact_retained_true,
            "fact_retained_false": fact_retained_false,
            "fact_removed": 80 - fact_retained,
            "fact_removed_true": fact_true - fact_retained_true,
            "fact_removed_false": fact_false - fact_retained_false,
            "fact_precision": precision,
            "fact_recall": recall,
            "fact_f1": f1,
            "solver_objective": objective,
            "recomputed_objective": objective,
            "graph_is_dag": True,
            "n_directed": int(self.W.sum()),
            "n_bidirected": 0,
            "n_tailtail": 0,
            "elapsed_total": 1.25,
            "testing_time": 0.5,
            "encoding_time": 0.25,
            "solving_time": 0.5,
        }])
        diagnostics.to_csv(self.paths["diagnostics"], index=False)
        pd.DataFrame([[1.25]]).to_csv(self.paths["time"], index=False, header=False)

    def test_orientation_constraint_accounting_and_objective(self) -> None:
        result = runner.load_and_validate_aspcr_outputs(
            self.data_path,
            algorithm="log-weights",
            model_space="dag_sufficient",
            n_nodes=5,
        )
        np.testing.assert_array_equal(result.W_est, self.W)
        self.assertTrue(result.diagnostics["graph_is_dag"])
        self.assertEqual(result.diagnostics["fact_total"], 80)
        self.assertEqual(result.diagnostics["solver_objective"], result.diagnostics["recomputed_objective"])

    def test_bidirected_edge_is_rejected_in_dag_mode(self) -> None:
        bidirected = np.zeros((5, 5), dtype=int)
        bidirected[0, 1] = bidirected[1, 0] = 1
        pd.DataFrame(bidirected).to_csv(self.paths["bidirected"], index=False, header=False)
        with self.assertRaisesRegex(RuntimeError, "bidirected"):
            runner.load_and_validate_aspcr_outputs(
                self.data_path,
                algorithm="log-weights",
                model_space="dag_sufficient",
                n_nodes=5,
            )

    def test_cycle_is_rejected_in_dag_mode(self) -> None:
        cyclic = self.W.copy()
        cyclic[2, 0] = 1
        pd.DataFrame(cyclic.T).to_csv(self.paths["directed"], index=False, header=False)
        with self.assertRaisesRegex(RuntimeError, "cycle"):
            runner.load_and_validate_aspcr_outputs(
                self.data_path,
                algorithm="log-weights",
                model_space="dag_sufficient",
                n_nodes=5,
            )

    def test_objective_mismatch_is_rejected(self) -> None:
        diagnostics = pd.read_csv(self.paths["diagnostics"])
        diagnostics.loc[0, "solver_objective"] += 1
        diagnostics.to_csv(self.paths["diagnostics"], index=False)
        with self.assertRaisesRegex(RuntimeError, "solver_objective"):
            runner.load_and_validate_aspcr_outputs(
                self.data_path,
                algorithm="log-weights",
                model_space="dag_sufficient",
                n_nodes=5,
            )

    def test_native_bayesian_configuration_mismatch_is_rejected(self) -> None:
        diagnostics = pd.read_csv(self.paths["diagnostics"])
        diagnostics.loc[0, "prior_independence"] = 0.5
        diagnostics.to_csv(self.paths["diagnostics"], index=False)
        with self.assertRaisesRegex(RuntimeError, "native prior_independence"):
            runner.load_and_validate_aspcr_outputs(
                self.data_path,
                algorithm="log-weights",
                model_space="dag_sufficient",
                n_nodes=5,
            )


class ResumeAndCountingTests(unittest.TestCase):
    def test_constraint_counts(self) -> None:
        self.assertEqual(runner.expected_constraint_count(5), 80)
        self.assertEqual(runner.expected_constraint_count(6), 240)

    def test_resume_uses_successful_seed_identity(self) -> None:
        dag = pd.DataFrame([
            {"seed": 2026, "status": "ok"},
            {"seed": 2027, "status": "error"},
            {"seed": 2029, "status": "ok"},
        ])
        cpdag = pd.DataFrame([
            {"seed": 2026, "status": "ok"},
            {"seed": 2027, "status": "ok"},
            {"seed": 2029, "status": "ok"},
        ])
        self.assertEqual(runner._successful_seed_ids(dag, cpdag), {2026, 2029})

    def test_r_d_separation_and_orientation_helpers(self) -> None:
        rscript = runner.DEFAULT_RSCRIPT
        if not rscript.exists():
            self.skipTest(f"Rscript is unavailable: {rscript}")
        aspcr_r_dir_value = os.environ.get("ASPCR_R_DIR")
        if not aspcr_r_dir_value:
            self.skipTest("ASPCR_R_DIR is not set")
        directed_reachable = Path(aspcr_r_dir_value).expanduser() / "directed_reachable.R"
        if not directed_reachable.exists():
            self.skipTest(f"ASPCR helper is unavailable: {directed_reachable}")
        wrapper = runner.REPO_ROOT / "cd_algorithms" / "run_aspcr_csvdata.R"
        expression = f"""
source({str(directed_reachable)!r})
source({str(wrapper)!r})
empty <- matrix(0, 3, 3)
chain <- list(G=empty, Ge=empty, Gs=empty)
chain$G[2,1] <- 1
chain$G[3,2] <- 1
stopifnot(directed_reachable(1,3,c(),c(),chain))
stopifnot(!directed_reachable(1,3,c(2),c(),chain))
stopifnot(aspcr_is_dag(chain$G))
cycle <- chain
cycle$G[1,3] <- 1
stopifnot(!aspcr_is_dag(cycle$G))
collider <- list(G=empty, Ge=empty, Gs=empty)
collider$G[2,1] <- 1
collider$G[2,3] <- 1
stopifnot(!directed_reachable(1,3,c(),c(),collider))
stopifnot(directed_reachable(1,3,c(2),c(),collider))
"""
        completed = subprocess.run(
            [str(rscript), "--vanilla", "-e", expression],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
