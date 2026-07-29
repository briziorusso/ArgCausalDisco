from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import collect_matched_baseline_tables as collector


def _aba_row() -> dict[str, object]:
    return {"rep": 1, "solver": "causalaba_increm", "opt_mode": "optN"}


def _opt_row() -> dict[str, object]:
    return {
        "rep": 1,
        "encoding": "inc",
        "objective": "lex",
        "opt_strategy": "bb",
        "opt_mode": "optN",
        "reification": "mus",
    }


class InjectedFactFilterTests(unittest.TestCase):
    def _write_summary(self, root: Path, *, pct_wrong_facts: float | None) -> None:
        run_dir = root / "wc_sweep_5_2026_test"
        run_dir.mkdir(parents=True)
        (run_dir / "summary.json").write_text(json.dumps({
            "seed": 2026,
            "pct_wrong_facts": pct_wrong_facts,
            "baseline_results": [_aba_row()],
            "wc_results": [_opt_row()],
        }))

    def test_injected_facts_are_excluded_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_summary(root, pct_wrong_facts=0.2)
            with warnings.catch_warnings(record=True) as caught:
                rows = list(collector._iter_mcs_method_rows(root))
            self.assertEqual(rows, [])
            self.assertTrue(any("not matched by the MPC baseline" in str(item.message) for item in caught))

    def test_injected_facts_require_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_summary(root, pct_wrong_facts=0.2)
            rows = list(collector._iter_mcs_method_rows(root, allow_injected_wrong_facts=True))
            self.assertEqual({row["method"] for row in rows}, {"ABA-PC", "OptABA-PC"})
            self.assertTrue(all(row["pct_wrong_facts"] == 0.2 for row in rows))

    def test_unmodified_fact_runs_remain_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_summary(root, pct_wrong_facts=None)
            rows = list(collector._iter_mcs_method_rows(root))
            self.assertEqual({row["method"] for row in rows}, {"ABA-PC", "OptABA-PC"})
            self.assertTrue(all(row["pct_wrong_facts"] == 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
