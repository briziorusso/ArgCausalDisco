"""Protect pairing, missing-value reporting and graph-count conventions."""
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_aij_tables import paired_effect, validate_seeds, count_edges, cell, NotebookTable, noninformative_arrowhead_scores, significance_marker


class TableDataTests(unittest.TestCase):
    def test_pairs_are_aligned_by_seed_not_row_position(self):
        left = pd.DataFrame({"seed": [11, 22, 33], "score": [3., 5., 8.]})
        right = pd.DataFrame({"seed": [33, 11, 22], "score": [7., 1., 2.]})
        result = paired_effect(left, right, "score")
        self.assertAlmostEqual(result["delta_mean"], 2.)
        self.assertAlmostEqual(result["delta_std"], 1.)
        self.assertEqual(result["n"], 3)

    def test_wrong_cohort_is_rejected(self):
        left = pd.DataFrame({"seed": [1, 2], "score": [2., 3.]})
        right = pd.DataFrame({"seed": [1, 3], "score": [2., 3.]})
        with self.assertRaises(ValueError):
            paired_effect(left, right, "score")

    def test_missing_values_are_counted_not_imputed(self):
        left = pd.DataFrame({"seed": [1, 2, 3], "score": [5., np.nan, 7.]})
        right = pd.DataFrame({"seed": [1, 2, 3], "score": [2., 99., 6.]})
        result = paired_effect(left, right, "score")
        self.assertEqual((result["n"], result["n_expected"], result["excluded_seeds"]), (2, 3, "[2]"))
        self.assertEqual(result["delta_mean"], 2.)

    def test_no_usable_pair_produces_no_inference(self):
        left = pd.DataFrame({"seed": [1, 2], "score": [np.nan, np.nan]})
        right = pd.DataFrame({"seed": [1, 2], "score": [1., 2.]})
        result = paired_effect(left, right, "score")
        self.assertEqual(result["n"], 0)
        self.assertTrue(np.isnan(result["p_value_unadjusted"]))

    def test_duplicate_seed_and_incomplete_coverage_fail(self):
        with self.assertRaises(ValueError):
            validate_seeds(pd.DataFrame({"dataset": ["x", "x"], "seed": [1, 1]}), ["dataset"], [1, 2])
        with self.assertRaises(ValueError):
            validate_seeds(pd.DataFrame({"dataset": ["x"], "seed": [1]}), ["dataset"], [1, 2])

    def test_undirected_edge_is_counted_once(self):
        matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]])
        self.assertEqual(count_edges(matrix), (1, 1))
        matrix[0, 1], matrix[1, 0] = -1, 0
        self.assertEqual(count_edges(matrix), (1, 1))

    def test_true_zero_and_missing_value_remain_distinct(self):
        self.assertEqual(cell(0, digits=2), "$0.00$")
        self.assertEqual(cell(np.nan), "--")
        self.assertEqual(cell(-0.00001, digits=3), "$0.000$")
        self.assertEqual(cell(0, applicable=False, bold=True), r"\textnormal{n/a}")
        self.assertEqual(cell(np.nan, applicable=False), r"\textnormal{n/a}")

    def test_not_applicable_depends_on_reference_not_method_score(self):
        sizes = pd.DataFrame({'dataset':['empty_arrows','has_arrows','has_arrows'],
                              'method':['Reference','Reference','MPC'], 'directed':[0,2,0]})
        excluded = noninformative_arrowhead_scores(sizes)
        self.assertEqual(excluded, {('empty_arrows', m) for m in ('arrowhead_F1','arrowhead_precision','arrowhead_recall')})

    def test_highlights_use_adjusted_p_and_metric_direction(self):
        tests=pd.DataFrame([dict(dataset='x',kind='cpdag',reference='MPC',metric='precision',
                                 applicable=True,p_value_holm=.02,delta_mean=.1)])
        self.assertEqual(significance_marker(tests,'x','cpdag','MPC','precision','max'),r'\dagger')
        self.assertEqual(significance_marker(tests,'x','cpdag','MPC','precision','min'),r'\ddagger')
        tests.loc[0,'p_value_holm']=.2
        self.assertEqual(significance_marker(tests,'x','cpdag','MPC','precision','max'),'')
        tests.loc[0,'p_value_holm']=.02;tests.loc[0,'applicable']=False
        self.assertEqual(significance_marker(tests,'x','cpdag','MPC','precision','max'),'')

    def test_saved_notebook_table_includes_all_columns(self):
        parser = NotebookTable()
        parser.feed('<table><tr><th></th><th>method</th><th>mean</th></tr><tr><th>0</th><td>CO_MAX</td><td>1.25</td></tr></table>')
        self.assertEqual(parser.rows, [["", "method", "mean"], ["0", "CO_MAX", "1.25"]])


if __name__ == "__main__":
    unittest.main()
