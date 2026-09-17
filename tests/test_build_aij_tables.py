"""Protect pairing, missing-value reporting and graph-count conventions."""
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import build_aij_tables as builder
from scripts.build_aij_tables import paired_effect, validate_seeds, count_edges, cell, NotebookTable, noninformative_arrowhead_scores, adjust_bh, best_comparisons, pairwise_result, dataset_cell, test_cell


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

    def test_bh_retains_missing_tests_in_the_family(self):
        values = [.01, .04, .03, .002, np.nan]
        np.testing.assert_allclose(adjust_bh(values), [.025, .05, .05, .01, np.nan], equal_nan=True)
        self.assertTrue(np.isnan(adjust_bh([np.nan, np.nan])).all())

    def test_bh_matches_scipy_including_ties(self):
        from scipy import stats
        if not hasattr(stats, 'false_discovery_control'):
            self.skipTest('SciPy reference implementation requires SciPy >= 1.11')
        values = np.array([.04, .04, .001, .2, .008, 1., 0., np.nan])
        expected = stats.false_discovery_control(np.nan_to_num(values, nan=1.), method='bh')
        np.testing.assert_allclose(adjust_bh(values)[:-1], expected[:-1])

    def test_bh_rejects_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            adjust_bh([.1, -0.1])

    def test_dataset_label_uses_true_graph_sizes(self):
        self.assertIn(r'Cancer\\', dataset_cell('cancer'))
        self.assertIn(r'|V|=5$}\\[-1pt]{\scriptsize $|E|=4', dataset_cell('cancer'))
        self.assertIn(r'|V|=20$}\\[-1pt]{\scriptsize $|E|=25', dataset_cell('child'))

    def test_test_cells_keep_missing_tests_distinct_from_nonsignificance(self):
        row = dict(applicable=True, analysis_status='insufficient_pairs', delta_mean=.1, n=1, p_value_bh=np.nan)
        result = test_cell(row, 3)
        self.assertIn('_{1}', result)
        self.assertIn(r'\text{--}', result)
        self.assertNotIn(r'\mathbf', result)
        row.update(p_value_bh=.049999, n=10, analysis_status='available')
        self.assertIn(r'\mathbf{q<0.050}', test_cell(row, 3))
        row.update(applicable=False)
        self.assertEqual(test_cell(row, 3), r'\textnormal{n/a}')

    def test_best_group_is_not_tied_to_abapc_or_chained_through_followers(self):
        # A is best, A--B is nonsignificant, B--C is nonsignificant,
        # but A--C is significant. C must not inherit B's highlight.
        primary = pd.DataFrame(
            [dict(dataset='x', kind='dag', method=m, seed=i, F1=value)
             for m, value in [('A', .9), ('B', .8), ('C', .7)] for i in range(3)])
        tests = pd.DataFrame([
            dict(dataset='x', kind='dag', method=m, reference=r, metric='F1',
                 applicable=True, analysis_status='available', p_value_bh=q,
                 delta_mean=d, delta_std=.1, t_statistic=d/.1, left_mean=l,
                 right_mean=rr, n=3, n_expected=3, excluded_seeds='[]', p_value_unadjusted=q)
            for m, r, q, d, l, rr in [('B', 'A', .2, -.1, .8, .9), ('C', 'A', .02, -.2, .7, .9), ('C', 'B', .2, -.1, .7, .8)]
        ])
        with patch.multiple(builder, DATASETS=('x',), METHODS=('A', 'B', 'C'), TEST_METRICS={'dag': ('F1',)}):
            result = best_comparisons(primary, tests).set_index('method')
        self.assertEqual(result.highlight.to_dict(), {'A': True, 'B': True, 'C': False})
        self.assertEqual(set(result.reference), {'A'})
        reverse = pairwise_result(tests, 'x', 'dag', 'A', 'C', 'F1')
        self.assertAlmostEqual(reverse['delta_mean'], .2)
        self.assertEqual(reverse['p_value_bh'], .02)

    def test_all_pairs_are_corrected_before_choosing_the_best(self):
        primary = pd.DataFrame([dict(dataset='x', kind='dag', method=m, seed=i, nshd=v)
                                for m, vals in [('A', [1., 2., 3.]), ('B', [1., 2., 3.]), ('C', [7., 8., 9.])]
                                for i, v in enumerate(vals)])
        with patch.multiple(builder, DATASETS=('x',), METHODS=('A', 'B', 'C'), TEST_METRICS={'dag': ('nshd',)}):
            tests = builder.make_tests(primary)
            result = best_comparisons(primary, tests).set_index('method')
        self.assertEqual(len(tests), 3)
        self.assertEqual(result.highlight.to_dict(), {'A': True, 'B': True, 'C': False})
        self.assertEqual(result.is_best_mean.to_dict(), {'A': True, 'B': True, 'C': False})

    def test_an_unavailable_comparison_does_not_receive_tie_highlighting(self):
        primary = pd.DataFrame([dict(dataset='x', kind='dag', method=m, seed=i, F1=v)
                                for m, vals in [('A', [.9, .9]), ('B', [.8, np.nan])]
                                for i, v in enumerate(vals)])
        with patch.multiple(builder, DATASETS=('x',), METHODS=('A', 'B'), TEST_METRICS={'dag': ('F1',)}):
            tests = builder.make_tests(primary)
            result = best_comparisons(primary, tests).set_index('method')
        self.assertEqual(result.highlight.to_dict(), {'A': True, 'B': False})

    def test_saved_notebook_table_includes_all_columns(self):
        parser = NotebookTable()
        parser.feed('<table><tr><th></th><th>method</th><th>mean</th></tr><tr><th>0</th><td>CO_MAX</td><td>1.25</td></tr></table>')
        self.assertEqual(parser.rows, [["", "method", "mean"], ["0", "CO_MAX", "1.25"]])


if __name__ == "__main__":
    unittest.main()
