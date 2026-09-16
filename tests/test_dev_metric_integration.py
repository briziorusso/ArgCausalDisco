import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from utils.aij_graph_metrics import decode_estimate
from utils.graph_evaluation import evaluate_estimate
from utils.graph_eval import graph_eval_from_accepted


class DevelopmentMetricTests(unittest.TestCase):
    def test_mpc_llm_endpoint_encoding(self):
        raw = np.array([[0, 1, 0], [-1, 0, -1], [0, -1, 0]])
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(decode_estimate(raw, 'mpc_llm'), expected)

    def test_invalid_mpc_retains_edges_without_sid_or_dag(self):
        cycle = np.array([[0, 1, 0, 1], [1, 0, 1, 0],
                          [0, 1, 0, 1], [1, 0, 1, 0]])
        truth = np.zeros((4, 4), dtype=int)
        truth[0, 1] = truth[1, 2] = truth[2, 3] = 1
        result = evaluate_estimate(-cycle, truth, 'mpc')
        self.assertIsNone(result['dag'])
        np.testing.assert_array_equal(result['cpdag'], cycle)
        self.assertAlmostEqual(result['cpdag_metrics']['F1'], 6 / 7)
        self.assertTrue(np.isnan(result['cpdag_metrics']['sid']).all())
        self.assertTrue(np.isnan(result['dag_metrics']['F1']))
        self.assertEqual(result['cpdag_status']['sid']['status'], 'no_consistent_extension')

    def test_prior_orientations_are_retained_without_claiming_cpdag_bounds(self):
        # a -> b -- c and a -- c: a valid background-knowledge PDAG.
        raw = np.array([[0, 1, -1], [-1, 0, -1], [-1, -1, 0]])
        truth = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        result = evaluate_estimate(raw, truth, 'mpc_llm', seed=7)
        self.assertEqual(result['dag'][0, 1], 1)
        np.testing.assert_array_equal(result['cpdag'], decode_estimate(raw, 'mpc_llm'))
        self.assertTrue(np.isfinite(result['dag_metrics']['sid']))
        self.assertTrue(np.isnan(result['cpdag_metrics']['sid']).all())
        self.assertEqual(result['cpdag_status']['sid']['status'], 'noncompleted_pdag')

    def test_backend_failure_is_not_silently_recorded_as_a_bad_graph(self):
        truth = np.array([[0, 1], [0, 0]])
        with patch('utils.aij_graph_metrics.exact_sid_bounds', side_effect=ImportError('gadjid')):
            with self.assertRaisesRegex(ImportError, 'gadjid'):
                evaluate_estimate(truth, truth, 'abapc')

    def test_completed_output_has_exact_bounds(self):
        truth = np.array([[0, 1], [0, 0]])
        result = evaluate_estimate(np.array([[0, -1], [-1, 0]]), truth, 'mpc')
        self.assertEqual(result['cpdag_metrics']['sid'], (0, 2))
        self.assertEqual(result['cpdag_metrics']['F1'], 1)

    def test_sweep_uses_shared_f1_for_empty_and_directed_estimates(self):
        # Feed actual clingo models to the sweep callback without grounding a
        # causal program: this tests aggregation and CPDAG caller semantics.
        import clingo
        control = clingo.Control(['0'])
        control.add('base', [], '{pick}. arrow(0,1) :- pick. arrow(2,1) :- pick. #show arrow/2.')
        control.ground([('base', [])])
        truth = nx.DiGraph([(0, 1), (1, 2)])
        with patch('causalaba.compile_and_ground', return_value=control):
            scores = graph_eval_from_accepted(
                n_nodes=3, G_true1=truth, keys_in_file_order=['var(0)'],
                accepted_keys={'var(0)'}, timeout_sec=10, threads=1, cache={},
                include_extrema=True,
            )
        self.assertEqual(len(scores), 29)
        self.assertEqual(scores[0], 2)  # empty DAG and collider
        self.assertAlmostEqual(scores[2], .25)  # mean of 0 and 1/2, including empty DAG
        self.assertEqual(scores[6], 2)
        self.assertEqual(scores[9], 0)  # collider edges do not match undirected edges
        self.assertEqual(scores[10], .5)  # skeleton matches for only the collider
        self.assertFalse(scores[-1])


if __name__ == '__main__':
    unittest.main()
