import itertools
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.aij_graph_metrics import *


class CorrectedMetricsTests(unittest.TestCase):
    def test_endpoints_and_one_sided_undirected_are_distinct(self):
        a=np.array([[0,1,0],[-1,0,-1],[0,-1,0]])
        p=adjacency(a,endpoints=True)
        self.assertEqual(graph_sets(p)[1],{(0,1)})
        self.assertEqual(graph_sets(np.array([[0,-1],[0,0]]))[1],set())

    def test_completion_propagates_beyond_colliders(self):
        d=np.zeros((4,4),dtype=int);d[0,2]=d[1,2]=d[2,3]=1
        np.testing.assert_array_equal(complete_cpdag(d),d)

    def test_f1_identity_and_permutation_invariance(self):
        p=np.array([[0,1,0],[1,0,1],[0,1,0]])
        self.assertEqual(structural_scores(p,p)['F1'],1)
        q=p.copy();q[1,0]=0
        order=[2,0,1]
        before=structural_scores(q,p);after=structural_scores(q[np.ix_(order,order)],p[np.ix_(order,order)])
        for key in before:
            self.assertTrue(np.isclose(before[key],after[key],equal_nan=True),key)
        self.assertEqual(structural_scores(np.zeros_like(p),p)['F1'],0)

    def test_nonextendible_pdag_is_rejected(self):
        p=np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
        with self.assertRaises(InvalidGraph): consistent_extension(p)

    def test_exact_sid_against_exhaustive_all_three_node_dags(self):
        dags=[];classes={}
        for states in itertools.product(range(3),repeat=3):
            d=np.zeros((3,3),dtype=np.int8)
            for (i,j),s in zip(itertools.combinations(range(3),2),states):
                if s==1:d[i,j]=1
                if s==2:d[j,i]=1
            try: require_dag(d)
            except InvalidGraph: continue
            dags.append(d);classes.setdefault(complete_cpdag(d).tobytes(),[]).append(d)
        self.assertEqual(len(dags),25)
        for truth in dags:
            for members in classes.values():
                p=complete_cpdag(members[0]);values=[dag_sid(truth,d) for d in members]
                result=exact_sid_bounds(truth,p)
                self.assertEqual((result['sid_low'],result['sid_high']),(min(values),max(values)))

    def test_component_factorization_matches_full_enumeration(self):
        truth=np.zeros((5,5),dtype=np.int8)
        truth[0,1]=truth[1,2]=truth[0,3]=truth[3,4]=1
        p=np.zeros_like(truth);p[0,1]=p[1,0]=p[2,3]=p[3,2]=1
        p[1,4]=p[3,4]=1
        a=exact_sid_bounds(truth,p);b=enumerate_sid_bounds(truth,p)
        self.assertEqual((a['sid_low'],a['sid_high']),(b['sid_low'],b['sid_high']))

    def test_four_node_equivalence_classes_against_exhaustive_enumeration(self):
        dags=[];classes={}
        for states in itertools.product(range(3),repeat=6):
            d=np.zeros((4,4),dtype=np.int8)
            for (i,j),s in zip(itertools.combinations(range(4),2),states):
                if s==1:d[i,j]=1
                if s==2:d[j,i]=1
            try:require_dag(d)
            except InvalidGraph:continue
            dags.append(d);classes.setdefault(complete_cpdag(d).tobytes(),[]).append(d)
        self.assertEqual(len(dags),543)
        for truth in (dags[0],dags[123],dags[-1]):
            for members in classes.values():
                values=[dag_sid(truth,d) for d in members]
                result=exact_sid_bounds(truth,complete_cpdag(members[0]))
                self.assertEqual((result['sid_low'],result['sid_high']),(min(values),max(values)))

    def test_resource_limit_never_returns_an_approximate_extremum(self):
        p=np.ones((4,4),dtype=np.int8)-np.eye(4,dtype=np.int8)
        result=exact_sid_bounds(np.triu(p,1),p,max_states=1)
        self.assertEqual(result['sid_status'],'enumeration_limit')
        self.assertTrue(np.isnan(result['sid_low']) and np.isnan(result['sid_high']))

    def test_wrapper_f1_and_forced_cpdag_reference(self):
        from utils.graph_utils import DAGMetrics, matrix_arrowhead_set
        d=np.zeros((3,3),dtype=int);d[0,1]=d[1,2]=1
        c=complete_cpdag(d)
        scores=DAGMetrics(c,d,sid=False,evaluation_kind='cpdag').metrics
        self.assertEqual(scores['F1'],1)
        self.assertEqual(scores['fpr'],0)
        self.assertEqual(matrix_arrowhead_set(np.array([[0,-1],[0,0]])),set())
        self.assertEqual(DAGMetrics(np.zeros_like(d),d,sid=False,evaluation_kind='cpdag').metrics['F1'],0)
        with self.assertRaises(InvalidGraph):
            DAGMetrics(np.array([[0,1,0],[0,0,1],[1,0,0]]),d,sid=False,evaluation_kind='dag')

if __name__=='__main__':unittest.main()
