import unittest
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from unittest.mock import Mock, patch
import numpy as np
from scripts.run_fgs_once import categorical_frame


class FGSTests(unittest.TestCase):
    def test_fgs_preserves_endpoints_and_named_column_order(self):
        from scripts.run_fgs_once import tetrad_adjacency
        matrix=tetrad_adjacency(['X10 --> X2','X2 --- X1'],['X1','X2','X10','isolated'])
        self.assertEqual(matrix[2,1],1)
        self.assertEqual(matrix[1,2],0)
        self.assertEqual(matrix[0,1],-1)
        self.assertEqual(matrix[1,0],-1)
        self.assertFalse(matrix[3].any())


    def test_category_relabelling_preserves_partitions(self):
        a=np.array([[0,4],[0,7],[1,4],[1,7]])
        np.testing.assert_array_equal(categorical_frame(a),categorical_frame(a*3-8))
        with self.assertRaises(ValueError):categorical_frame(np.array([[0,np.nan],[1,2]]))

    def test_discrete_score_reaches_tetrad_and_preserves_export(self):
        from scripts.run_fgs_once import main
        fitted=Mock();fitted.getEdges.return_value=['X2 --> X1','X2 --- X3']
        package=ModuleType('pycausal');bridge=ModuleType('pycausal.pycausal')
        bridge.pycausal=Mock(return_value=Mock())
        package.search=Mock();package.search.tetradrunner.return_value=fitted
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);np.save(root/'input.npy',[[1,4,8],[2,4,9],[1,7,8]])
            argv=['run_fgs_once.py','--input',str(root/'input.npy'),'--output',str(root/'graph.npy'),
                  '--meta',str(root/'meta.json'),'--score','bdeu-score']
            with patch.dict(sys.modules,{'pycausal':package,'pycausal.pycausal':bridge}),patch.object(sys,'argv',argv):main()
            self.assertTrue(json.loads((root/'meta.json').read_text())['ok'])
            matrix=np.load(root/'graph.npy')
            self.assertEqual((matrix[1,0],matrix[0,1],matrix[1,2],matrix[2,1]),(1,0,-1,-1))
        options=fitted.run.call_args.kwargs
        self.assertEqual((options['scoreId'],options['dataType'],options['samplePrior'],options['structurePrior']),
                         ('bdeu-score','discrete',15.,1.))
        self.assertEqual(options['dfs'].columns.tolist(),['X1','X2','X3'])


if __name__=='__main__':unittest.main()
