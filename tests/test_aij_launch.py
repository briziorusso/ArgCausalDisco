import sys
import unittest
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_matched_baseline_experiments import selected_seeds, _assert_metadata_compatible
from scripts.run_aij_baselines import commands, SEEDS, evaluate_completed, VERSIONS


class LaunchTests(unittest.TestCase):
    def test_explicit_nonconsecutive_seeds_and_legacy_default(self):
        self.assertEqual(selected_seeds(SimpleNamespace(seeds=SEEDS)), SEEDS)
        self.assertEqual(selected_seeds(SimpleNamespace(seed_start=2026, n_runs=3)), [2026, 2027, 2028])
        with self.assertRaises(ValueError):
            selected_seeds(SimpleNamespace(seeds=[1, 1]))

    def test_launch_and_audit_use_identical_seed_identities(self):
        launch, audit = commands(SimpleNamespace(method="aspcr",datasets=["cancer","earthquake","survey"],results_dir=Path('results'), aspcr_r_dir=Path('/aspcr/R'),
                                                rscript='Rscript', clingo_bin_dir=None))
        for command in (launch, audit):
            start = command.index('--seeds') + 1
            self.assertEqual(command[start:start + 10], list(map(str, SEEDS)))
        self.assertIn('--resume', launch)
        self.assertIn('aspcr_log_dag', launch)

    def test_resume_rejects_different_seed_cohort(self):
        with self.assertRaises(SystemExit):
            _assert_metadata_compatible({'seeds': [1, 2]}, {'seeds': [1, 3]})

    def test_completed_rerun_is_evaluated_and_changed_input_is_rejected(self):
        from scripts.run_matched_baseline_experiments import FGS_BDEU_CONFIG
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);version=VERSIONS['fgs']
            progress=root/'progress'/version;progress.mkdir(parents=True)
            truth=np.array([[0,1],[0,0]],dtype=int)
            info=dict(dataset='cancer',seed=17,method='fgs_bdeu',sample_size=5000,
                      sample_sha256='sample',fgs_bdeu=FGS_BDEU_CONFIG)
            graph=root/'run.npz'
            np.savez(graph,W_est=np.array([[0,-1],[-1,0]]),B_true=truth,metadata=np.array(json.dumps(info)))
            row=dict(seed=17,status='ok',elapsed=.2,raw_graph_path=str(graph))
            for kind in ('dag','cpdag'):pd.DataFrame([row]).to_csv(progress/f'cancer__fgs_bdeu_{kind}.csv',index=False)
            (root/f'metadata_{version}.json').write_text(json.dumps({'graph_metric_protocol':'aij_v2'}))
            inputs=[dict(dataset='cancer',seed=17,sample_sha256='sample',
                         truth_matrix_sha256=hashlib.sha256(np.asarray(truth,dtype='<i8').tobytes()).hexdigest())]
            args=SimpleNamespace(results_dir=root,method='fgs',datasets=['cancer'])
            with patch('scripts.run_aij_baselines.SEEDS',[17]):
                evaluate_completed(args,inputs)
                records=pd.read_csv(root/'aij_evaluation/reruns'/version/'records.csv')
                cp=records[records.kind.eq('cpdag')].iloc[0]
                self.assertEqual((cp.F1,cp.sid_low,cp.sid_high),(1,0,2))
                inputs[0]['sample_sha256']='changed'
                with self.assertRaisesRegex(ValueError,'Input hash mismatch'):evaluate_completed(args,inputs)


if __name__=='__main__':unittest.main()
