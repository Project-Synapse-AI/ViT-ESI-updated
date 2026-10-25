"""Real-output A5 comparison/audit regression checks (no model training)."""
import argparse
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import a5_compare as comparison
from a4_compare import METRICS, SETS, load_run
from a5_smoke_audit import audit


class MatchedComparisonChecks(unittest.TestCase):
    def test_cross_source_review_and_training_math(self):
        self.assertTrue(comparison.source_audit()['train_eval_ast_equal_after_removing_reviewed_option_plumbing'])
        original=comparison.git
        def changed(*args):
            value=original(*args)
            if args==('show',f'{comparison.SOURCE}:model_training/main_train.py'):
                value=value.replace(b'lr = 1e-3',b'lr = 2e-3')
            return value
        with patch.object(comparison,'git',changed):
            with self.assertRaisesRegex(ValueError,'Non-option training/evaluation change'):
                comparison.source_audit()

    def test_pairing_and_architecture_options(self):
        comparison.paired_compatible(self.control,self.cnn)
        for key,value in [('inter_layer',4096),('kernel_size',3),('lstm_hidden_size',295),('vit_mixer','mean')]:
            variant=copy.deepcopy(self.cnn);variant['train']['args'][key]=value
            with self.assertRaises(ValueError):comparison.paired_compatible(self.control,variant)

    def test_parameter_or_source_changes_rejected(self):
        for key,value in [('parameter_count',5610000),('git_commit','wrong')]:
            variant=copy.deepcopy(self.cnn);variant['train'][key]=value
            with self.assertRaises(ValueError):comparison.paired_compatible(self.control,variant)

    def test_noise_and_sample_metadata_rejected(self):
        variant=copy.deepcopy(self.cnn);variant['eval'][SETS[1]]['summary']['args']['seed']=1
        with self.assertRaisesRegex(ValueError,'Unmatched evaluation seed'):
            comparison.paired_compatible(self.control,variant)
        variant=copy.deepcopy(self.cnn);variant['eval'][SETS[1]]['rows'][0]['n_patch']='wrong'
        with self.assertRaisesRegex(ValueError,'Unpaired sample metadata'):
            comparison.paired_compatible(self.control,variant)

    def test_checkpoint_hash_guard(self):
        with patch('a5_smoke_audit.digest',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'Selected weight hash mismatch'):
                audit(self.cnn['root'],'1DCNN',full=True)

    def test_smoke_is_not_full_result(self):
        with self.assertRaisesRegex(ValueError,'Not the expected passed full'):
            audit(self.cnn['root'].parents[2]/'cnn2604-workers4-smoke/download/out','1DCNN',full=True)

    def test_all_five_metrics_and_paired_deltas(self):
        rows=comparison.compare(self.control,[self.cnn]);self.assertEqual(len(rows),20)
        for dataset in SETS:
            selected=[r for r in rows if r['model']=='1DCNN' and r['dataset']==dataset]
            self.assertEqual([r['metric'] for r in selected],list(METRICS))
            delta=self.cnn['eval'][dataset]['values']-self.control['eval'][dataset]['values']
            np.testing.assert_allclose([r['delta'] for r in selected],delta.mean(axis=0),atol=1e-12)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control',type=Path,required=True)
    p.add_argument('--cnn',type=Path,required=True);args=p.parse_args()
    MatchedComparisonChecks.control=load_run(args.control)
    MatchedComparisonChecks.cnn=comparison.load_variant(args.cnn,'1DCNN')
    unittest.main(argv=[__file__])
