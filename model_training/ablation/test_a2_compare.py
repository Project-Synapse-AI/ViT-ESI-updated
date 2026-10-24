"""Regression checks using real archived control/A2 output files."""
import argparse
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import a2_compare as comparison
from a2_smoke_audit import audit
from a4_compare import METRICS, SETS, load_run


class ComparisonChecks(unittest.TestCase):
    def test_cross_source_review(self):
        self.assertTrue(comparison.source_audit()["train_eval_ast_equal_after_removing_only_A2_options"])
        for filename in ("model_training/models/vit.py",):
            self.assertEqual(comparison.git("show", f"{comparison.CONTROL_SOURCE}:{filename}"),
                             comparison.git("show", f"13364c1:{filename}"))

    def test_changed_training_math_is_rejected(self):
        original = comparison.git
        def changed(*args):
            value = original(*args)
            if args == ("show", f"{comparison.VARIANT_SOURCE}:model_training/main_train.py"):
                value = value.replace(b"lr = 1e-3", b"lr = 2e-3")
            return value
        with patch.object(comparison, "git", changed):
            with self.assertRaisesRegex(ValueError, "Non-plumbing change"):
                comparison.source_audit()

    def test_pairing_and_explicit_options(self):
        comparison.paired_compatible(self.control, self.variant)
        for field, value, message in (("vit_mixer", "window", "mixer mismatch"),
                                      ("vit_window_radius", 7, "Unplanned averaging window"),
                                      ("eeg_snr", 10, "Unmatched argument eeg_snr")):
            variant = copy.deepcopy(self.variant)
            variant["train"]["args"][field] = value
            with self.assertRaisesRegex(ValueError, message):
                comparison.paired_compatible(self.control, variant)

    def test_wrong_source_or_depth_rejected(self):
        for field, value in (("depth", 3), ("source", "wrong")):
            variant = copy.deepcopy(self.variant)
            if field == "depth": variant["depth"] = value
            else: variant["train"]["git_commit"] = value
            with self.assertRaises(ValueError):
                comparison.paired_compatible(self.control, variant)

    def test_corrupt_selected_weight_claim_rejected(self):
        with patch("a2_smoke_audit.digest", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "Selected weight hash mismatch"):
                audit(self.variant["root"], "mean", full=True)

    def test_eval_noise_and_sample_metadata_rejected(self):
        variant = copy.deepcopy(self.variant)
        variant["eval"][SETS[1]]["summary"]["args"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "Unmatched evaluation seed"):
            comparison.paired_compatible(self.control, variant)
        variant = copy.deepcopy(self.variant)
        variant["eval"][SETS[1]]["rows"][0]["peak_time_ms"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Unpaired sample metadata"):
            comparison.paired_compatible(self.control, variant)

    def test_all_metrics_and_paired_differences(self):
        records = comparison.compare(self.control, [self.variant])
        self.assertEqual(len(records), 20)
        for dataset in SETS:
            delta = self.variant["eval"][dataset]["values"] - self.control["eval"][dataset]["values"]
            rows = [r for r in records if r["mixer"] == "mean" and r["dataset"] == dataset]
            self.assertEqual([r["metric"] for r in rows], list(METRICS))
            np.testing.assert_allclose([r["delta"] for r in rows], delta.mean(axis=0), atol=1e-12)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--mean", type=Path, required=True)
    args = p.parse_args()
    ComparisonChecks.control = load_run(args.control)
    ComparisonChecks.variant = comparison.load_variant(args.mean, "mean")
    unittest.main(argv=[__file__])
