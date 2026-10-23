"""Regression checks using a real downloaded control; no synthetic model data."""

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from a4_compare import SETS, compare, compatible, interval, load_run


class ComparisonChecks(unittest.TestCase):
    control = None

    def test_control_identity(self):
        records = compare([self.control], repeats=1000)
        self.assertEqual(len(records), 10)
        for row in records:
            self.assertEqual((row["delta"], row["delta_ci_low"], row["delta_ci_high"]), (0., 0., 0.))

    def test_duplicate_depth_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate depths"):
            compare([self.control, self.control], repeats=1000)

    def test_paired_intervals_use_row_differences(self):
        # Reorder real measured values in memory to exercise the numerical
        # comparison, without inventing a trained variant or saving fake results.
        variant = copy.deepcopy(self.control)
        variant["depth"] = 1
        for dataset in SETS:
            variant["eval"][dataset]["values"] = variant["eval"][dataset]["values"][::-1].copy()
        records = compare([self.control, variant], repeats=1000)
        for dataset in SETS:
            delta = variant["eval"][dataset]["values"] - self.control["eval"][dataset]["values"]
            expected = interval(delta, 1000, 0)
            rows = [r for r in records if r["depth"] == 1 and r["dataset"] == dataset]
            np.testing.assert_allclose([r["delta"] for r in rows], delta.mean(axis=0), atol=1e-12)
            np.testing.assert_array_equal([r["delta_ci_low"] for r in rows], expected[0])
            np.testing.assert_array_equal([r["delta_ci_high"] for r in rows], expected[1])

    def test_bootstrap_reproducible(self):
        values = self.control["eval"][SETS[1]]["values"]
        a = interval(values, 1000, 0)
        np.testing.assert_array_equal(a, interval(values, 1000, 0))
        # Independent direct resampling verifies batching and percentile axes.
        indices = np.random.default_rng(0).integers(len(values), size=(1000, len(values)))
        expected = np.quantile(values[indices].mean(axis=1), [.025, .975], axis=0)
        np.testing.assert_array_equal(a, expected)

    def test_split_order_change_rejected(self):
        variant = copy.deepcopy(self.control)
        variant["train"]["train_indices"].reverse()
        with self.assertRaisesRegex(ValueError, "Unmatched training train_indices"):
            compatible(self.control, variant)

    def test_training_option_change_rejected(self):
        variant = copy.deepcopy(self.control)
        variant["train"]["args"]["deterministic"] = True
        with self.assertRaisesRegex(ValueError, "Unmatched argument deterministic"):
            compatible(self.control, variant)

    def test_eval_noise_change_rejected(self):
        variant = copy.deepcopy(self.control)
        variant["eval"][SETS[1]]["summary"]["args"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "Unmatched evaluation seed"):
            compatible(self.control, variant)

    def test_sample_metadata_change_rejected(self):
        variant = copy.deepcopy(self.control)
        variant["eval"][SETS[1]]["rows"][0]["id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Unpaired sample metadata"):
            compatible(self.control, variant)

    def test_smoke_and_failed_jobs_rejected(self):
        for status, kind in (("passed", "smoke"), ("failed", "full")):
            with self.subTest(status=status, kind=kind), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                (path / "job_manifest.json").write_text(json.dumps(dict(status=status, args=dict(job=kind))))
                with self.assertRaisesRegex(ValueError, "Not a passed full job"):
                    load_run(path)

    def test_hash_mismatch_rejected(self):
        # Keep real files in place and alter only the claimed digest in memory.
        from unittest.mock import patch
        import a4_compare
        original = a4_compare.read_json

        def changed(path):
            value = original(path)
            if path.name == "run_manifest.json":
                value["checkpoints"]["validation_loss"]["weights_sha256"] = "0" * 64
            return value

        with patch.object(a4_compare, "read_json", changed):
            with self.assertRaisesRegex(ValueError, "Bad weights hash"):
                load_run(self.control["root"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=Path)
    args = parser.parse_args()
    ComparisonChecks.control = load_run(args.control)
    unittest.main(argv=[__file__])
