"""CUDA checks for A2 mathematics and backward-compatible default behaviour.

These are deterministic unit inputs, not simulated scientific-run results.
"""
import subprocess
import sys
import types
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model_training"))
from models.vit import EEGViT, EEGViTpl, FixedAveragingLayer


class AveragingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("HARD STOP: CUDA is required for these checks")
        cls.old = types.ModuleType("pre_a2_vit")
        source = subprocess.check_output(
            ["git", "show", "13364c1:model_training/models/vit.py"], cwd=ROOT, text=True)
        exec(compile(source, "pre_a2_vit.py", "exec"), cls.old.__dict__)

    def test_default_attention_matches_original_in_eval_and_train(self):
        for training in (False, True):
            torch.manual_seed(0)
            original = self.old.EEGViT(75, 994, n_times=17).cuda().train(training)
            torch.manual_seed(0)
            current = EEGViT(75, 994, n_times=17).cuda().train(training)
            self.assertEqual(list(original.state_dict()), list(current.state_dict()))
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, current.state_dict()[key]), key)
            current.load_state_dict(original.state_dict(), strict=True)
            x = torch.linspace(-1, 1, 75 * 17, device="cuda").reshape(1, 75, 17)
            torch.manual_seed(8)
            with torch.no_grad():
                expected = original(x)
            torch.manual_seed(8)
            with torch.no_grad():
                actual = current(x)
            self.assertTrue(torch.equal(expected, actual))

    def test_shared_initialization_and_parameter_counts(self):
        torch.manual_seed(0)
        original = EEGViT(75, 994)
        self.assertEqual(sum(p.numel() for p in original.parameters()), 3565538)
        for mixer in ("mean", "window"):
            torch.manual_seed(0)
            model = EEGViT(75, 994, mixer=mixer)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 1986530)
            self.assertFalse(any("self_attn" in name for name in model.state_dict()))
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, original.state_dict()[key]), key)

    def test_mixing_boundaries_and_feature_independence(self):
        layer = torch.nn.TransformerEncoderLayer(2, 1, batch_first=True, norm_first=True)
        x = torch.tensor([[[1., 10.], [2., 20.], [4., 40.], [8., 80.]]], device="cuda")
        mean = FixedAveragingLayer(layer, "mean", 1).cuda()
        torch.testing.assert_close(mean.mix(x), torch.tensor([[[3.75, 37.5]]], device="cuda").expand_as(x))
        window = FixedAveragingLayer(layer, "window", 1).cuda()
        expected = torch.tensor([[[1.5, 15.], [7./3, 70./3], [14./3, 140./3], [6., 60.]]], device="cuda")
        torch.testing.assert_close(window.mix(x), expected)
        torch.testing.assert_close(window.mix(torch.ones_like(x)), torch.ones_like(x))
        identity = FixedAveragingLayer(layer, "window", 0).cuda()
        torch.testing.assert_close(identity.mix(x), x)

    def test_forward_backward_and_strict_serialization(self):
        for mixer in ("mean", "window"):
            model = EEGViTpl(75, 994, n_times=17, depth=2, mixer=mixer).cuda()
            x = torch.linspace(-1, 1, 75 * 17, device="cuda").reshape(1, 75, 17)
            y = model(x)
            self.assertEqual(y.shape, (1, 994, 17))
            y.square().mean().backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            clone = EEGViTpl(75, 994, n_times=17, depth=2, mixer=mixer).cuda()
            clone.load_state_dict(model.state_dict(), strict=True)

    def test_invalid_mixer_and_window_rejected(self):
        for options in ({"mixer": "unknown"}, {"window_radius": -1},
                        {"mixer": "window", "window_radius": 17}):
            with self.assertRaises(ValueError):
                EEGViT(75, 994, n_times=17, **options)


if __name__ == "__main__":
    unittest.main()
