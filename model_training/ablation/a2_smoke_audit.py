"""Audit real A2 Kaggle outputs; the CLI remains smoke-only.

Downloads must include selected VIT_model.pt; resumable smoke checkpoints are
not required or claimed as verified. The explicit full=True API also checks all
exported full-run checkpoint hashes and the complete 6000/2000 training split.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from a4_compare import METRICS, PROFILE, artifact, digest, require

SOURCE = "a999ec5f3c838cce2574edc50ecd2e36f5f456bb"


def audit(root, mixer, *, full=False):
    root = root.resolve()
    require(mixer in ("mean", "window"), "Invalid fixed mixer")
    kind, epochs, total, heldout = ("full", 100, 8000, 2000) if full else ("smoke", 2, 100, 25)
    job = json.loads((root / "job_manifest.json").read_text())
    require(job["args"]["job"] == kind and job["status"] == "passed", f"Not a passed {kind}")
    require(job["training_profile_sha256"] == PROFILE, "Wrong profile")
    require(not job["requirements_problems"], "Requirements failed")
    require(job["environment"]["git_commit"] == SOURCE and not job["environment"]["git_dirty"], "Wrong source")
    require(job["environment"]["gpu_used"] == ["Tesla T4"] and
            job["environment"]["cuda_visible_devices"] == "0", "Wrong GPU")
    expected_stages = {"install", "build_model_folder", "train_VIT", "generate_train8k_s0",
                       "generate_test2k_s1", "fingerprint_train8k_s0", "fingerprint_test2k_s1",
                       "eval_train8k_s0_heldout", "eval_test2k_s1"}
    require({s["stage"] for s in job["stages"]} == expected_stages and
            all(s["returncode"] == 0 for s in job["stages"]), "Failed/missing stage")
    paths = list(root.glob("training/**/run_manifest.json"))
    require(len(paths) == 1, "Expected one training manifest")
    train = json.loads(paths[0].read_text())
    require(train["status"] == "completed" and train["completed_epochs"] == epochs, "Incomplete epochs")
    require(train["git_commit"] == SOURCE and not train["git_dirty"], "Training source mismatch")
    require(train["seed"] == 0 and not train["early_stopping_enabled"], "Wrong seed/stopping")
    require(train["optimizer"] == "Adam" and train["learning_rate"] == .001, "Wrong optimizer")
    expected = dict(model="VIT", simu_name="train8k_s0", to_load=total, per_valid=.25,
                    ep=epochs, bs=8, loss="cosine", scaler="linear", eeg_snr=5, n_times=500,
                    vit_depth=6, vit_embed_dim=256, vit_heads=8, vit_mlp_dim=512,
                    vit_dropout=.1, vit_mixer=mixer, vit_window_radius=5,
                    checkpoint_monitor="validation_loss", require_cuda=True,
                    num_workers=4, pin_memory=True, persistent_workers=True,
                    prefetch_factor=1, devices=1, precision="32-true")
    for key, value in expected.items():
        require(train["args"][key] == value, f"Wrong training {key}")
    require(train["parameter_count"] == 1986530, "Wrong parameter count")
    require(len(train["train_indices"]) == len(set(train["train_indices"])) == total-heldout and
            len(train["heldout_indices"]) == len(set(train["heldout_indices"])) == heldout,
            "Wrong split sizes")
    require(set(train["train_indices"]).isdisjoint(train["heldout_indices"]) and
            set(train["train_indices"] + train["heldout_indices"]) == set(range(total)), "Bad split")
    selected = train["checkpoints"]["validation_loss"]
    require(selected["monitor"] == "validation_loss" and 0 <= selected["epoch"] < epochs, "Wrong checkpoint selection")
    require(digest(artifact(root, selected["weights"])) == selected["weights_sha256"], "Selected weight hash mismatch")
    if full:
        require(set(train["checkpoints"]) == {"validation_loss", "train_loss"}, "Missing checkpoint exports")
        for ckpt in train["checkpoints"].values():
            for key in ("weights", "checkpoint"):
                require(digest(artifact(root, ckpt[key])) == ckpt[key + "_sha256"], f"Bad {key} hash")
        last = list(root.glob("training/**/last.ckpt"))
        require(len(last) == 1 and last[0].stat().st_size > 0, "Missing last checkpoint")
    for dataset, count in (("train8k_s0_heldout", heldout), ("test2k_s1", 2000 if full else 100)):
        summary = json.loads((root / "eval" / dataset / "summary.json").read_text())
        require(summary["git_commit"] == SOURCE and not summary["git_dirty"], "Eval source mismatch")
        require(summary["ckpt_sha256"] == selected["weights_sha256"], "Eval used wrong weights")
        require(summary["device"] == "cuda" and summary["n_evaluated"] == count, "Eval size/device mismatch")
        ea = summary["args"]
        is_heldout = dataset == "train8k_s0_heldout"
        for key, value in dict(model="VIT", vit_mixer=mixer, vit_window_radius=5, vit_depth=6,
                               vit_embed_dim=256, vit_heads=8, vit_mlp_dim=512, vit_dropout=.1,
                               to_load=total if is_heldout else count,
                               per_valid=.25 if is_heldout else 1.,
                               simu_name="train8k_s0" if is_heldout else "test2k_s1",
                               eeg_snr=5, seed=0, legacy_order=False).items():
            require(ea[key] == value, f"Eval mismatch {key}")
        require(ea["heldout"] == (dataset == "train8k_s0_heldout"), "Wrong evaluation set")
        with (root / "eval" / dataset / "per_sample.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        ids = [int(r["index"]) for r in rows]
        expected_ids = set(train["heldout_indices"]) if is_heldout else set(range(count))
        require(len(ids) == count and set(ids) == expected_ids, "Wrong sample indices")
        require(all(int(r["id"]) == int(r["index"]) + 1 for r in rows), "Sample ID mismatch")
        for metric in METRICS:
            values = np.array([float(r[metric]) for r in rows])
            require(np.isfinite(values).all() and
                    np.isclose(values.mean(), summary["metrics"][metric]["mean"], rtol=1e-10, atol=1e-12),
                    f"Nonfinite/incorrect summary {metric}")
    report = dict(status="passed", scope="full single-seed run" if full else "smoke only; not a full scientific result",
                  source_commit=SOURCE, mixer=mixer, window_radius=5, epochs=epochs,
                  parameters=1986530, profile_sha256=PROFILE,
                  selected_weights_sha256=selected["weights_sha256"],
                  selected_weight_binary_hash_verified=True,
                  exported_checkpoint_hashes_verified=full,
                  resumable_checkpoint_hashes_verified=False,
                  checks=["all stages/fingerprints", "source/profile/train/eval settings",
                          f"{total-heldout}/{heldout} split", "selected weights",
                          "expected sample IDs and finite five-metric summaries"])
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Downloaded smoke out directory")
    p.add_argument("--mixer", choices=["mean", "window"], required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    report = audit(args.out, args.mixer)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.mixer} smoke audit passed")
