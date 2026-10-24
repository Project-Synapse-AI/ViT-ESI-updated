"""Audit completed A4 outputs and compare paired samples against depth 6.

See A4_DEPTH.md for usage. This only analyses saved metrics; it never runs a model.
"""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

import numpy as np

METRICS = ("LE_mm", "AUC", "nMSE", "PSNR", "time_err_ms")
SETS = ("train8k_s0_heldout", "test2k_s1")
SOURCE = "2f4068eefb2e90345d3332bf856db130bfc402e1"
PROFILE = "793187c9c72b312bf7a324b50a3a2a17a1e45fe7cbc51fa910f24ba1fd3259eb"
NOTICE = ("Single training seed (0). 95% percentile bootstrap intervals resample "
          "evaluation samples, not training runs. Deltas = variant minus depth 6; "
          "negative favours the variant for LE, nMSE and time error; positive for "
          "AUC and PSNR. Intervals are descriptive and not multiplicity adjusted. "
          "Held-out data also selected the checkpoint; test2k_s1 is independent.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root, remote):
    """Resolve recorded Kaggle output paths within a downloaded out directory."""
    parts = PurePosixPath(remote).parts
    require(parts[:4] == ("/", "kaggle", "working", "out"), "Unexpected artifact path")
    path = root.joinpath(*parts[4:]).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes output directory")
    return path


def load_run(root):
    root = root.resolve()
    job = read_json(root / "job_manifest.json")
    require(job["status"] == "passed" and job["args"]["job"] == "full", "Not a passed full job")
    env = job["environment"]
    require(env["git_commit"] == SOURCE and env["git_dirty"] is False, "Unexpected/dirty source")
    require(env["gpu_used"] == ["Tesla T4"] and env["cuda_visible_devices"] == "0", "GPU mismatch")
    require(not job["requirements_problems"], "Requirements check failed")
    require(job["training_profile_sha256"] == PROFILE, "Frozen profile hash mismatch")
    expected_stages = {"install", "build_model_folder", "train_VIT",
                       "generate_train8k_s0", "generate_test2k_s1",
                       "fingerprint_train8k_s0", "fingerprint_test2k_s1",
                       "eval_train8k_s0_heldout", "eval_test2k_s1"}
    require({s["stage"] for s in job["stages"]} == expected_stages, "Missing/unexpected stages")
    require(all(s["returncode"] == 0 for s in job["stages"]), "Failed stage")
    paths = list(root.glob("training/**/run_manifest.json"))
    require(len(paths) == 1, "Expected one training manifest")
    train = read_json(paths[0])
    require(train["status"] == "completed" and train["completed_epochs"] == 100, "Incomplete training")
    require(train["git_commit"] == SOURCE and train["git_dirty"] is False, "Training source mismatch")
    require(train["seed"] == 0 and train["early_stopping_enabled"] is False, "Seed/early stop mismatch")
    require(train["optimizer"] == "Adam" and train["learning_rate"] == .001, "Optimizer mismatch")
    args = train["args"]
    fixed = dict(model="VIT", simu_name="train8k_s0", to_load=8000, per_valid=.25,
                 ep=100, bs=8, loss="cosine", scaler="linear", eeg_snr=5, n_times=500,
                 vit_embed_dim=256, vit_heads=8, vit_mlp_dim=512, vit_dropout=.1,
                 checkpoint_monitor="validation_loss", require_cuda=True,
                 num_workers=4, pin_memory=True, persistent_workers=True,
                 prefetch_factor=1, devices=1, precision="32-true")
    for key, value in fixed.items():
        require(args[key] == value, f"Training setting mismatch: {key}")
    depth = args["vit_depth"]
    require(depth in (1, 2, 3, 4, 6), "Depth outside initial sweep")
    require(train["parameter_count"] == 402914 + 527104 * depth, "Unexpected parameter count")
    for key, n in (("train_indices", 6000), ("heldout_indices", 2000)):
        require(len(train[key]) == len(set(train[key])) == n, f"Invalid {key}")
    require(set(train["train_indices"]).isdisjoint(train["heldout_indices"]), "Split overlap")
    require(set(train["train_indices"] + train["heldout_indices"]) == set(range(8000)), "Split coverage")
    for ckpt in train["checkpoints"].values():
        for key in ("weights", "checkpoint"):
            require(digest(artifact(root, ckpt[key])) == ckpt[key + "_sha256"], f"Bad {key} hash")
    require(set(train["checkpoints"]) == {"validation_loss", "train_loss"}, "Missing checkpoint exports")
    selected = train["checkpoints"]["validation_loss"]
    require(selected["monitor"] == "validation_loss" and 0 <= selected["epoch"] < 100, "Bad checkpoint selection")
    last = list(root.glob("training/**/last.ckpt"))
    require(len(last) == 1 and last[0].stat().st_size > 0, "Missing last checkpoint")
    evaluations = {}
    for dataset in SETS:
        folder = root / "eval" / dataset
        summary = read_json(folder / "summary.json")
        require(summary["ckpt_sha256"] == selected["weights_sha256"], "Evaluation checkpoint mismatch")
        require(summary["git_commit"] == SOURCE and summary["git_dirty"] is False, "Evaluation source mismatch")
        require(summary["n_evaluated"] == 2000, "Incomplete evaluation")
        ea = summary["args"]
        heldout = dataset == SETS[0]
        expected = dict(model="VIT", simu_name="train8k_s0" if heldout else "test2k_s1",
                        to_load=8000 if heldout else 2000, heldout=heldout,
                        per_valid=.25 if heldout else 1., device="cuda", eeg_snr=5,
                        seed=0, legacy_order=False, vit_depth=depth, vit_embed_dim=256,
                        vit_heads=8, vit_mlp_dim=512, vit_dropout=.1)
        for key, value in expected.items():
            require(ea[key] == value, f"Evaluation setting mismatch: {dataset}/{key}")
        with (folder / "per_sample.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [int(r["index"]) for r in rows]
        expected_ids = train["heldout_indices"] if heldout else range(2000)
        require(len(rows) == len(set(ids)) == 2000 and set(ids) == set(expected_ids), "Evaluation IDs mismatch")
        rows.sort(key=lambda r: int(r["index"]))
        require(all(int(r["id"]) == int(r["index"]) + 1 for r in rows), "Sample ID/index mismatch")
        values = np.array([[float(r[m]) for m in METRICS] for r in rows])
        require(np.isfinite(values).all(), "Nonfinite metric")
        for i, metric in enumerate(METRICS):
            require(np.isclose(values[:, i].mean(), summary["metrics"][metric]["mean"], rtol=1e-10, atol=1e-12), "Summary mean mismatch")
        evaluations[dataset] = dict(values=values, rows=rows, summary=summary,
                                    csv_sha256=digest(folder / "per_sample.csv"))
    return dict(root=root, job=job, train=train, depth=depth, eval=evaluations)


def compatible(control, variant):
    for key in ("seed", "train_indices", "heldout_indices", "train_ids", "heldout_ids",
                "heldout_sample_ids", "optimizer", "learning_rate", "noise_type",
                "early_stopping_enabled", "gradient_clip_val", "data_loading", "trainer_options"):
        require(control["train"][key] == variant["train"][key], f"Unmatched training {key}")
    # Kaggle controls the Linux host kernel. Keep every other recorded runtime
    # field exact, retain architecture/glibc checks, and disclose kernel drift.
    for key in control["train"]["environment"].keys() | variant["train"]["environment"].keys():
        a = control["train"]["environment"].get(key)
        b = variant["train"]["environment"].get(key)
        if key == "platform" and a != b:
            normalize = lambda s: re.sub(r"^Linux-[^-]+-", "Linux-<kernel>-", s)
            require(a.startswith("Linux-") and b.startswith("Linux-") and
                    normalize(a) == normalize(b), "Unmatched training platform architecture/glibc")
        else:
            require(a == b, f"Unmatched training environment {key}")
    require(set((control["root"] / "pip_freeze.txt").read_text().splitlines()) ==
            set((variant["root"] / "pip_freeze.txt").read_text().splitlines()), "Unmatched package inventory")
    # Paths may differ between notebooks; every other training option must match.
    ignored = {"vit_depth", "results_path", "root_simu", "simu_folder", "leadfield_mat"}
    for key in control["train"]["args"].keys() | variant["train"]["args"].keys():
        if key not in ignored:
            require(control["train"]["args"].get(key) == variant["train"]["args"].get(key), f"Unmatched argument {key}")
    require(control["job"]["training_profile"] == variant["job"]["training_profile"], "Unmatched profile")
    for dataset in SETS:
        a, b = control["eval"][dataset], variant["eval"][dataset]
        for ra, rb in zip(a["rows"], b["rows"]):
            for key in ("index", "id", "peak_time_ms", "erp_center_ms", "n_patch", "overlap"):
                require(ra[key] == rb[key], f"Unpaired sample metadata: {dataset}/{key}")
        ignored_eval = {"vit_depth", "out", "root_simu", "ckpt", "leadfield_mat"}
        aa, ba = a["summary"]["args"], b["summary"]["args"]
        for key in aa.keys() | ba.keys():
            if key not in ignored_eval:
                require(aa.get(key) == ba.get(key), f"Unmatched evaluation {key}")
        require(a["summary"]["torch"] == b["summary"]["torch"], "Evaluation torch mismatch")


def interval(values, repeats, seed):
    """Resample entire sample rows, retaining metric/variant pairing."""
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, repeats, 100):
        indices = rng.integers(len(values), size=(min(100, repeats - start), len(values)))
        means.append(values[indices].mean(axis=1))
    return np.quantile(np.concatenate(means), [.025, .975], axis=0)


def compare(runs, repeats=5000, seed=0):
    control = runs[0]
    require(control["depth"] == 6, "Control must have depth 6")
    require(len({r["depth"] for r in runs}) == len(runs), "Duplicate depths")
    records = []
    for run in runs:
        compatible(control, run)
        for dataset in SETS:
            values = run["eval"][dataset]["values"]
            delta = values - control["eval"][dataset]["values"]
            ci, dci = interval(values, repeats, seed), interval(delta, repeats, seed)
            for j, metric in enumerate(METRICS):
                records.append(dict(depth=run["depth"], parameters=run["train"]["parameter_count"],
                                    dataset=dataset, n=len(values), metric=metric,
                                    mean=float(values[:, j].mean()), ci_low=float(ci[0, j]), ci_high=float(ci[1, j]),
                                    delta=float(delta[:, j].mean()), delta_ci_low=float(dci[0, j]), delta_ci_high=float(dci[1, j])))
    return records


def write_outputs(runs, records, out, repeats, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = dict(notice=NOTICE, bootstrap_repeats=repeats, bootstrap_seed=seed,
                  analysis_script_sha256=digest(Path(__file__)),
                  analysis_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip(),
                  runs=[dict(depth=r["depth"], source_commit=r["train"]["git_commit"],
                             training_environment=r["train"]["environment"],
                             package_inventory_sha256=digest(r["root"] / "pip_freeze.txt"),
                             profile_sha256=r["job"]["training_profile_sha256"],
                             checkpoint=r["train"]["checkpoints"]["validation_loss"],
                             output=str(r["root"]),
                             csv_sha256={d: r["eval"][d]["csv_sha256"] for d in SETS}) for r in runs],
                  records=records)
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# A4 depth comparison", "", NOTICE, "",
             "| Depth | Params | Dataset | Metric | Mean [95% CI] | Delta vs 6 [95% CI] |",
             "|---|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['depth']} | {r['parameters']} | {r['dataset']} | {r['metric']} | "
                     f"{r['mean']:.6g} [{r['ci_low']:.6g}, {r['ci_high']:.6g}] | "
                     f"{r['delta']:.6g} [{r['delta_ci_low']:.6g}, {r['delta_ci_high']:.6g}] |")
    lines += ["", "## Provenance", ""]
    for run in report["runs"]:
        lines.append(f"- Depth {run['depth']}: source `{run['source_commit']}`, validation-selected weights "
                     f"`{run['checkpoint']['weights_sha256']}`, outputs `{run['output']}`.")
        lines.append(f"  Host platform: `{run['training_environment']['platform']}`; "
                     f"package inventory SHA-256 `{run['package_inventory_sha256']}`.")
    platforms = {r["training_environment"]["platform"] for r in report["runs"]}
    if len(platforms) > 1:
        lines += ["", "Host-kernel variation is present. Recorded runtime fields and package inventories "
                  "match; only the Linux kernel release differs. This is a disclosed limitation, "
                  "especially for timing comparisons; host environments are not identical."]
    if len(runs) == 1:
        lines += ["", "Control only: variants are pending. No depth effect can be inferred yet."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for xkey, label, filename in (("depth", "Encoder depth", "le_vs_depth"),
                                  ("parameters", "Parameters (millions)", "le_vs_parameters")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, dataset in zip(axes, SETS):
            rows = sorted((r for r in records if r["dataset"] == dataset and r["metric"] == "LE_mm"), key=lambda r: r[xkey])
            x = np.array([r[xkey] for r in rows], dtype=float)
            if xkey == "parameters":
                x /= 1e6
            y = np.array([r["mean"] for r in rows])
            ax.errorbar(x, y, yerr=[y - [r["ci_low"] for r in rows], [r["ci_high"] for r in rows] - y],
                        fmt="o", capsize=4, color="#176b87")
            for xx, yy, r in zip(x, y, rows):
                ax.annotate(f"d={r['depth']}", (xx, yy), xytext=(6, 0), textcoords="offset points")
            ax.set(title=dataset, xlabel=label, ylabel="Localization error (mm)")
            if xkey == "depth":
                ax.set_xticks([1, 2, 3, 4, 6])
                ax.set_xlim(.5, 6.5)
            ax.grid(alpha=.2)
        fig.suptitle("A4: seed 0, sample bootstrap 95% CI" + (" (control only)" if len(runs) == 1 else ""))
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{filename}.{suffix}", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True, help="Downloaded depth-6 out directory")
    parser.add_argument("--variant", type=Path, action="append", default=[], help="Completed variant out directory (repeatable)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0, help="Analysis bootstrap RNG only")
    args = parser.parse_args()
    require(args.bootstrap >= 1000, "Use at least 1000 bootstrap draws")
    runs = [load_run(p) for p in [args.control, *args.variant]]
    records = compare(runs, args.bootstrap, args.seed)
    write_outputs(runs, records, args.out, args.bootstrap, args.seed)
    print(f"Audited {len(runs)} completed runs; wrote {args.out}")


if __name__ == "__main__":
    main()
