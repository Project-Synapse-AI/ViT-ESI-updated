"""Audit A2 full runs and paired five-metric comparisons with the A4 control."""
import argparse
import ast
import copy
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from a2_smoke_audit import SOURCE as VARIANT_SOURCE, audit
from a4_compare import (METRICS, SETS, SOURCE as CONTROL_SOURCE, compatible,
                        digest, interval, load_run, require)

ROOT = Path(__file__).resolve().parents[2]
NOTICE = ("Preliminary single-training-seed results. Paired 95% percentile bootstrap intervals "
          "resample evaluation samples (5000 draws, seed 0), not training runs; no multiplicity "
          "adjustment. Held-out data selected checkpoints. Only test2k_s1 is independent. "
          "Fixed mixers also remove attention projections/weight dropout and reduce capacity. "
          "Further training-set/seed and real-data verification is needed.")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


class RemoveA2Options(ast.NodeTransformer):
    """Remove only reviewed A2 configuration plumbing from train/eval ASTs."""
    def visit_Expr(self, node):
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "add_argument" and value.args
                and isinstance(value.args[0], ast.Constant)
                and value.args[0].value in ("-vit_mixer", "-vit_window_radius")):
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        keep = [i for i, k in enumerate(node.keys)
                if not (isinstance(k, ast.Constant) and k.value in ("mixer", "window_radius"))]
        node.keys = [node.keys[i] for i in keep]
        node.values = [node.values[i] for i in keep]
        return self.generic_visit(node)

    def visit_Call(self, node):
        node.keywords = [k for k in node.keywords if k.arg not in ("mixer", "window_radius")]
        return self.generic_visit(node)


def source_audit():
    allowed = {"model_training/models/vit.py", "model_training/main_train.py",
               "model_training/ablation/evaluate.py"}
    changed = git("diff", "--name-only", CONTROL_SOURCE, VARIANT_SOURCE, "--",
                  "model_training", "data_generation", "utils", "anatomy", "kaggle").decode().splitlines()
    for path in changed:
        if path in allowed or path == "kaggle/push_job.py":
            continue
        if path.startswith("model_training/ablation/") and path not in (
                "model_training/ablation/baseline_workflow.py",
                "model_training/ablation/build_model_folder.py",
                "model_training/ablation/requirements-lock.txt"):
            continue  # analysis/docs/tests are not invoked in scientific jobs
        require(False, f"Unreviewed scientific source change: {path}")
    for path in ("model_training/main_train.py", "model_training/ablation/evaluate.py"):
        before = ast.parse(git("show", f"{CONTROL_SOURCE}:{path}").decode())
        after = RemoveA2Options().visit(ast.parse(git("show", f"{VARIANT_SOURCE}:{path}").decode()))
        require(ast.dump(before) == ast.dump(after), f"Non-plumbing change in {path}")
    # The architecture intervention/default-preservation checks are independently
    # exercised by test_a2_averaging.py against the committed pre-A2 implementation.
    return dict(control_source=CONTROL_SOURCE, variant_source=VARIANT_SOURCE,
                changed_scientific_files=sorted(allowed),
                train_eval_ast_equal_after_removing_only_A2_options=True,
                default_attention_cuda_parity_test="test_a2_averaging.py",
                metric_loader_generation_anatomy_profile_changes=False)


def load_variant(root, mixer):
    report = audit(root, mixer, full=True)
    root = root.resolve()
    job = json.loads((root / "job_manifest.json").read_text())
    train = json.loads(next(root.glob("training/**/run_manifest.json")).read_text())
    evaluations = {}
    for dataset in SETS:
        p = root / "eval" / dataset
        with (p / "per_sample.csv").open(newline="") as f:
            rows = sorted(csv.DictReader(f), key=lambda r: int(r["index"]))
        evaluations[dataset] = dict(rows=rows, values=np.array([[float(r[m]) for m in METRICS] for r in rows]),
                                    summary=json.loads((p / "summary.json").read_text()),
                                    csv_sha256=digest(p / "per_sample.csv"))
    return dict(root=root, job=job, train=train, depth=6, mixer=mixer, eval=evaluations, audit=report)


def paired_compatible(control, variant):
    require(control["depth"] == variant["depth"] == 6, "A2 requires six-layer models")
    require(control["train"]["git_commit"] == CONTROL_SOURCE, "Wrong control source")
    require(variant["train"]["git_commit"] == VARIANT_SOURCE, "Wrong variant source")
    mixer = variant["mixer"]
    require(mixer in ("mean", "window"), "Unexpected variant mixer")
    normalized = copy.deepcopy(variant)
    for args in [normalized["train"]["args"], *[normalized["eval"][d]["summary"]["args"] for d in SETS]]:
        require(args.pop("vit_mixer") == mixer, "Train/eval mixer mismatch")
        require(args.pop("vit_window_radius") == 5, "Unplanned averaging window")
    compatible(control, normalized)  # all remaining settings/runtime/packages/sample metadata must match


def compare(control, variants):
    require(len({v["mixer"] for v in variants}) == len(variants), "Duplicate mixer")
    records = []
    for run in [control, *variants]:
        if run is not control:
            paired_compatible(control, run)
        label = run.get("mixer", "attention")
        for dataset in SETS:
            values = run["eval"][dataset]["values"]
            delta = values - control["eval"][dataset]["values"]
            ci, dci = interval(values, 5000, 0), interval(delta, 5000, 0)
            for j, metric in enumerate(METRICS):
                records.append(dict(mixer=label, parameters=run["train"]["parameter_count"],
                                    dataset=dataset, n=len(values), metric=metric,
                                    mean=float(values[:, j].mean()), ci_low=float(ci[0, j]), ci_high=float(ci[1, j]),
                                    delta=float(delta[:, j].mean()), delta_ci_low=float(dci[0, j]), delta_ci_high=float(dci[1, j])))
    return records


def write_outputs(runs, records, provenance, out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    report = dict(notice=NOTICE, source_audit=provenance, analysis_commit=git("rev-parse", "HEAD").decode().strip(),
                  script_sha256=digest(Path(__file__)),
                  auditor_sha256=digest(Path(__file__).with_name("a2_smoke_audit.py")),
                  runs=[dict(mixer=r.get("mixer", "attention"), source=r["train"]["git_commit"],
                             environment=r["train"]["environment"],
                             package_inventory_sha256=digest(r["root"] / "pip_freeze.txt"),
                             checkpoint=r["train"]["checkpoints"]["validation_loss"],
                             csv_sha256={d:r["eval"][d]["csv_sha256"] for d in SETS}) for r in runs], records=records)
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Fixed-averaging comparison", "", NOTICE, "",
             "| Mixer | Set | Metric | Mean [95% interval] | Delta vs attention [95% interval] |",
             "|---|---|---|---|---|"]
    for r in records:
        lines.append(f"| {r['mixer']} | {r['dataset']} | {r['metric']} | {r['mean']:.6g} "
                     f"[{r['ci_low']:.6g}, {r['ci_high']:.6g}] | {r['delta']:+.6g} "
                     f"[{r['delta_ci_low']:+.6g}, {r['delta_ci_high']:+.6g}] |")
    lines += ["", "## Provenance and host limitation", ""]
    for r in report["runs"]:
        lines.append(f"- {r['mixer']}: source `{r['source']}`, selected weights "
                     f"`{r['checkpoint']['weights_sha256']}`, host `{r['environment']['platform']}`, "
                     f"package inventory `{r['package_inventory_sha256']}`.")
    lines += ["", "Control and variants use different reviewed sources. Training/evaluation AST changes "
              "are restricted to mixer options; data generation/loaders/metric math/anatomy/profile are unchanged. "
              "Host Linux kernel releases may differ (architecture/glibc and all other recorded runtime fields "
              "and packages match). Host environments are not identical; runtime comparisons are limited."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, ylabel, name in (("LE_mm", "Localization error (mm)", "localization"),
                                ("time_err_ms", "Temporal error (ms)", "temporal_error")):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, dataset in zip(axes, SETS):
            rs = [r for r in records if r["dataset"] == dataset and r["metric"] == metric]
            y = np.array([r["mean"] for r in rs]);x=np.arange(len(rs))
            ax.errorbar(x, y, yerr=[y-[r["ci_low"] for r in rs], [r["ci_high"] for r in rs]-y], fmt="o", capsize=4)
            ax.set_xticks(x, [r["mixer"] for r in rs]); ax.set(title=dataset, ylabel=ylabel)
            ax.grid(alpha=.2)
        fig.suptitle("Fixed averaging: seed 0, evaluation-sample 95% intervals")
        fig.tight_layout()
        for ext in ("png", "pdf"): fig.savefig(out / f"{name}.{ext}", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--mean", type=Path, required=True)
    p.add_argument("--window", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    provenance = source_audit()
    control = load_run(args.control)
    variants = [load_variant(args.mean, "mean"), load_variant(args.window, "window")]
    records = compare(control, variants)
    write_outputs([control, *variants], records, provenance, args.out)
    for run in variants:
        (run["root"].parents[1] / "audit.json").write_text(json.dumps(run["audit"], indent=2) + "\n")
    print("Audited attention control and both fixed-averaging full runs; wrote", args.out)
