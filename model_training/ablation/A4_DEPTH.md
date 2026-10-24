# A4: encoder depth on the frozen Kaggle protocol

Tracking: issue #7. Initial sweep uses training seed 0 and depths 6 (control),
1, 2, 3 and 4. Additional seeds are outside this initial GPU budget.

The initial sweep is complete: all full runs, paired five-metric audits and
mounted Drive archives passed. See [A4_RESULTS.md](A4_RESULTS.md) for the tables,
temporal-error findings, checkpoint provenance and preliminary interpretation.

## Fixed conditions

- Data: frozen `train8k_s0`, verified against its 8,000-sample fingerprint;
  seed-0 split of 6,000 training / 2,000 validation samples.
- Depth-6 control and all variants: width 256, 8 heads, MLP 512, dropout 0.1;
  only depth changes. Adam 0.001, cosine loss, linear scaling, 5 dB EEG,
  500 timepoints, batch 8, 100 epochs, no early stopping.
- Locked `kaggle/training_profile.json`: 4 seeded persistent workers,
  pinned memory, prefetch factor 1, one visible T4, FP32.
- Select the primary checkpoint by validation loss. Retain comparison
  training-loss weights and the resumable last checkpoint, with progress logs.
- Evaluate both held-out `train8k_s0` 2,000 and independent `test2k_s1` 2,000
  using the shared harness, CUDA, 5 dB and evaluation noise seed 0 + sample index.
  Report LE, AUC, nMSE, PSNR and time error, together with parameter counts.

## Order and commands

Start the matching depth-6 control first. This is required because the worker
change alters training/validation noise streams relative to the historical
workers-0 run (`c8ce93f`, weights `c0b4ad1a...`, 7.34807 / 7.17185 mm).
The historical checkpoint is a diagnostic comparison, not the new A4 control.

```powershell
python kaggle/push_job.py --job full --model VIT --title "a4 depth6 workers4 s0" --push
```

Once the completed control passes the audit below, smoke each remaining
architecture before submitting its full job. Keep train/eval depth flags equal:

```powershell
python kaggle/push_job.py --job smoke --model VIT --train-args="-vit_depth 1" --eval-args="-vit_depth 1" --title "a4 depth1 workers4 smoke" --push
python kaggle/push_job.py --job full --model VIT --train-args="-vit_depth 1" --eval-args="-vit_depth 1" --title "a4 depth1 workers4 s0" --push
```

Repeat for depths 2, 3 and 4 with unique titles. Build without `--push` to
prepare a notebook only. Initial authorization is the matched control plus
this seed-0 depth sweep; do not silently add tuning, seeds or architectures.
Use measured full-epoch timings and actual Kaggle quota to schedule jobs.
The short benchmark's throughput gain is not a whole-run duration guarantee.

## Audit before comparing results

1. All generation/fingerprint/training/evaluation stages pass, with clean source
   and the frozen profile SHA recorded. Exactly one GPU; no distributed run.
2. All 100 epochs finish, split indices/IDs match the control, primary checkpoint
   is validation-selected, and downloaded weights/checkpoint hashes match.
3. Both evaluation CSVs have 2,000 unique expected sample indices, finite five
   metrics and means matching their summaries. Dataset/noise/harness settings
   and evaluation IDs match across depths.
4. Review full calibration relative to historical results as a diagnostic.
   No equivalence margin has been predeclared; do not tune to obtain a preferred
   localization error or select a run using independent-test results. Investigate
   a material anomaly before launching variants.
5. Report paired per-sample deltas/uncertainty against the matched depth-6
   control, label single-training-seed evidence, and plot LE against both depth
   and parameter count. Depth changes both architecture and parameter count;
   interpret together with A1/A2/A5 rather than claiming causal separation here.

## Analysis while training

`A4_METHODS.md` contains the methods draft. `a4_compare.py` analyses saved
outputs without training or changing the evaluation harness:

```powershell
python model_training/ablation/a4_compare.py --control model_training/results/ablation/a4/depth6-workers4-s0/download/out --out model_training/results/ablation/a4/comparison
```

When a full variant finishes and its artifacts have been downloaded, repeat
with `--variant model_training/results/ablation/a4/depth1-workers4-s0/download/out`.
Add `--variant` once per completed depth. Incomplete runs and smoke runs are
rejected. The first control-only report is explicitly labelled as such.
All runs must use the frozen source commit above; analysis-only commits do not
change the source pinned in future Kaggle notebooks (`push_job.py --commit`).

Outputs: `metrics.csv`, `comparison.json` with checkpoint/CSV hashes,
`REPORT.md`, and PNG/PDF plots of LE versus depth and parameters. All five
metrics receive means, 95% sample-bootstrap intervals and paired differences
against depth 6 (5,000 draws, analysis seed 0). These intervals do not measure
training-seed variation. No metric formula is modified.

Run the regression checks on the real archived control:

```powershell
python model_training/ablation/test_a4_compare.py --control model_training/results/ablation/a4/depth6-workers4-s0/download/out
```

## Storage and completion

Source/commands/decisions go in the A4 PR. Generated outputs remain gitignored:
`model_training/results/ablation/a4/<notebook-slug>/` locally. Archive verified
results under `SynapseAI-FYP/results/ablation/a4/` and checkpoints under
`SynapseAI-FYP/checkpoints/ablation-A4/` on mounted Drive, with README/checksums.
Update issue #7, roadmap #1 and the research vault after each stage. Keep the
PR draft and #7 open until the table, figures, audit and archive handoff are done.
