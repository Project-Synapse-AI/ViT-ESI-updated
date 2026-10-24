# Fixed-averaging ablation: completed initial seed-0 comparison

Both six-layer fixed-mixing alternatives completed100epochs and passed full-output audits against the matched learned-attention control. These are preliminary single-training-seed findings, not final outcomes. Verification with different training sets/seeds and real EEG data remains necessary.

## Intervention and reference

Scientific variant source: `a999ec5f3c838cce2574edc50ecd2e36f5f456bb`; attention control source: `2f4068eefb2e90345d3332bf856db130bfc402e1`; analysis: `02cdeb3`. Source/profile/splits/settings, checkpoint hashes, sample pairing and runtime/package compatibility passed audit.

The control is the retrained original six-layer architecture used in the depth sweep (heldout/test LE7.03977/6.96316mm). It is distinct from the earlier local checkpoint7.14785/7.10056mm used for attention-removal/timing analyses.

Global mean broadcasts the mean of all pre-normalized tokens to each timepoint. The local alternative uses an equally weighted window of up to five timepoints either side, with boundary renormalization. Both retain six layers, input/output projections, positional embeddings, pre-LayerNorms, residuals, feed-forward blocks and branch-output dropout. Attention projections and attention-weight dropout are removed. No feed-forward widening or window tuning was performed. Both fixed models have1,986,530parameters versus3,565,538:44.28%fewer, so capacity remains a confound. This is not a pure test of learned attention weights alone.

## Protocol

Frozen train8k_s0,6000training/2000validation,seed0,100epochs,batch8,Adam0.001,cosine loss,linear scaling,5dB,validation-selected weights,noearlystop. Production profile: workers4,pinned/persistent,prefetch1,onevisibleT4,FP32; SHA-256 `793187c9c72b312bf7a324b50a3a2a17a1e45fe7cbc51fa910f24ba1fd3259eb`. Both2000-sample evaluations use the unchanged CUDA five-metric harness and noise0+index. No independent-test result was used to tune settings or select a run.

## Five-metric means

Lower localization error,nMSE and temporal error are preferable; higher AUC and PSNR are preferable. Temporal error is an accuracy metric in milliseconds, distinct from runtime.

### train8k_s0_heldout (2000 samples)

| Mixer | LE mm | AUC | nMSE | PSNR dB | Temporal error ms |
|---|---:|---:|---:|---:|---:|
| Learned attention | 7.03977 | 0.820640 | 0.00471633 | 35.4371 | 11.40958 |
| Global mean | 7.65001 | 0.847318 | 0.00576419 | 33.4378 | 15.92792 |
| Local window (+/-5) | 7.14634 | 0.817517 | 0.00486528 | 33.6832 | 13.17875 |

### test2k_s1 (2000 samples)

| Mixer | LE mm | AUC | nMSE | PSNR dB | Temporal error ms |
|---|---:|---:|---:|---:|---:|
| Learned attention | 6.96316 | 0.823210 | 0.00454612 | 35.4284 | 11.81350 |
| Global mean | 7.57099 | 0.851743 | 0.00555794 | 33.4565 | 14.89058 |
| Local window (+/-5) | 6.93076 | 0.822912 | 0.00467976 | 33.6707 | 11.64717 |

## Paired independent-test deltas versus attention

| Mixer | Metric | Delta | Descriptive95%interval |
|---|---|---:|---|
| Global mean | LE_mm | +0.607828 | [+0.414998, +0.815771] |
| Global mean | AUC | +0.0285334 | [+0.0261981, +0.030875] |
| Global mean | nMSE | +0.00101183 | [+0.000861649, +0.00116295] |
| Global mean | PSNR | -1.97181 | [-2.07235, -1.87406] |
| Global mean | time_err_ms | +3.07708 | [+0.88395, +5.37256] |
| Local window (+/-5) | LE_mm | -0.0324065 | [-0.229537, +0.175548] |
| Local window (+/-5) | AUC | -0.000297153 | [-0.00228912, +0.00165902] |
| Local window (+/-5) | nMSE | +0.000133647 | [-1.1841e-06, +0.000269874] |
| Local window (+/-5) | PSNR | -1.75762 | [-1.86583, -1.64604] |
| Local window (+/-5) | time_err_ms | -0.166333 | [-1.92425, +1.55791] |

Intervals use5000paired evaluation-sample bootstrap resamples,analysis RNGseed0. They condition on these trained checkpoints, do not estimate training-seed uncertainty, and are not multiplicity-adjusted. Held-out data selected weights; independent test2k_s1 did not.

## Interpretation

- Global averaging increased independent-test LE by0.60783mm (interval[0.41500,0.81577]) and temporal error by3.07708ms ([0.88395,5.37256]). nMSE increased and PSNR fell by1.97181dB; AUC improved. The metrics therefore show a trade-off, not a uniform effect.
- Local averaging had LE6.93076 versus6.96316mm and temporal error11.64717 versus11.81350ms. Paired intervals span zero for both LE (delta-0.03241mm,[-0.22954,0.17555]) and temporal error (delta-0.16633ms,[-1.92425,1.55791]). These observations do not prove equivalence or superiority.
- Despite close local-averaging LE/timing, PSNR was lower by1.75762dB ([-1.86583,-1.64604]). Similar localization error does not establish unchanged source-signal reconstruction quality.
- This setting suggests that a simple local mixer may retain useful localization/timing performance after retraining. It does not establish that attention is unnecessary generally or isolate learned mixing from the capacity/projection/dropout changes.
- Different training sets/seeds and real-data verification remain necessary. Real-data ablations follow stable validation, consistent preprocessing and compatible electrode/leadfield/ground-truth procedures.

## Observed training duration and provenance

| Mixer | Training seconds | Selected epoch (1-based) | Validation-selected weights SHA-256 |
|---|---:|---:|---|
| Learned attention | 15957.178 | 100 | `532cbeee32130b0c6145c42f676177f06aca509e4ecf5eef3ae09ac764997ca6` |
| Global mean | 16697.451 | 98 | `937cb21bb7977e32b9ece7da9ea42e008cbcb13c4fff112f6b4130e068497f9e` |
| Local window (+/-5) | 17613.011 | 93 | `d31a33ef8884e55bb92f260c4e5b0d76d08038e6e44058d90541ff4b2d67f013` |

No training-speed gain was demonstrated. Host Linux kernel differs (control6.18.48,variants6.18.55), while architecture/glibc,Python,PyTorch,CUDA,Lightning,precision/determinism fields and full package inventories match. The host difference is disclosed and especially limits runtime comparisons. No inference-speed or FLOPs benchmark is claimed.

Cross-source review: training/evaluation ASTs differ only in the reviewed mixer/radius option plumbing. Scientific loader/data generation/metric math/anatomy/profile code is unchanged. Default attention state keys, initialized tensors and train/eval outputs were bit-identical to pre-A2 under fixed RNG in five CUDA checks; shared non-attention initialization matches too. Seven real-output comparison regression checks passed, including rejection of source/depth/window/noise/sample/hash drift and changed training mathematics. Both original smoke audits still pass after extending the auditor for full outputs.

## Storage and reproduction

Variant notebooks: `dasunwijethunga/a2-mean-workers4-s0` and `dasunwijethunga/a2-window5-workers4-s0`,version1. Raw local outputs: ignored `model_training/results/ablation/a2/{mean,window5}-workers4-s0/download/out/`. Final tables,paired intervals,JSON/CSV provenance and localization/temporal-error PNG/PDF figures: `model_training/results/ablation/a2/comparison-02cdeb3/`.

Mounted Drive root `H:/My Drive/SynapseAI-FYP/`: checkpoints under `checkpoints/ablation-A2/{mean,window5}-workers4-s0/`, raw results under `results/ablation/a2/{mean,window5}-workers4-s0/`, and final analysis under `results/ablation/a2/comparison-02cdeb3/`. Globalmean30files88,835,777bytes; window30files88,834,034bytes; generated analysis7files. All copies passed read-back SHA verification; local originals retained. Remote cloud sync was not independently checked.

See A2_AVERAGING.md for exact full-run commands. Analysis:

```powershell
python model_training/ablation/a2_compare.py --control model_training/results/ablation/a4/depth6-workers4-s0/download/out --mean model_training/results/ablation/a2/mean-workers4-s0/download/out --window model_training/results/ablation/a2/window5-workers4-s0/download/out --out model_training/results/ablation/a2/comparison-02cdeb3
python model_training/ablation/test_a2_compare.py --control model_training/results/ablation/a4/depth6-workers4-s0/download/out --mean model_training/results/ablation/a2/mean-workers4-s0/download/out
```

The initial prespecified seed0 A2 scope is complete. Further seeds/windows/real-data verification extend the study and were not silently added to this run budget.
