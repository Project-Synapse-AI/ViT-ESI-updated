# A4 methods draft

This describes the locked initial sweep. Results and interpretation remain pending
until the planned depths have completed and passed audit.

We vary the ViTESI Transformer encoder depth over 1, 2, 3, 4 and 6 layers,
with depth 6 as the matched control. All models use 256-dimensional embeddings,
8 attention heads per layer, a 512-dimensional feed-forward layer and dropout
0.1. Inputs contain 75 EEG channels and 500 timepoints; outputs contain source
activity in 994 cortical regions. Parameter counts are 930,018; 1,457,122;
1,984,226; 2,511,330; and 3,565,538 respectively. Depth changes parameter count
as well as the number of attention layers, so this experiment alone cannot
separate the effects of attention and capacity.

Training uses the fingerprint-verified frozen SEREEGA dataset `train8k_s0`
(8,000 samples), with the fixed seed-0 split into 6,000 training and 2,000
validation samples. Every depth is trained for 100 epochs using Adam with
learning rate 0.001, cosine loss, linear scaling, batch size 8 and 5 dB EEG
noise. Early stopping is disabled. We select the checkpoint with the lowest
validation loss, retaining the training-loss-selected and last checkpoints for
provenance and recovery. The initial sweep uses one training seed, 0.

All runs use the same Kaggle source commit
`2f4068eefb2e90345d3332bf856db130bfc402e1` and the locked profile
`a4-workers4-fp32-v1`: four seeded persistent data-loader workers, pinned
memory, prefetch factor 1, one visible Tesla T4 and FP32 precision. The profile
SHA-256 is `793187c9c72b312bf7a324b50a3a2a17a1e45fe7cbc51fa910f24ba1fd3259eb`.
We use a newly trained depth-6 control because worker settings change the noise
streams relative to the historical workers-0 baseline. A two-epoch real-data
smoke test checks each shallow architecture before its full run.

We evaluate each validation-selected model on both the 2,000 held-out samples
from `train8k_s0` and the independent frozen dataset `test2k_s1` (2,000 samples).
The held-out set was used for checkpoint selection; only the independent set
provides an assessment on samples unused for model selection. Evaluation uses
the unchanged shared CUDA harness and five metrics: localization error (mm),
AUC, normalized mean squared error, PSNR (dB), and temporal error (ms). Noise is
seeded by 0 plus the original sample index to pair conditions across depths.
No independent-test result is used to select a run or tune settings.

For each dataset and metric, we report the sample mean and the paired mean
difference (variant minus depth 6). We calculate descriptive 95% percentile
bootstrap intervals from 5,000 resamples of the 2,000 sample rows, with
replacement and analysis RNG seed 0. The same sampled row indices are used
across metrics and depths; differences are formed before resampling. Lower
LE, nMSE and time error are preferable; higher AUC and PSNR are preferable.
Intervals quantify evaluation-sample uncertainty conditional on the trained
checkpoints. They do not estimate variation across training seeds and are not
adjusted for multiple comparisons. We do not use them as confirmatory tests
of superiority or equivalence. We plot LE against both depth and parameter
count and interpret this sweep alongside A1/A2/A5.

Before accepting a comparison, the analysis checks source/profile provenance,
successful fingerprint stages, 100-epoch completion, matching training splits
and settings, downloaded checkpoint hashes, selected-weight provenance,
2,000 unique paired evaluation indices, finite five-metric rows, and agreement
between CSV means and evaluation summaries. Generated tables, plots and JSON
provenance stay in the ignored results directory and are archived separately.

Kaggle changed its host Linux kernel from 6.18.48 to 6.18.55 between the
depth6/1 and depth2/3 runs. The analysis permits Linux kernel-release differences
only while requiring matching architecture/glibc, all other recorded training
runtime fields, and the full recorded package inventory. It retains each platform
and package-inventory hash in JSON and Markdown reports. Host environments are
therefore not identical; disclose this limitation, especially for timing claims.
No scientific setting, package version or metric formula is changed by this
analysis policy.
