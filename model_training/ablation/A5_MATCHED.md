# A5: parameter-matched CNN and LSTM (preparation)

Tracking: issue #9. Initial seed0 scope is a CNN and LSTM close to the
3,565,538-parameter original six-layer ViTESI architecture. No A5 training
has been submitted. This note records sizing and the current tooling blocker;
it is not a frozen or completed experiment record.

## Architecture choices verified from actual modules

Scale width, preserving each baseline's depth, operations and dropout:

| Model | Configuration | Actual trainable parameters | Difference vs ViTESI |
|---|---|---:|---:|
| ViTESI reference | original depth6, width256, heads8, MLP512 | 3,565,538 | 0% |
| CNN | channels75 -> 2604 -> 994, kernel5, no biases | 3,564,876 | -0.0186% |
| LSTM | hidden295, two bidirectional layers, dropout0.2, output biasFalse | 3,557,700 | -0.2198% |

Counts were verified using the existing model constructors. CNN width2605
gives3,566,245 parameters, further from the target. LSTM hidden294/296 gives
3,536,232/3,579,232 respectively, so295 is closest among these integer choices.
No validation/test results were used to choose widths. These are initial
parameter-count choices, not claims of compute matching or final outcomes.

CNN already exposes `-inter_layer`. LSTM training/evaluation currently hard-code
hidden85; add a backward-compatible hidden-size flag with default85. Propagate
it to A3's patch-timing harness to enable the planned recurrent follow-up after
the matched LSTM is trained. Do not change the default architectures or metric
formulas.

## Resolved profiling dependency and resource gate

The user authorized installation: `pip install --no-deps torchinfo==1.8.0` in
the local venv. No training-package version changed; PyTorch remains2.11.0+cu128.
The dependency blocker is resolved. Coverage review found that torchinfo's raw
MAC totals omit sequence-length factors for timepoint-wise linear layers and
functional attention products. They must not be treated as complete FLOPs.

`count_flops.py --a5_matched --device cuda --out <json>` now reports an explicit
dense matrix/convolution convention alongside torchinfo's diagnostic raw totals
and independent parameter/output-shape checks. Matrix FLOP estimate=2*MACs;
excludes bias adds, normalization, activations, softmax, recurrent pointwise
gates and data movement consistently. Includes both LSTM directions/layers,
all timepoints, and both attention score/value products. This is an operation
estimate, not a hardware-throughput or inference-time measurement.

| Model | Dense matrix MACs/sample | Matrix FLOP estimate/sample |
|---|---:|---:|
| ViTESI original6-layer | 2,477,696,000 | 4,955,392,000 |
| CNN width2604 | 1,782,438,000 | 3,564,876,000 |
| LSTM hidden295 | 1,774,130,000 | 3,548,260,000 |

Four CUDA/model-count tests cover nearest-width targets, default LSTM parity,
strict weight compatibility, matched forward/backward shapes/finite gradients,
and independent matrix-count derivations with batch/token scaling. LSTMhidden
default85 is preserved in training/evaluation/A3; new flag `-lstm_hidden_size`.

Local Windows assertions pass, but the process exits0xC0000409 after LSTM
backpropagation (also isolated at the production500timepoints). CNN backward
and the other isolated checks exit normally. Do not count the combined local
test run as passed. Kaggle is the prespecified training platform, not a CPU or
package-version fallback. Require `--a5-checks` on each A5 notebook: it runs
the same CUDA test suite inside Kaggle before any scientific stages and stops
the job on a nonzero exit. Full training remains gated by successful Linux CUDA
checks and real-data smoke audits. No training/runtime package is changed to
suppress the Windows failure.

Kaggle quota observed2026-10-25:6.43GPUh remaining, refresh2026-10-31UTC.
Use actual smoke epoch timing plus setup/evaluation allowance to schedule full
jobs. Do not assume both100epoch jobs fit, shrink the scientific budget, change
precision/workers/batch, or add seeds without a separate decision.
Prioritize the matched LSTM if only one full run fits, because it also unlocks
the recurrent timing comparison in issue5. Defer a second full job if the quota
cannot cover it; do not submit a knowingly underfunded100-epoch run.

## Intended protocol after the blocker is resolved

Frozen train8k_s0,6000/2000split,seed0,100epochs,batch8,Adam0.001,cosine/linear,
5dB,validation-selected checkpoints,noearlystop. Locked4workers,pinning and
persistence,prefetch1,oneT4,FP32. Real frozen-data Kaggle smoke before full jobs.
Both2000-sample evaluations retain LE,AUC,nMSE,PSNR and temporal error (ms),
with shared CUDA harness and noise0+index. Match against the A4six-layer control
only after explicit source/runtime/settings/sample compatibility audits; do not
mix in the earlier local checkpoint. Matching parameters does not match compute.

Archive checkpoints/results/figures on mounted Drive with read-back SHA checks,
keep local originals, and label all results as preliminary single-training-seed
evidence pending other training sets/seeds and real-data verification.
