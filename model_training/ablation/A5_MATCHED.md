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

## Blocker and resource gate

`count_flops.py` supports thop, torchinfo or fvcore, but local import discovery
found none of these installed. FLOPs reporting is required by issue9. The repo's
hard-stop rule applies: do not silently omit FLOPs, accept incomplete zero counts,
or change training dependencies to bypass this missing tool. Obtain user approval
for installing a counting dependency and verify its operation coverage before
reporting CNN/LSTM/attention counts. CPU training/evaluation is not an alternative.

Kaggle quota observed2026-10-25:6.43GPUh remaining, refresh2026-10-31UTC.
Use actual smoke epoch timing plus setup/evaluation allowance to schedule full
jobs. Do not assume both100epoch jobs fit, shrink the scientific budget, change
precision/workers/batch, or add seeds without a separate decision.

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
