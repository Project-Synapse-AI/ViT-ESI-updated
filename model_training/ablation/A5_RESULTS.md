# A5 interim results: matched CNN complete; LSTM pending

The parameter-matched CNN completed100epochs and both2000-sample evaluations. Full-output and paired cross-source audits passed. This is a preliminary single-training-seed result, not a final outcome. The LSTM full run remains unsubmitted because of GPU quota; A5 and the dependent recurrent timing comparison remain open.

## Configuration and reference

CNNchannels75/2604/994,kernel5,no biases:3,564,876parameters versus3,565,538 for the original six-layer ViTESI architecture (difference-0.0186%). Width was selected from counts,not validation/test accuracy; baseline CNN operations/depth are unchanged. Matching parameters does not match computation, receptive fields, optimization behaviour or inductive biases.

CNN scientific source `c5706f169bcda3d0693a17ce0d549bbd86660f8e`; control source `2f4068eefb2e90345d3332bf856db130bfc402e1`; analysis `c671696`. Control weights `532cbeee32130b0c6145c42f676177f06aca509e4ecf5eef3ae09ac764997ca6`; CNN weights `7aa658ae216a340b4fbe3c83494edbea41d38d49fdbce2a594857de7b4cffd45`. The control is the matched Kaggle six-layer run,not the earlier local7.15/7.10mm reference.

Frozen train8k_s0,6000training/2000validation,seed0,100epochs,batch8,Adam0.001,cosine/linear,5dB,validation-selected weights,noearlystop. Locked workers4,pinned/persistent,prefetch1,oneT4,FP32; profile SHA `793187c9c72b312bf7a324b50a3a2a17a1e45fe7cbc51fa910f24ba1fd3259eb`. Both evaluation sets use the unchanged five-metric CUDA harness,noise0+index. Mandatory four-test Linux CUDA suite exited0 before scientific stages.

## All five metric means

Lower LE,nMSE and temporal error are preferable; higher AUC and PSNR are preferable. Temporal error measures activity-timing accuracy in milliseconds; training duration is a separate runtime measurement.

### train8k_s0_heldout:2000samples

| Model | LE mm | AUC | nMSE | PSNR dB | Temporal error ms |
|---|---:|---:|---:|---:|---:|
| VIT | 7.03977 | 0.820640 | 0.00471633 | 35.4371 | 11.40958 |
| 1DCNN | 7.20424 | 0.802196 | 0.00710188 | 32.3714 | 20.38850 |

### test2k_s1:2000samples

| Model | LE mm | AUC | nMSE | PSNR dB | Temporal error ms |
|---|---:|---:|---:|---:|---:|
| VIT | 6.96316 | 0.823210 | 0.00454612 | 35.4284 | 11.81350 |
| 1DCNN | 7.10661 | 0.805955 | 0.00689362 | 32.3662 | 18.65375 |

## Paired independent-test differences:CNN minus ViTESI

| Metric | Mean delta | Descriptive95%interval |
|---|---:|---|
| LE_mm | +0.143444 | [-0.0804089, +0.365346] |
| AUC | -0.0172542 | [-0.0193469, -0.0152433] |
| nMSE | +0.0023475 | [+0.00215803, +0.00254995] |
| PSNR | -3.06215 | [-3.17608, -2.9418] |
| time_err_ms | +6.84025 | [+4.50695, +9.26428] |

Intervals use5000paired evaluation-sample bootstrap draws,analysis RNGseed0. They condition on the trained checkpoints and do not represent training-seed uncertainty; no multiplicity adjustment. Held-out data selected checkpoints; test2k_s1 was independent of model selection.

## Preliminary interpretation

- CNNtestLE7.10661mm versus6.96316mm: delta+0.14344mm,interval[-0.08041,+0.36535]. This small observed localization difference does not establish a reliable gap,superiority or equivalence.
- CNNtemporal error18.65375ms versus11.81350ms: delta+6.84025ms,[4.50695,9.26428]. The temporal difference is more pronounced in this conditional sample comparison.
- CNNtestAUC is lower,nMSE higher,and PSNR lower by3.06215dB ([-3.17608,-2.94180]). Close localization error therefore does not imply unchanged temporal/source-signal reconstruction quality.
- These observations favour the ViTESI checkpoint on several metrics beyond localization error at nearly matched parameter counts,in this setup. They do not isolate self-attention as the cause or establish general architecture superiority. LSTMcomparison,different training sets/seeds and real-data verification remain necessary.

## Compute and measured training duration

| Model | Parameters | Matrix MACs/sample | Matrix FLOP estimate/sample | Training seconds |
|---|---:|---:|---:|---:|
| ViTESI original6-layer | 3,565,538 | 2,477,696,000 | 4,955,392,000 | 15,957.178 |
| MatchedCNN | 3,564,876 | 1,782,438,000 | 3,564,876,000 | 15,960.486 |
| MatchedLSTM (not trained) | 3,557,700 | 1,774,130,000 | 3,548,260,000 | pending |

Operation estimates use the explicit dense matrix/convolution convention in count_flops.py:alltimepoints,both LSTMdirections/layers,andattentionQK/AV products included; bias,norm,activation,softmax,recurrentpointwise gates and data movement excluded. FLOP estimate=2*MAC,not full measured hardware operations. Torchinfo1.8.0 shape/parameter checks supplement the count; its incomplete raw totals are only diagnostic. Lower estimated matrix operations did not produce an observed training-duration benefit here. No inference-speed benchmark is claimed.

Host Linux kernel differs(control6.18.48,CNN6.18.55); architecture/glibc and all other recorded runtime fields/packages match. Host environments are not identical,which especially limits timing comparisons. CNNvalidation-selected epoch99(1-based); control epoch100. Both nevertheless trained all100epochs.

## Audits,storage and pending work

Seven real-output comparison tests passed,including rejection of source/parameter/width/kernel/noise/sample/hash changes,smoke-as-full rejection,and direct verification of all-five-metric paired deltas. Both archived smokes revalidated. Full CNN audit checks mandatoryLinuxCUDAgate,all scientific stages,fingerprints,clean pinned source,locked profile/settings,split,all100epochs,exported validation/training checkpoint and weight hashes,selected-weight provenance,both2000sample IDs/finite metric means and cross-source sample/runtime compatibility. SourceAST review confirms only reviewed option plumbing changes to train/eval; loaders/data/metrics/anatomy/profile/requirements and baseline CNN/LSTM model source are unchanged.

Raw local outputs: ignored `model_training/results/ablation/a5/cnn2604-workers4-s0/download/out/`. Comparison CSV/JSON/Markdown plus localization/temporal PNG/PDF figures: `model_training/results/ablation/a5/comparison-cnn-c671696/`.

Mounted Drive root `H:/My Drive/SynapseAI-FYP/`: checkpoints under `checkpoints/ablation-A5/cnn2604-workers4-s0/`,results under `results/ablation/a5/cnn2604-workers4-s0/`,analysis under `results/ablation/a5/comparison-cnn-c671696/`. CNNarchive31files158,081,865bytes; generated comparison7files. Copies passed read-back SHA verification; local originals retained. Last.ckpt retained with archive hash; no originating last.ckpt digest was provided by the run manifest. Remote cloud sync not independently checked.

Fresh quota after CNNcompletion:1.72GPUh,nextrefresh2026-10-31UTC. LSTM295smoke/checks passed, but conservative full budget8.03h does not fit. Do not shorten its100epoch protocol or submit an underfunded job. The recurrent A3follow-up remains waiting on the full LSTMcheckpoint. No additional training job or automation was created.

Reproduce current comparison:

```powershell
python model_training/ablation/a5_compare.py --control model_training/results/ablation/a4/depth6-workers4-s0/download/out --cnn model_training/results/ablation/a5/cnn2604-workers4-s0/download/out --out model_training/results/ablation/a5/comparison-cnn-c671696
python model_training/ablation/test_a5_compare.py --control model_training/results/ablation/a4/depth6-workers4-s0/download/out --cnn model_training/results/ablation/a5/cnn2604-workers4-s0/download/out
```

Add `--lstm <full-LSTM-out>` only after that complete job passes audit. This interim record is not completion of A5 or of the overall ablation study. Different training sets/seeds and realEEGverification remain future work,after validation stabilizes.
