# Encoder-depth ablation: completed initial seed-0 sweep

All planned depths (1,2,3,4 and matched6-layer control) completed100epochs and passed full-output audits. These are **preliminary single-training-seed findings, not final generalization claims**. Verification on different training sets/seeds and real EEG remains future work. No exact thesis-reproduction claim is made.

## Protocol and reference

Scientific training/evaluation source: `2f4068eefb2e90345d3332bf856db130bfc402e1`. Analysis: `64dc0b8`. Frozen profile SHA-256: `793187c9c72b312bf7a324b50a3a2a17a1e45fe7cbc51fa910f24ba1fd3259eb`.

Frozen train8k_s0,6000training/2000validation split,seed0,100epochs,batch8,Adam0.001,cosine loss,linear scaling,5dB,validation-loss checkpoint selection,noearlystop. Workers4,pinned and persistent,prefetch1,one visibleT4,FP32. Width256,heads8,MLP512,dropout0.1; only encoderdepth changes. See A4_METHODS.md for complete methods.

The six-layer control is a separately retrained instance of the original architecture. Its heldout/test LE7.03977/6.96316mm is distinct from the earlier local validation reference7.14785/7.10056mm used in attention-removal/timing analyses. Do not mix these checkpoints across comparisons.

## All five evaluation metrics

Lower LE,nMSE and temporal error are preferable; higher AUC and PSNR are preferable. Temporal error measures activity timing accuracy in milliseconds; training duration below is separate runtime information.

### train8k_s0_heldout (2000 samples)

| Depth | LE mm | AUC | nMSE | PSNR dB | Temporal error ms |
|---|---:|---:|---:|---:|---:|
| 6 | 7.03977 | 0.820640 | 0.00471633 | 35.4371 | 11.40958 |
| 1 | 7.56880 | 0.825668 | 0.00514016 | 34.6889 | 12.20150 |
| 2 | 7.09753 | 0.826802 | 0.00463640 | 35.4436 | 11.70008 |
| 3 | 6.94706 | 0.819604 | 0.00449323 | 35.4165 | 12.33558 |
| 4 | 7.01463 | 0.824721 | 0.00462859 | 34.9344 | 13.14250 |

### test2k_s1 (2000 samples)

| Depth | LE mm | AUC | nMSE | PSNR dB | Temporal error ms |
|---|---:|---:|---:|---:|---:|
| 6 | 6.96316 | 0.823210 | 0.00454612 | 35.4284 | 11.81350 |
| 1 | 7.54314 | 0.831391 | 0.00501716 | 34.6677 | 11.24817 |
| 2 | 7.11797 | 0.831811 | 0.00454678 | 35.4880 | 10.83458 |
| 3 | 6.90158 | 0.823793 | 0.00438414 | 35.4212 | 12.45675 |
| 4 | 6.93942 | 0.827877 | 0.00451614 | 34.8970 | 15.51858 |

## Paired independent-test differences versus six layers

| Depth | LE delta mm | 95% interval | Temporal-error delta ms | 95% interval |
|---|---:|---|---:|---|
| 1 | +0.57997 | [+0.38488, +0.78853] | -0.56533 | [-2.24836, +1.02342] |
| 2 | +0.15481 | [-0.02670, +0.33829] | -0.97892 | [-2.68752, +0.65542] |
| 3 | -0.06158 | [-0.23958, +0.12639] | +0.64325 | [-1.36463, +2.56912] |
| 4 | -0.02374 | [-0.20665, +0.15319] | +3.70508 | [+1.76773, +5.64375] |

Intervals are descriptive95% percentile sample-bootstrap intervals (5000draws,RNGseed0), paired by sample/noise. They condition on the trained checkpoints, do not estimate training-seed uncertainty, and are not multiplicity adjusted. Held-out data selected checkpoints; only test2k_s1 is independent of selection.

## Capacity and measured training duration

| Depth | Parameters | Reduction vs6 | Training seconds | Selected epoch (1-based) |
|---|---:|---:|---:|---:|
| 6 | 3565538 | 0.00% | 15957.178 | 100 |
| 1 | 930018 | 73.92% | 15466.985 | 95 |
| 2 | 1457122 | 59.13% | 16195.248 | 99 |
| 3 | 1984226 | 44.35% | 15654.982 | 94 |
| 4 | 2511330 | 29.57% | 15151.199 | 94 |

## Interpretation and limits

- One layer substantially reduces parameter count but increases test LE by0.57997mm (interval[0.38488,0.78853]).
- Depths2,3 and4 have LE-difference intervals spanning zero. This does not establish equivalence or superiority. Depth3 has44.35%fewer parameters and a small observed testLE difference of-0.06158mm; it is a candidate for further verification, not a selected replacement based on this test.
- Depth4 testLE is6.93942mm versus6.96316mm control, but temporal error rises15.51858 versus11.81350ms: delta+3.70508ms (interval[1.76773,5.64375]). PSNR also falls by0.53137dB (interval[-0.61675,-0.44460]). Close localization error therefore does not imply unchanged temporal or reconstruction quality.
- Recorded training durations are similar across depths. There is no demonstrated substantial training-speed gain. No inference-speed or FLOPs benchmark was performed in this sweep.
- Depth changes parameter count as well as layer count; this cannot by itself distinguish attention from capacity effects. Interpret alongside averaging and parameter-matched baselines.
- Kaggle changed Linux kernel6.18.48 to6.18.55 between depth6/1 and depth2/3/4. Recorded package inventories and other runtime fields match; architecture/glibc match. This host difference is disclosed in the analysis and especially limits timing comparisons. No runtime package or scientific setting was changed to accommodate it.
- Different training datasets/seeds and real-data evaluation remain necessary. Real-data ablations follow stabilization of the validation procedure and compatible preprocessing/montages/ground-truth assessment.

## Provenance and storage

| Depth | Validation-selected weights SHA-256 |
|---|---|
| 6 | `532cbeee32130b0c6145c42f676177f06aca509e4ecf5eef3ae09ac764997ca6` |
| 1 | `eb1a3024dd81f307c05bfa819c98d55366aa5334a757a0b4b75647e2fef32b84` |
| 2 | `f3e23c2fb0c12fb035165a6615656d5faaa1c4fd8b48088954f9b48e819ac0ae` |
| 3 | `aa050e470b7acf98dbcd4ed0ff77ce1c134408c29f2190f26927a4a638048ede` |
| 4 | `df937de714755d7bb4ce364a53ee165c2b35e4bbbb505021467fe261e66ad0eb` |

Each notebook is `dasunwijethunga/a4-depthD-workers4-s0`,version1. Raw local outputs are ignored under `model_training/results/ablation/a4/depthD-workers4-s0/download/out/`. Final tables,paired intervals,JSON/CSV provenance,and LE-versus-depth/parameter PNG/PDF figures are in `model_training/results/ablation/a4/comparison-final-64dc0b8/`.

Mounted Drive root: `H:/My Drive/SynapseAI-FYP/`. Checkpoints: `checkpoints/ablation-A4/depthD-workers4-s0/`; results: `results/ablation/a4/depthD-workers4-s0/`; final analysis: `results/ablation/a4/comparison-final-64dc0b8/`. Each archive has README/checksums; copies passed read-back SHA256 verification. Local originals remain. Remote cloud sync was not independently confirmed.

Audit: all planned100epochs,source/profile,successful generation/fingerprint/training/evaluation stages,matching splits/settings,selected/exported checkpoint hashes,paired2000sample IDs,noiseseeds,finite five-metric rows and summary means. Comparison regression suite:13checks passed, including kernel-only drift acceptance and runtime/package/glibc drift rejection. Metric formulas were unchanged.

Reproduce: use A4_DEPTH.md's full-job commands with frozen source and depthDflags. Analyse by passing downloaded depth6control and repeated `--variant` paths for1,2,3,4 to `a4_compare.py` (5000draws,seed0). No further training is required to reproduce these tables from the archived outputs.
