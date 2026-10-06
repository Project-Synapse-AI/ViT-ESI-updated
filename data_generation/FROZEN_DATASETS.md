# Frozen datasets (Objective 2 ablations, issue #3)

Every ablation trains on `train8k_s0` using the **thesis split** and is scored on `test2k_s1`. A3 also uses `widepeak2k_s2`. Don't regenerate them with other settings. Any new set needs its own name and an entry here.

- Generated 2026-10-06 with generator commit `e9edffb`.
- Local path: `simulation/fsaverage/constrained/standard_1020/fsav_994/simu/<name>/`.

| Name | Purpose | Samples | Seed | Base ERP centre | Patch peak centres | Size |
|---|---|---|---|---|---|---|
| `train8k_s0` | training, thesis setup (6,000 train + 2,000 held out) | 8000 | 0 | 125–375 ms | 89–487 ms | 8.0 GB |
| `test2k_s1` | independent test set, never seen in training | 2000 | 1 | 125–375 ms | 90–479 ms | 2.0 GB |
| `widepeak2k_s2` | A3: peaks spread across the window | 2000 | 2 | 100–749 ms | 74–963 ms | 2.0 GB |

Checked: 0 samples of either test set match a `train8k_s0` sample, by metadata or by exact EEG.

## How the thesis used its data (and how we do)
BrainForge's 5.07 mm (thesis Table 5.1) comes from **8,000 generated samples: 6,000 trained on and 2,000 held out** (thesis p46 and p48).
- **No separate test set.** `main_train.py` loads `-to_load` samples and holds back `-per_valid` of them.
- **eval.py scores that same held-out part.** Both scripts call `seed_everything(0)` (hard-coded) before `random_split`, so the held-out set depends only on `-to_load` and `-per_valid`.
- **Those held-out 2,000 also drive early stopping** (validation loss, patience 20). The saved model is picked by *training* loss.

So, for every training run:
```
main_train.py train8k_s0 ... -to_load 8000 -per_valid 0.25     # 6,000 train / 2,000 held out (split seed 0)
```
Report two numbers per model:
1. **Thesis-style:** the held-out 2,000 of `train8k_s0`:
   `evaluate.py train8k_s0 -to_load 8000 -heldout -per_valid 0.25 ...`
   This is the number to compare with 5.07 mm.
2. **Clean:** all of `test2k_s1`. No model ever sees it, not even for early stopping:
   `evaluate.py test2k_s1 -to_load 2000 ...`

Two seeds are in play, and both are 0. The **generator seed** (`--seed`) decides what the samples look like. The **split seed** (`seed_everything(0)` in main_train/eval) decides which 2,000 are held out.

## Commands (run from the repo root, `$env:PROJECT_ROOT = (Get-Location).Path`)
Common arguments for all three:
```
-mk standard_1020 -ss fsav_994 -o constrained -sn fsaverage -rf "$env:PROJECT_ROOT" --leadfield_mat "$env:PROJECT_ROOT/anatomy/leadfield_75_20k.mat" -fs 500 -d 1000 -af "$env:PROJECT_ROOT/anatomy"
```
```
python data_generation/sereega/simu_extended_source.py -sin train8k_s0    -ne 8000 <common> --seed 0
python data_generation/sereega/simu_extended_source.py -sin test2k_s1     -ne 2000 <common> --seed 1
python data_generation/sereega/simu_extended_source.py -sin widepeak2k_s2 -ne 2000 <common> -c 425 -interdev 0.5 0.765 0.02 --seed 2
```
Everything else uses the generator defaults:
- 1 s window at 500 Hz, ERP width 50 ms, base centre 250 ms ± 50%.
- Patches within a sample vary their centre by ±30% of the base.
- Amplitude: base 1 ± 50%, and each patch ±70% of that.
- Patches per sample: `-np_min 1 -np_max 5`. Patch order: `-o_min 1 -o_max 5`.

## Things to know
- **Seed 0 = the old hard-coded seed.** `train8k_s0` is what the unmodified generator produces for 8,000 samples, which is very likely BrainForge's thesis data. Samples are generated in order, so a smaller seed-0 run gives exactly the first N samples of this set.
- **Test sets need a different seed.** With the old hard-coded seed, any "test" set generated separately is a copy of the first training samples.
- **Patches per sample are 1–4, and patch order is 1–4, not 1–5.** `np.random.randint` excludes its upper bound. This is kept to match the thesis distribution.
- **Wide-peak changes only the sample-level centre distribution** (`-c 425`, centre deviation 0.765, giving base 100–750 ms). The within-sample spread (±30%) is the same as in training, so peaks stay inside the 1 s window.
- **Peak timing is in each sample's metadata.** The md JSON stores `base_center`, `base_width` and `base_amplitude`, plus `erp` per patch (`center`, `width`, `ampl`, with patch k ↔ `seeds[k]`). A patch peaks at its `center`, in ms from the start of the window.

## Shared copy
**None, by decision (2026-10-06).** The commands above rebuild every set byte-identically in a few minutes, so there's no need to upload 12 GB to Drive. To check a rebuild, compare against the per-set numbers in the table: sample count, centre ranges and size.
