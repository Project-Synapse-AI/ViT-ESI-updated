# Frozen datasets (Objective 2 ablations, issue #3)

Every ablation trains on `train6k_s0` and reports on `test2k_s1`. A3 also uses `widepeak2k_s2`. Don't regenerate them with other settings. Any new set needs its own name and an entry here.

- Generated 2026-10-06 with generator commit `e9edffb`.
- Local path: `simulation/fsaverage/constrained/standard_1020/fsav_994/simu/<name>/`.
- Shared copy: team Drive (see the bottom of this file).

| Name | Purpose | Samples | Seed | Base ERP centre | Patch peak centres | Size |
|---|---|---|---|---|---|---|
| `train6k_s0` | training (thesis setup: 6,000 samples) | 6000 | 0 | 125–375 ms | 89–484 ms | 6.0 GB |
| `test2k_s1` | main test set (thesis size: 2,000) | 2000 | 1 | 125–375 ms | 90–479 ms | 2.0 GB |
| `widepeak2k_s2` | A3: peaks spread across the window | 2000 | 2 | 100–749 ms | 74–963 ms | 2.0 GB |

Checked: 0 samples of either test set match a training sample, by metadata or by exact EEG.

## Commands (run from the repo root, `$env:PROJECT_ROOT = (Get-Location).Path`)
Common arguments for all three:
```
-mk standard_1020 -ss fsav_994 -o constrained -sn fsaverage -rf "$env:PROJECT_ROOT" --leadfield_mat "$env:PROJECT_ROOT/anatomy/leadfield_75_20k.mat" -fs 500 -d 1000 -af "$env:PROJECT_ROOT/anatomy"
```
```
python data_generation/sereega/simu_extended_source.py -sin train6k_s0    -ne 6000 <common> --seed 0
python data_generation/sereega/simu_extended_source.py -sin test2k_s1     -ne 2000 <common> --seed 1
python data_generation/sereega/simu_extended_source.py -sin widepeak2k_s2 -ne 2000 <common> -c 425 -interdev 0.5 0.765 0.02 --seed 2
```
Everything else uses the generator defaults:
- 1 s window at 500 Hz, ERP width 50 ms, base centre 250 ms ± 50%.
- Patches within a sample vary their centre by ±30% of the base.
- Amplitude: base 1 ± 50%, and each patch ±70% of that.
- Patches per sample: `-np_min 1 -np_max 5`. Patch order: `-o_min 1 -o_max 5`.

## Things to know
- **Seed 0 = the old hard-coded seed.** `train6k_s0` is exactly what the unmodified generator produces for 6,000 samples, so it is most likely BrainForge's training data, if they used the defaults.
- **Test sets need a different seed.** With the old hard-coded seed, any "test" set generated separately is a copy of the first training samples.
- **Patches per sample are 1–4, and patch order is 1–4, not 1–5.** `np.random.randint` excludes its upper bound. This is kept to match the thesis distribution.
- **Wide-peak changes only the sample-level centre distribution** (`-c 425`, centre deviation 0.765, giving base 100–750 ms). The within-sample spread (±30%) is the same as in training, so peaks stay inside the 1 s window.
- **Peak timing is in each sample's metadata.** The md JSON stores `base_center`, `base_width` and `base_amplitude`, plus `erp` per patch (`center`, `width`, `ampl`, with patch k ↔ `seeds[k]`). A patch peaks at its `center`, in ms from the start of the window.

## Shared copy
**None, by decision (2026-10-06).** The commands above rebuild every set byte-identically in about 5 minutes, so there's no need to upload 10 GB to Drive. To check a rebuild, compare against the per-set numbers in the table: sample count, centre ranges and size.
