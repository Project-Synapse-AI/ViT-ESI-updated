# CLAUDE.md

Team SynapseAI's fork of BrainForge's ViTESI: EEG source imaging that maps 75 scalp electrodes to 994 cortical regions, 500 timepoints @ 500 Hz. `upstream` = Project-BrainForge (read-only). See `CONTRIBUTING.md` for branches and remotes.

## How we work (follow exactly)
1. **Pick an issue.** Current plan: milestone *Obj 2 · Ablation study*, roadmap in #1, work issues in the order its flow shows.
2. **Branch off `dev`** (`ablation/<test>`, `real-data/<dataset>`, `freq/<topic>`, `chore/<thing>`) and open a **draft PR right away**, titled `WIP: <issue title>`, body `Closes #N`, using the PR template.
3. **Post every decision as a PR comment**, in plain simple English (no jargon without a one-line explanation), with these four parts:
   - **What** we decided.
   - **Why** this option.
   - **Why not** the alternatives we considered.
   - **How** it's done (files, commands, key numbers).
4. Push small commits as you go. When done: fill in the PR template's results, update the results table in #1, mark the PR ready, merge into `dev`.

## ⛔ Hard stop rule
If any dependency or piece of infra is missing or not visible, **stop immediately** and report it. That means a package, CUDA/GPU, dataset, checkpoint, anatomy file, thesis setting, credential or permission. **Don't hack around it**: workarounds silently degrade results.

The report should say:
- what is missing,
- where you looked,
- what it blocks,
- what the user needs to provide.

Add the `blocked` label to the issue.

Never, without asking first:
- fall back to CPU,
- shrink the dataset, epochs or batch "just to make it run",
- use mock or random data,
- swap package versions,
- edit `anatomy/`,
- skip a metric,
- guess a thesis hyperparameter.

## Setup & commands
Windows + PowerShell, RTX 3060 Laptop 6 GB. From the repo root: `python -m venv venv; venv\Scripts\activate; pip install -r requirements.txt` (CUDA build of torch), then `$env:PROJECT_ROOT = (Get-Location).Path`.
- Generate data: `python data_generation/sereega/simu_extended_source.py -sin <name> -ne <N> -mk standard_1020 -ss fsav_994 -o constrained -sn fsaverage -rf "$PROJECT_ROOT" --leadfield_mat "$PROJECT_ROOT/anatomy/leadfield_75_20k.mat" -fs 500 -d 1000 -af "$PROJECT_ROOT/anatomy"`
- Train (from `model_training/`): `python main_train.py <simu_name> -simu_type sereega -source_space fsav_994 -electrode_montage standard_1020 -orientation constrained -model {VIT,1DCNN,LSTM,DEEPSIF} -loss cosine -scaler linear -eeg_snr 5 -n_times 500 -per_valid 0.2 -to_load <N> -n_epochs <E> -leadfield_mat ... -simu_folder ... -results_path "$PROJECT_ROOT/model_training/results"`. ViT size flags: `-vit_depth -vit_heads -vit_embed_dim -vit_mlp_dim`.
- Evaluate: `model_training/eval.py` (all 5 metrics, per sample). Params/FLOPs: `python count_flops.py --model VIT --input_shape 500 75`.
- No test suite. "Testing" means a smoke run with small `-ne`, `-to_load` and `-n_epochs`, and it must pass before any full run.

## Code facts that matter
- `model_training/models/vit.py` `EEGViT`: each **timepoint is a token** (75 → 256 embed, depth 6, 8 heads), built on `nn.TransformerEncoderLayer`. In `eval()` it uses PyTorch's fused fast path, which skips Python hooks. Disable it (`torch.backends.mha.set_fastpath_enabled(False)`) before masking heads.
- SEREEGA data is EEG `(75, 500)` sensors-first. `Jact` holds only active regions, scattered back to 994 via `md/<id>_md_json_flie.json`.
  - ERP peaks default to 125–375 ms of the 1000 ms window.
  - `-itradev`/`-interdev` are broken on the CLI (`type=list`), see #3.
  - The md JSON doesn't store the peak centre until #3 lands.
- The 5 metrics: LE (mm), AUC, nMSE, PSNR (dB), time error (ms). Thesis baselines (params M / LE mm): 1D-CNN 5.61/6.52 · LSTM 0.45/7.17 · DeepSIF 22.36/5.19 · **ViTESI 3.57/5.07**.

## Data & results
- Never commit `simulation/`, `model_training/results/`, `.pt`/`.ckpt` or `.mat` outputs.
- Datasets and checkpoints live on the team Drive (`SynapseAI-FYP/datasets/`, `SynapseAI-FYP/checkpoints/`), each with a README giving the exact command, seed and commit hash.
- Every reported number must say which dataset, checkpoint and commit it came from.
- Compare against the #4 baseline on the same frozen test set.
