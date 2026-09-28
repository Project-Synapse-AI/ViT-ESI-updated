# Contributing (Team SynapseAI)

This repo is Team SynapseAI's fork of [Project-BrainForge/ViT-ESI-updated](https://github.com/Project-BrainForge/ViT-ESI-updated), the ViTESI code from Team BrainForge's thesis. We extend it for our FYP objectives.

## Branches
| Branch | Purpose |
|---|---|
| `main` | BrainForge's code as inherited. The reproduction baseline (ViTESI ≈ 5.07 mm LE). Only updated by syncing from `upstream`. |
| `dev` | Integration branch and the default. All work merges here through PRs. |
| `ablation/<test>` | Objective 2, e.g. `ablation/a3-peak-position` |
| `real-data/<dataset>` | Objective 1 |
| `channel-token` | Objective 3 (montage-agnostic, electrode-as-token) |
| `freq/<topic>` | Objective 4 (STFT / wavelet input) |

Branch off `dev`, open a PR back into `dev`, and fill in the PR template.

## Remotes
```bash
git remote add upstream https://github.com/Project-BrainForge/ViT-ESI-updated.git
git fetch upstream
```
To pull in BrainForge fixes, merge `upstream/main` into `main`, then `main` into `dev`.

## Large files
Simulated data (`simulation/`) and training output (`model_training/results/`) are gitignored. Share datasets and checkpoints through the team Drive, and record the path plus the commit hash in the PR and in the results log.
