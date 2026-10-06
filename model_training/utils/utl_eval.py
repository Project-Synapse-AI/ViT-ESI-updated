"""
Per-sample evaluation helpers shared by eval.py and the ablation harness.

The math in `sample_metrics` and `rescale_prediction` is moved verbatim from eval.py
(BrainForge's thesis evaluation) so that every number stays comparable with the thesis.
Known quirks are kept on purpose (see comments) - do not "fix" them here.
"""

import numpy as np
import torch
from scipy.io import loadmat
from skimage.metrics import peak_signal_noise_ratio as psnr

from utils import utl
from utils import utl_metrics as met


def load_leadfield_mat(mat_path: str):
    m = loadmat(mat_path)
    if "G" in m:
        return m["G"]
    if "fwd" in m:
        return m["fwd"]
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and getattr(v, "ndim", 0) == 2:
            return v
    raise KeyError(f"No leadfield matrix found in {mat_path}. Keys={list(m.keys())}")


def _strip_prefix(state_dict, prefix):
    changed = False
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state[k[len(prefix) :]] = v
            changed = True
        else:
            new_state[k] = v
    return new_state if changed else None


def _add_prefix(state_dict, prefix):
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


def load_module_weights(module, weights_path):
    checkpoint = torch.load(weights_path, map_location=torch.device("cpu"))
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        base_state = checkpoint["state_dict"]
    else:
        base_state = checkpoint

    candidates = [base_state]
    for prefix in ("model.", "model.model."):
        stripped = _strip_prefix(base_state, prefix)
        if stripped is not None:
            candidates.append(stripped)
    for prefix in ("model.",):
        candidates.append(_add_prefix(base_state, prefix))

    last_error = None
    for cand in candidates:
        try:
            module.load_state_dict(cand)
            return
        except RuntimeError as err:
            last_error = err
    raise RuntimeError(f"Failed to load weights from {weights_path}: {last_error}")


def rescale_prediction(j_hat, M_unscaled, fwd, loss, max_src):
    """Bring a network output back to source amplitude units (eval.py, per-method branch)."""
    if loss == "cosine":
        return utl.gfp_scaling(M_unscaled, j_hat, torch.from_numpy(fwd))
    return j_hat * max_src


def sample_metrics(j, j_unscaled, j_hat, seeds, patches, spos, neighbors, t_vec):
    """
    LE, time error, nMSE, AUC and PSNR for one sample, exactly as eval.py computes them.

    j          : scaled ground-truth sources (994, T) - used to break patch overlaps
    j_unscaled : ground-truth sources in amplitude units (994, T)
    j_hat      : rescaled estimate (994, T)
    seeds      : list of seed region ids, patches: list of region-id lists (one per seed)
    Returns a dict with le, te, nmse, auc, psnr, seeds_hat and overlap (bool).
    """
    le = 0
    te = 0
    nmse = 0
    auc_val = 0
    seeds_hat = []
    overlap = False

    # Overlap handling (only meaningful if there are >= 2 sources)
    if len(patches) >= 2:
        inter = list(set(patches[0]).intersection(patches[1]))
        if len(inter) > 0:
            overlap = True
            to_keep = torch.argmax(
                torch.Tensor(
                    [j[seeds[0], :].abs().max(), j[seeds[1], :].abs().max()]
                )
            )
            seeds = [seeds[to_keep]]
            patches = [patches[to_keep]]

    act_src = [s for l in patches for s in l]

    for kk in range(len(seeds)):
        s = seeds[kk]
        other_sources = np.setdiff1d(act_src, patches[kk])
        t_eval_gt = torch.argmax(j[s, :].abs())

        # find estimated seed in a neighboring area
        # (quirk kept from eval.py: the order-5 zone is computed then overwritten by order 2)
        eval_zone = utl.get_patch(order=5, idx=s, neighbors=neighbors)
        eval_zone = np.setdiff1d(eval_zone, other_sources)
        eval_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)

        s_hat = eval_zone[torch.argmax(j_hat[eval_zone, t_eval_gt].abs())]
        t_eval_pred = torch.argmax(j_hat[s_hat, :].abs())

        le += torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
        te += np.abs(t_vec[t_eval_gt] - t_vec[t_eval_pred])
        auc_val += met.auc_t(j_unscaled, j_hat, t_eval_gt, thresh=True, act_thresh=0.0)

        nmse_tmp = (
            (
                j_unscaled[:, t_eval_gt] / j_unscaled[:, t_eval_gt].abs().max()
                - j_hat[:, t_eval_gt] / j_hat[:, t_eval_gt].abs().max()
            )
            ** 2
        ).mean()
        nmse += nmse_tmp

        seeds_hat.append(s_hat)

    le = le / len(seeds)
    te = te / len(seeds)
    nmse = nmse / len(seeds)
    auc_val = auc_val / len(seeds)

    # quirk kept from eval.py: data_range = min(gt_norm) - max(hat_norm)
    psnr_val = psnr(
        (j_unscaled / j_unscaled.abs().max()).numpy(),
        (j_hat / j_hat.abs().max()).numpy(),
        data_range=(
            (j_unscaled / j_unscaled.abs().max()).min()
            - (j_hat / j_hat.abs().max()).max()
        ),
    )

    return {
        "le": le,
        "te": te,
        "nmse": nmse,
        "auc": auc_val,
        "psnr": psnr_val,
        "seeds_hat": seeds_hat,
        "overlap": overlap,
    }
