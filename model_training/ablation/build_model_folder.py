"""
Build the head-model folder that eval.py expects:
    <root>/simulation/<subject>/constrained/<montage>/fsav_994/model/

The original pipeline produced it with stESI's `create_head_model.py`, which BrainForge
dropped. For NN-only evaluation on the 994-region source space, only these are read:

- fs_cortex_20k_region_mapping.mat : copied byte-for-byte from anatomy/. Its `nbs` holds the
  region neighbours, which the localisation error needs.
- fs_cortex_neighbors_994.mat : `nbs` reshaped to a padded (994, max_nbs) matrix, using the
  same code as eval.py (which otherwise builds it lazily, see below).
- ch_fsav_994.mat : electrode descriptor used only to build HeadModel.ElectrodeSpace; no
  metric reads it. Built from anatomy/electrode_75.mat, the electrode file matching the
  75-channel leadfield (positions converted from mm to metres).

Usage (from repo root):
    python model_training/ablation/build_model_folder.py -root .
"""

import argparse
import os
import shutil

import numpy as np
from scipy.io import loadmat, savemat


def build(root, subject="fsaverage", montage="standard_1020", src="fsav_994"):
    anatomy = os.path.join(root, "anatomy")
    model_dir = os.path.join(root, "simulation", subject, "constrained", montage, src, "model")
    os.makedirs(model_dir, exist_ok=True)

    rm_src = os.path.join(anatomy, "fs_cortex_20k_region_mapping.mat")
    rm = loadmat(rm_src)
    if "nbs" not in rm:
        raise KeyError(f"`nbs` missing from {rm_src}")
    shutil.copyfile(rm_src, os.path.join(model_dir, "fs_cortex_20k_region_mapping.mat"))

    # Pre-build fs_cortex_neighbors_994.mat with eval.py's own reshape code. eval.py only
    # builds it lazily, and on that first run it keeps the raw object array, which then
    # crashes `get_patch` (2-D indexing); every later run loads this file instead.
    neighbors = rm["nbs"][0]
    m = -1
    for n in neighbors:
        if n.shape[1] > m:
            m = n.shape[1]
    neighbors_ref = np.ones((neighbors.shape[0], m), dtype=int) * (-1)
    for r in range(neighbors_ref.shape[0]):
        nbs = neighbors[r][0]
        neighbors_ref[r, : len(nbs)] = nbs
    savemat(os.path.join(model_dir, "fs_cortex_neighbors_994.mat"), {"nbs": neighbors_ref})

    eloc = loadmat(os.path.join(anatomy, "electrode_75.mat"))["eloc75"][0]
    names = [str(e["labels"][0]) for e in eloc]
    positions = np.array(
        [[float(e["X"][0, 0]), float(e["Y"][0, 0]), float(e["Z"][0, 0])] for e in eloc]
    ) / 1000.0
    savemat(
        os.path.join(model_dir, f"ch_{src}.mat"),
        {"nb_channels": len(names), "positions": positions, "names": np.array(names, dtype=object)},
    )
    print(f"model folder ready: {model_dir} ({len(names)} electrodes)")
    return model_dir


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-root", default=".", help="repo root (contains anatomy/ and simulation/)")
    p.add_argument("-subject", default="fsaverage")
    p.add_argument("-montage", default="standard_1020")
    a = p.parse_args()
    build(os.path.abspath(a.root), a.subject, a.montage)
