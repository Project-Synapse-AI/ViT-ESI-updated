"""
2023-08-25 script to evaluate results
modification august 2023 - to use with deepsif datasets (neural mass model based and sereega based)
"""

import argparse
import datetime
import os
import sys
from os.path import expanduser
from pathlib import Path

# Import librairies
try:
    import mne  # optional dependency
    HAS_MNE = True
except ModuleNotFoundError:
    mne = None
    HAS_MNE = False
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import yaml
import json
import pandas as pd
from pytorch_lightning import seed_everything
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split

from loaders import ModSpikeEEGBuild, EsiDatasetds_new
from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from utils import utl
from utils import utl_metrics as met
from utils import utl_inv as inv
from utils.utl_eval import (
    load_leadfield_mat as _load_leadfield_mat,
    load_module_weights as _load_module_weights,
    rescale_prediction,
    sample_metrics,
)

############# METHODS ############################
linear_methods = ["MNE", "sLORETA"]  # , "eLORETA"]
nn_methods = ["cnn_1d", "lstm", "deep_sif", "eeg_vit"]
methods = linear_methods + nn_methods

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
seed_everything(0)
device = torch.device("cpu")
print(f"Device: {device}")

home = expanduser("~")

save_suffix = "test"

################################################################################################################
parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
#argument to load the data
parser.add_argument("simu_name", type=str, help="name of the simulation")
parser.add_argument("-root_simu", type=str, required=True, help="Simulation folder (parent folder in the folder tree containing the simulations)")
parser.add_argument("-results_path", type=str, required=True, help="Path to where to save results")
parser.add_argument(
    "-eval_simu_type", type=str, help="type of simulation used (NMM or SEREEGA)"
)
parser.add_argument("-orientation", type=str, default="constrained", help="constrained or unconstrained, orientation of the sources")
parser.add_argument("-electrode_montage", type=str, default="standard_1020", help="name of the electrode montage to use")
parser.add_argument("-source_space", type=str, default="ico3", help="name of the source space")
parser.add_argument(
    "-subject_name",
    type=str,
    default="fsaverage",
    help="Subject name when using `simulation/<subject>/...` folder layout",
)
parser.add_argument(
    "-leadfield_mat",
    type=str,
    default=None,
    help="Optional path to a .mat leadfield to use for evaluation (e.g. anatomy/leadfield_75_20k.mat).",
)

parser.add_argument("-spikes_folder", type=str, default="nmm_spikes_nov23", help="folder with spikes for NMM based simulations")
parser.add_argument(
    "-n_times", type=int, default=500, help="number of time samples in the signal"
)
parser.add_argument(
    "-to_load",
    default=100,
    type=int,
    help="number of samples to load in the train+val dataset",
)
parser.add_argument(
    "-per_valid",
    default=0.2,
    type=float,
    help="fraction of the dataset to use for validation",
)
parser.add_argument(
    "-eeg_snr",
    default=5,
    type=int,
    help="SNR of the EEG data (additive white gaussian noise)",
)


parser.add_argument(
    "-net_from_file", action="store_true", help="load network parameters from yaml file"
)
parser.add_argument(
    "-params_nn",
    type=str,
    default="./params_nns.yaml",
    help="yaml file with the parameters of the networks to load",
)

# parameters to load the trained model
parser.add_argument(
    "-train_bs", type=int, default=8, help="batch size used for training"
)
parser.add_argument(
    "-n_epochs", type=int, default=100, help="number of epochs used for training11"
)
parser.add_argument(
    "-train_loss", type=str, default="cosine", help="loss used to train the networks"
)
parser.add_argument(
    "-scaler", type=str, default="linear", help="type of scaling to use"
)
parser.add_argument(
    "-train_simu_type", type=str, default="sereega", help="simulation type used for training"
)
parser.add_argument(
    "-train_simu_name", type=str, default="eval", help="name of the simulation used for training" 
)
parser.add_argument(
    "-n_train_samples", type=int, default=-1, help="number of training samples, if <0 : number of samples in the params file will be used"
)
parser.add_argument(
    "-train_sfolder", type=str, default="eval", help="name of the folder in which network are saved"
)
parser.add_argument(
    "-train_run_dir",
    type=str,
    default=None,
    help=(
        "Optional path to a specific training run directory (containing a "
        "`trained_models/` subfolder). If provided, NN model weights are loaded from "
        "this folder instead of being inferred from -train_* parameters."
    ),
)
parser.add_argument(
    "-ckpt_path",
    action="append",
    default=[],
    help=(
        "Override model weights with explicit checkpoints."
        " Format: METHOD:/abs/path/to/checkpoint (repeat flag per method)."
    ),
)
parser.add_argument(
    "-inter_layer", type=int, default=2048, help="number of channels of the 1dcnn"
)
parser.add_argument(
    "-kernel_size", type=int, default=5, help="kernel size of the 1dcnn"
)
# 
parser.add_argument(
    "-mets", "--methods", nargs="+", help="methods to use", default=methods
)
parser.add_argument(
    "-sfolder",
    type=str,
    default=f"valid_{datetime.datetime.now().year}-{datetime.datetime.now().month}-{datetime.datetime.now().day}",
    help="Name of the folder to create to save results.",
)
parser.add_argument(
    "-save_suff", type=str, default=save_suffix, help="suffix to save metric values"
)

args = parser.parse_args()
# ----------------------------------------------------------------------#
def _normalize_methods(method_list):
    """Accept a few common aliases for method names."""
    normalized = []
    for m in method_list:
        ml = m.lower()
        if ml in ("deepsif", "deep_sif", "deep-sif"):
            normalized.append("deep_sif")
        elif ml in ("cnn1d", "cnn_1d", "1dcnn", "1d_cnn"):
            normalized.append("cnn_1d")
        elif ml in ("vit", "eegvit", "eeg_vit", "transformer"):
            normalized.append("eeg_vit")
        else:
            normalized.append(m)
    return normalized


def _pick_model_path_from_run_dir(run_dir: str, method: str) -> str:
    """
    Find a `.pt` model weights file inside a training run directory.
    Expected structure (as produced by `main_train.py`):
      <run_dir>/trained_models/<MODEL>_model.pt
    """
    trained_models_dir = os.path.join(run_dir, "trained_models")
    candidates = []
    if method == "cnn_1d":
        candidates = [
            os.path.join(trained_models_dir, "1dcnn_model.pt"),
            os.path.join(trained_models_dir, "1DCNN_model.pt"),
            os.path.join(trained_models_dir, "CNN1D_model.pt"),
            os.path.join(trained_models_dir, "cnn_1d_model.pt"),
        ]
    elif method == "lstm":
        candidates = [
            os.path.join(trained_models_dir, "lstm_model.pt"),
            os.path.join(trained_models_dir, "LSTM_model.pt"),
        ]
    elif method == "deep_sif":
        candidates = [
            os.path.join(trained_models_dir, "DEEPSIF_model.pt"),
            os.path.join(trained_models_dir, "deepsif_model.pt"),
            os.path.join(trained_models_dir, "DeepSIF_model.pt"),
            os.path.join(trained_models_dir, "deep_sif_model.pt"),
        ]
    elif method == "eeg_vit":
        candidates = [
            os.path.join(trained_models_dir, "VIT_model.pt"),
            os.path.join(trained_models_dir, "vit_model.pt"),
            os.path.join(trained_models_dir, "EEGVIT_model.pt"),
            os.path.join(trained_models_dir, "eegvit_model.pt"),
            os.path.join(trained_models_dir, "eeg_vit_model.pt"),
            os.path.join(trained_models_dir, "TRANSFORMER_model.pt"),
        ]

    for p in candidates:
        if os.path.exists(p):
            return p

    if os.path.isdir(trained_models_dir):
        pt_files = [
            f
            for f in os.listdir(trained_models_dir)
            if isinstance(f, str) and f.lower().endswith(".pt")
        ]
        if len(pt_files) == 1:
            return os.path.join(trained_models_dir, pt_files[0])

        # try a soft keyword match if multiple pt files exist
        kw = {"cnn_1d": "cnn", "lstm": "lstm", "deep_sif": "sif", "eeg_vit": "vit"}.get(method, "")
        if kw:
            for f in pt_files:
                if kw in f.lower():
                    return os.path.join(trained_models_dir, f)

    raise FileNotFoundError(
        f"Could not locate model weights for method '{method}' under:\n"
        f"- {trained_models_dir}\n"
        "Expected something like `<run_dir>/trained_models/<MODEL>_model.pt`."
    )


def _parse_ckpt_overrides(entries):
    overrides = {}
    for entry in entries or []:
        if ":" not in entry:
            raise ValueError(
                "-ckpt_path entries must look like METHOD:/abs/path/to/checkpoint"
            )
        method, path = entry.split(":", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(
                "-ckpt_path entries require both a method name and a checkpoint path"
            )
        method_norm = _normalize_methods([method])[0]
        overrides[method_norm] = path
    return overrides


# Normalize methods early so we can decide whether MNE is needed.
try:
    ckpt_overrides = _parse_ckpt_overrides(args.ckpt_path)
except ValueError as exc:
    parser.error(str(exc))
methods_requested = _normalize_methods(args.methods)

# Only linear inverse methods require `mne`
USE_MNE_LINEAR = HAS_MNE and any(m in linear_methods for m in methods_requested)
if any(m in linear_methods for m in methods_requested) and not HAS_MNE:
    # If user explicitly requested methods, fail loudly. If they used defaults,
    # silently drop linear methods so the script can run without MNE.
    user_set_methods = any(f in sys.argv for f in ("-mets", "--methods"))
    if user_set_methods:
        sys.exit(
            "You selected linear methods (MNE/sLORETA) but `mne` is not installed. "
            "Install `mne` or remove linear methods from `-mets`."
        )
    methods_requested = [m for m in methods_requested if m not in linear_methods]
    USE_MNE_LINEAR = False
#----------------------------------------------------------------------#
root_simu = args.root_simu
results_path = args.results_path
dataset = f"{args.simu_name}{args.source_space}_"
eval_results_path = f"{results_path}/{dataset}/eval/{args.sfolder}"
os.makedirs(eval_results_path, exist_ok=True)

##----------------LOAD EVAL DATA---------------------##
root_simu_path = Path(root_simu)
if (root_simu_path / "simulation" / args.subject_name).is_dir():
    root_base = root_simu_path / "simulation" / args.subject_name
else:
    root_base = root_simu_path

simu_path = str(
    root_base
    / args.orientation
    / args.electrode_montage
    / args.source_space
    / "simu"
    / args.simu_name
)
model_path = str(
    root_base / args.orientation / args.electrode_montage / args.source_space / "model"
)
config_file = f"{simu_path}/{args.simu_name}{args.source_space}_config.json"

with open(config_file, "r") as f:
    general_config_dict = json.load(f)
general_config_dict["eeg_snr"] = args.eeg_snr
general_config_dict["simu_name"] = args.simu_name

folders = FolderStructure(str(root_base), general_config_dict)
source_space = HeadModel.SourceSpace(folders, general_config_dict)
electrode_space = HeadModel.ElectrodeSpace(folders, general_config_dict)
head_model = HeadModel.HeadModel(electrode_space, source_space, folders, "fsaverage")

if args.leadfield_mat:
    fwd = _load_leadfield_mat(args.leadfield_mat)
elif args.source_space == "fsav_994":
    fwd = loadmat(f"{model_path}/LF_fsav_994.mat")["G"]
else:
    fwd = head_model.fwd["sol"]["data"]

# Ensure consistent dtype for torch matmul (avoid float64 from .mat files)
fwd = np.asarray(fwd, dtype=np.float32)

## open neighbors file if it was already re-shaped
if os.path.isfile(f"{folders.model_folder}/fs_cortex_neighbors_994.mat"):
    neighbors = (
        loadmat(f"{folders.model_folder}/fs_cortex_neighbors_994.mat")["nbs"] - 1
    )
else:  # reshape and save otherwise
    neighbors = loadmat(f"{folders.model_folder}/fs_cortex_20k_region_mapping.mat")[
        "nbs"
    ][0]
    ### reformater
    m = -1
    for n in neighbors:
        if n.shape[1] > m:
            m = n.shape[1]
    neighbors_ref = np.ones((neighbors.shape[0], m), dtype=int) * (-1)
    for r in range(neighbors_ref.shape[0]):
        nbs = neighbors[r][0]
        neighbors_ref[r, : len(nbs)] = nbs

    from scipy.io import savemat

    savemat(
        f"{folders.model_folder}/fs_cortex_neighbors_994.mat", {"nbs": neighbors_ref}
    )


fs = general_config_dict["rec_info"]["fs"]
n_times = general_config_dict["rec_info"]["n_times"]
t_vec = np.arange(0, n_times / fs, 1 / fs)
spos = torch.from_numpy(source_space.positions)  # in meter
mne_info = getattr(head_model.electrode_space, "info", None)
if mne_info is None or getattr(mne_info, "nchan", None) != fwd.shape[0]:
    ch_names = [f"EEG{c:03d}" for c in range(1, fwd.shape[0] + 1)]
    mne_info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg", verbose=False)

### load the 2 source spaces and region mapping
if USE_MNE_LINEAR:
    fwd_vertices = mne.read_forward_solution(
        f"{folders.model_folder}/fwd_verticesfsav_994-fwd.fif"
    )
    fwd_vertices = mne.convert_forward_solution(
        fwd_vertices, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
    )
    fwd_regions = mne.read_forward_solution(f"{folders.model_folder}/fwd_fsav_994-fwd.fif")
    fwd_regions = mne.convert_forward_solution(
        fwd_regions, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
    )

    ## assign fwd_region the proper leadfield matrix values (summed version)
    fwd_regions["sol"]["data"] = fwd

    region_mapping = loadmat(f"{folders.model_folder}/fs_cortex_20k_region_mapping.mat")[
        "rm"
    ][0]
    n_vertices = fwd_vertices["nsource"]
    n_regs = len(np.unique(region_mapping))
else:
    # For NN-only evaluation on regional source spaces (e.g. 994 regions),
    # we never need to expand regions back to the 20k-vertex surface.
    fwd_vertices = None
    fwd_regions = None
    region_mapping = None
    n_vertices = fwd.shape[1]
    n_regs = fwd.shape[1]
####################################################################
## load dataset
if args.eval_simu_type.upper() == "NMM":
    spikes_data_path = f"{root_simu}/{args.orientation}/{args.electrode_montage}/{args.source_space}/simu/{args.spikes_folder}"
    dataset_meta_path = f"{simu_path}/{args.simu_name}.mat"

    ds_dataset = ModSpikeEEGBuild(
        spike_data_path=spikes_data_path,
        metadata_file=dataset_meta_path,
        fwd=fwd,
        n_times=args.n_times,
        args_params={"dataset_len": args.to_load},
        spos=source_space.positions,
        norm=args.scaler,
    )

elif args.eval_simu_type.upper() == "SEREEGA":
    ds_dataset = EsiDatasetds_new(
        root_simu,
        config_file,
        args.simu_name,
        args.source_space,
        general_config_dict["electrode_space"]["electrode_montage"],
        args.to_load,
        args.eeg_snr,
        noise_type={"white": 1.0, "pink": 0.0},
    )

else:
    sys.exit("unknown simulation type (argument simu_type)")

n_electrodes = fwd.shape[0]
n_sources = fwd.shape[1]
# split dataset
_, val_ds = random_split(ds_dataset, [1 - args.per_valid, args.per_valid])
val_dataloader = DataLoader(dataset=val_ds, batch_size=1, shuffle=False)
n_val_samples = len(val_dataloader)
print(f">>>>>>>>>>>> Evaluation on {n_val_samples} samples <<<<<<<<<<<<<<<<<")
##------------------------------------------------------------------------##
##------------------------------------------------------------------------##
## trained neural networks parameters
if args.net_from_file : 
    with open(args.params_nn, "r") as f:
        params_file = yaml.safe_load(f)

    cnn1d_params    = params_file["cnn1d"]
    lstm_params     = params_file["lstm"]
    deep_sif_params = params_file["deep_sif"]
    vit_params      = params_file.get("eeg_vit", deep_sif_params)
#if args.n_train_samples > 0 :
#    cnn1d_params['n_train_samples'] = args.n_train_samples
#    lstm_params['n_train_samples'] = args.n_train_samples
#    deep_sif_params['n_train_samples'] = args.n_train_samples
#if args.dataset : 
#    cnn1d_params['dataset'] = f"{args.simu_name}{args.source_space}_"
#    lstm_params['dataset'] = f"{args.simu_name}{args.source_space}_"
#    deep_sif_params['dataset'] = f"{args.simu_name}{args.source_space}_"

else : 
    train_dataset = f"{args.train_simu_name}{args.source_space}_"
    train_params = {
        "train_simu_type" : args.train_simu_type,
        "inter_layer" : args.inter_layer,
        "kernel_size" : args.kernel_size,
        "n_epochs" : args.n_epochs, 
        "batch_size" : args.train_bs, 
        "dataset" : train_dataset, 
        "exp" : args.train_sfolder,
        "n_train_samples" : args.n_train_samples,
        "loss" :args.train_loss,
        "norm" : args.scaler, 
        "n_electrodes" : n_electrodes, 
        "n_sources" : n_sources, 
        "hidden_size" : 85, 
        "temporal_input_size" : 500, 
    }
    cnn1d_params = train_params
    lstm_params = train_params
    deep_sif_params = train_params
    vit_params = train_params


##############################################################################################################################################
############### load networks

methods = methods_requested

if "cnn_1d" in methods:
    if args.net_from_file : 
        train_dataset = cnn1d_params['dataset']
    train_results_path = f"{results_path}/{train_dataset}"

    from models.cnn_1d import CNN1Dpl as cnn1d_net

    # Compare against the actually used forward model (`fwd`), not the MNE head model.
    if (cnn1d_params["n_electrodes"] != n_electrodes) or (cnn1d_params["n_sources"] != n_sources):
        sys.exit(
            (
                f"number of electrodes or sources in head model does not match with number of electrodes or sources in 1dcnn model"
                f"electrodes fwd : {n_electrodes} - electrodes 1dcnn : {cnn1d_params['n_electrodes']}\n"
                f"sources fwd : {n_sources} - sources 1dcnn : {cnn1d_params['n_sources']}"
            )
        )

    cnn_model_name = (
        f"simu_{args.train_simu_type}_"
        f"srcspace_{head_model.source_space.src_sampling}"
        f"_model_1dcnn"
        f"_interlayer_{ cnn1d_params['inter_layer'] }"
        f"_trainset_{cnn1d_params['n_train_samples']}"
        f"_epochs_{cnn1d_params['n_epochs']}"
        f"_loss_{cnn1d_params['loss']}"
        f"_norm_{cnn1d_params['norm']}.pt"
    )
    if args.train_run_dir:
        cnn_model_path = _pick_model_path_from_run_dir(args.train_run_dir, "cnn_1d")
    else:
        cnn_model_path = f"{train_results_path}/trained_models/{cnn1d_params['exp']}/{cnn_model_name}"
    cnn_model_path = ckpt_overrides.get("cnn_1d", cnn_model_path)
    if os.path.exists(cnn_model_path):
        print("CNN model is available for use")
    else:
        sys.exit(
            f"{cnn_model_path} \nCNN model is not accessible.\nTry other parameters or train your model first."
        )

    # from models.CNN1d_v1 import simple_1dCNN_v2 as cnn1d_net
    from models.cnn_1d import CNN1Dpl as cnn1d_net

    net_parameters = {
        "channels": [
            cnn1d_params["n_electrodes"],
            cnn1d_params["inter_layer"],
            cnn1d_params["n_sources"],
        ],
        "kernel_size": cnn1d_params["kernel_size"],
        "bias": False,
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,  # nn.MSELoss(reduction='sum'), #CosineSimilarityLoss(),
        # "dropout_rate" : 0.2
    }
    cnn = cnn1d_net(**net_parameters)
    _load_module_weights(cnn, cnn_model_path)
    cnn.eval()


if "lstm" in methods:
    if args.net_from_file : 
        train_dataset = lstm_params['dataset']
    train_results_path = f"{results_path}/{train_dataset}"
    # Compare against the actually used forward model (`fwd`), not the MNE head model.
    if (lstm_params["n_electrodes"] != n_electrodes) or (lstm_params["n_sources"] != n_sources):
        sys.exit(
            (
                f"number of electrodes or sources in head model does not match with number of electrodes or sources in lstm model"
                f"electrodes fwd : {n_electrodes} - electrodes lstm : {lstm_params['n_electrodes']}"
                f"sources fwd : {n_sources} - sources lstm : {lstm_params['n_sources']}"
            )
        )

    lstm_model_name = (
        f"simu_{args.train_simu_type}_"
        f"srcspace_{head_model.source_space.src_sampling}"
        f"_model_lstm"
        f"_trainset_{lstm_params['n_train_samples']}"
        f"_epochs_{lstm_params['n_epochs']}"
        f"_loss_{lstm_params['loss']}"
        f"_norm_{lstm_params['norm']}.pt"
    )
    if args.train_run_dir:
        lstm_model_path = _pick_model_path_from_run_dir(args.train_run_dir, "lstm")
    else:
        lstm_model_path = f"{train_results_path}/trained_models/{lstm_params['exp']}/{lstm_model_name}"
    lstm_model_path = ckpt_overrides.get("lstm", lstm_model_path)
    if os.path.exists(lstm_model_path):
        print("LSTM model is available for use")
    else:
        sys.exit(
            "LSTM model is not accessible.\nTry other parameters or train your model first."
        )

    # from models.lstm import HeckerLSTM as lstm_net
    from models.lstm import HeckerLSTMpl as lstm_net

    net_parameters = {
        "n_electrodes": lstm_params["n_electrodes"],
        "hidden_size": lstm_params["hidden_size"],
        "n_sources": lstm_params["n_sources"],
        "bias": False,
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,  # nn.MSELoss(reduction='sum'), #CosineSimilarityLoss(),
        "mc_dropout_rate": 0,
    }

    lstm = lstm_net(**net_parameters)
    _load_module_weights(lstm, lstm_model_path)
    lstm.eval()


if "deep_sif" in methods:
    if args.net_from_file : 
        train_dataset = deep_sif_params['dataset']
    train_results_path = f"{results_path}/{train_dataset}"
    # Compare against the actually used forward model (`fwd`), not the MNE head model.
    if (deep_sif_params["n_electrodes"] != n_electrodes) or (deep_sif_params["n_sources"] != n_sources):
        sys.exit(
            (
                f"number of electrodes or sources in head model does not match with number of electrodes or sources in deep sif model"
                f"electrodes fwd : {n_electrodes} - electrodes deep sif : {deep_sif_params['n_electrodes']}"
                f"sources fwd : {n_sources} - sources deep sif : {deep_sif_params['n_sources']}"
            )
        )

    deep_sif_model_name = (
        f"simu_{args.train_simu_type}_"
        f"srcspace_{head_model.source_space.src_sampling}"
        f"_model_deepsif"
        f"_trainset_{deep_sif_params['n_train_samples']}"
        f"_epochs_{deep_sif_params['n_epochs']}"
        f"_loss_{deep_sif_params['loss']}"
        f"_norm_{deep_sif_params['norm']}.pt"
    )
    if args.train_run_dir:
        deep_sif_model_path = _pick_model_path_from_run_dir(args.train_run_dir, "deep_sif")
    else:
        deep_sif_model_path = f"{train_results_path}/trained_models/{deep_sif_params['exp']}/{deep_sif_model_name}"
    deep_sif_model_path = ckpt_overrides.get("deep_sif", deep_sif_model_path)
    if os.path.exists(deep_sif_model_path):
        print("DEEP SIF model is available for use")
    else:
        sys.exit(
            f"DEEP SIF model is not accessible.\nTry other parameters or train your model first.\n{deep_sif_model_path}"
        )

    net_parameters = {
        "num_sensor": deep_sif_params["n_electrodes"],
        "num_source": deep_sif_params["n_sources"],
        "temporal_input_size": deep_sif_params["temporal_input_size"],
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,
    }

    from models.deepsif import DeepSIFpl as deep_sif_net

    deep_sif = deep_sif_net(**net_parameters)
    _load_module_weights(deep_sif, deep_sif_model_path)
    deep_sif.eval()

if "eeg_vit" in methods:
    train_results_path = f"{results_path}/{train_dataset}"
    if (vit_params["n_electrodes"] != n_electrodes) or (vit_params["n_sources"] != n_sources):
        sys.exit(
            (
                "number of electrodes or sources in fwd does not match with number of electrodes or sources in eeg_vit model"
                f"electrodes fwd : {n_electrodes} - electrodes eeg_vit : {vit_params['n_electrodes']}"
                f"sources fwd : {n_sources} - sources eeg_vit : {vit_params['n_sources']}"
            )
        )

    if args.train_run_dir:
        vit_model_path = _pick_model_path_from_run_dir(args.train_run_dir, "eeg_vit")
    else:
        # fallback naming (similar pattern as others)
        vit_model_name = (
            f"simu_{args.train_simu_type}_"
            f"srcspace_{head_model.source_space.src_sampling}"
            f"_model_vit"
            f"_trainset_{vit_params['n_train_samples']}"
            f"_epochs_{vit_params['n_epochs']}"
            f"_loss_{vit_params['loss']}"
            f"_norm_{vit_params['norm']}.pt"
        )
        vit_model_path = f"{train_results_path}/trained_models/{vit_params['exp']}/{vit_model_name}"
    vit_model_path = ckpt_overrides.get("eeg_vit", vit_model_path)

    if os.path.exists(vit_model_path):
        print("EEGViT model is available for use")
    else:
        sys.exit(
            f"EEGViT model is not accessible.\nTry other parameters or train your model first.\n{vit_model_path}"
        )

    from models.vit import EEGViTpl as vit_net

    net_parameters = {
        "num_sensor": vit_params["n_electrodes"],
        "num_source": vit_params["n_sources"],
        "n_times": vit_params.get("n_times", 500),
        "embed_dim": vit_params.get("vit_embed_dim", 256),
        "depth": vit_params.get("vit_depth", 6),
        "num_heads": vit_params.get("vit_heads", 8),
        "mlp_dim": vit_params.get("vit_mlp_dim", 512),
        "dropout": vit_params.get("vit_dropout", 0.1),
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,
    }
    eeg_vit = vit_net(**net_parameters)
    _load_module_weights(eeg_vit, vit_model_path)
    eeg_vit.eval()

##################################################################################################
# to save metric values
methods.append("gt")
nmse_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
loc_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
psnr_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
time_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
auc_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}

##############################################################
########## test for noise covariance estimation #############
"""
# 1. take 10 random samples in the dataset
# 2. perform SVD and noise decomposition on each and concatenate noise signals
# 3. use the concatenated results to estimate noise cov
from scipy.linalg import svd
rand_samp = np.random.randint( 0, len(val_ds), 10 )
sig_noise = np.zeros((n_electrodes, n_times))
for i in rand_samp : 
    sig = val_ds[i][0].numpy()
    u,s,v = svd( sig.transpose() )
    noise = sig.transpose() - np.dot( u[:,:2] * s[:2], v[:2,:] )
    sig_noise = np.concatenate( (sig_noise, noise.transpose()), axis=1 )
raw_noise = mne.io.RawArray(data = sig_noise, info=mne_info)
noise_cov = mne.compute_raw_covariance(raw_noise)
"""

######################## DO THE EVALUATION

noise_only_eeg_data = []
#################################
if args.eval_simu_type.lower() == "sereega":
    md_keys = [k for k, _ in val_ds.dataset.md_dict.items()]
c = 0
nf=0
overlapping_regions = 0
for k in val_ds.indices:
    M, j = val_ds.dataset[k]
    M, j = M.float(), j.float()

    M_unscaled = M * val_ds.dataset.max_eeg[k]
    j_unscaled = j * val_ds.dataset.max_src[k]

    j_unscaled_vertices = None
    if region_mapping is not None:
        j_unscaled_vertices = np.zeros((n_vertices, n_times))
        for r in range(n_regs):
            j_unscaled_vertices[np.where(region_mapping == r)[0], :] = j_unscaled[r, :]

    # data covariance:
    # activity_thresh = 0.1
    # noise_cov, data_cov, nap = inv.mne_compute_covs(
    #    (M_unscaled).numpy(), mne_info, activity_thresh
    # )

    # Noise covariance / EEG object are only needed for linear MNE inverses.
    if USE_MNE_LINEAR:
        raw_noise = mne.io.RawArray(
            data=np.random.randn(n_electrodes, 600),
            info=mne_info,
            verbose=False,
        )
        noise_cov = mne.compute_raw_covariance(raw_noise, verbose=False)

        eeg = mne.io.RawArray(
            data=M, info=mne_info, first_samp=0.0, verbose=False
        )
        eeg = mne.set_eeg_reference(eeg, "average", projection=True, verbose=False)[0]
    else:
        noise_cov = None
        eeg = None

    ## ici il y a un distinction à faire selon les jeux de données
    if args.eval_simu_type.lower() == "sereega":
        seeds = val_ds.dataset.md_dict[md_keys[k]]["seeds"]
        if type(seeds) is int:
            seeds = [seeds]
    else:
        seeds = list(val_ds.dataset.dataset_meta["selected_region"][k][:, 0])
        if type(seeds) is int:
            seeds = [seeds]

        # stc_gt = mne.SourceEstimate(
        #    data=j_unscaled_vertices, # 256//2 = instant du pic à visualiser @TODO : change le codage en dur
        #    vertices= [ fwd_vertices['src'][0]['vertno'], fwd_vertices['src'][1]['vertno'] ],
        #    tmin=0.,
        #    tstep=1/fs,
        #    subject="fsaverage"
        # )

    # compute the diverse inverse solutions
    for method in methods:
        if method == "gt":
            j_hat = j_unscaled
        # compute inverse solution
        elif method in linear_methods:
            lambda2 = 1.0 / (args.eeg_snr**2)
            inv_op = mne.minimum_norm.make_inverse_operator(
                info=eeg.info,
                forward=fwd_regions,
                noise_cov=noise_cov,
                loose=0,
                depth=0,
                verbose=False,
            )
            stc_hat = mne.minimum_norm.apply_inverse_raw(
                raw=eeg,
                inverse_operator=inv_op,
                lambda2=lambda2,
                method=method,
                verbose=False,
            )

            j_hat = torch.from_numpy(stc_hat.data)

        elif method == "cnn_1d":
            with torch.no_grad():
                j_hat = cnn.model(M.unsqueeze(0)).squeeze()
            j_hat = rescale_prediction(
                j_hat, M_unscaled, fwd, cnn1d_params["loss"], val_ds.dataset.max_src[k]
            )

        elif method == "lstm":
            with torch.no_grad():
                j_hat = lstm(M.unsqueeze(0)).squeeze()
            j_hat = rescale_prediction(
                j_hat, M_unscaled, fwd, lstm_params["loss"], val_ds.dataset.max_src[k]
            )

        elif method == "deep_sif":
            with torch.no_grad():
                j_hat = deep_sif(M.unsqueeze(0)).squeeze()
            j_hat = rescale_prediction(
                j_hat, M_unscaled, fwd, deep_sif_params["loss"], val_ds.dataset.max_src[k]
            )

        elif method == "eeg_vit":
            with torch.no_grad():
                j_hat = eeg_vit(M.unsqueeze(0)).squeeze()
            j_hat = rescale_prediction(
                j_hat,
                M_unscaled,
                fwd,
                vit_params.get("loss", args.train_loss),
                val_ds.dataset.max_src[k],
            )

        else:
            sys.exit(f"unrecognized method {method}")

        # -------------------- Metrics for this method -------------------- #
        # dataset-dependent seeds / patches
        if args.eval_simu_type.lower() == "sereega":
            seeds = val_ds.dataset.md_dict[md_keys[k]]["seeds"]
            if type(seeds) is int:
                seeds = [seeds]
        else:
            seeds = list(val_ds.dataset.dataset_meta["selected_region"][k][:, 0])
            seeds = [s.astype(int) for s in seeds]
            if type(seeds) is int:
                seeds = [seeds]

        patches = [[] for _ in range(len(seeds))]
        if args.eval_simu_type.lower() == "nmm":
            for kk in range(len(seeds)):
                patches[kk] = utl.get_patch(order=3, idx=seeds[kk], neighbors=neighbors)
        else:
            for kk in range(len(seeds)):
                patches[kk] = val_ds.dataset.md_dict[md_keys[k]]["act_src"][f"patch_{kk+1}"]

        r = sample_metrics(j, j_unscaled, j_hat, seeds, patches, spos, neighbors, t_vec)
        if r["overlap"]:
            overlapping_regions += 1

        time_error_dict[method][c] = r["te"]
        loc_error_dict[method][c] = r["le"]
        nmse_dict[method][c] = r["nmse"]
        auc_dict[method][c] = r["auc"]
        psnr_dict[method][c] = r["psnr"]

    c += 1

    if c%100 == 0 : 
        print(f"---------------{c} validation samples done -------------")
############################################################################
if len(noise_only_eeg_data) > 0:
    for method in methods:
        nmse_dict[method] = np.delete(nmse_dict[method], noise_only_eeg_data)
        loc_error_dict[method] = np.delete(loc_error_dict[method], noise_only_eeg_data)
        auc_dict[method] = np.delete(auc_dict[method], noise_only_eeg_data)
        time_error_dict[method] = np.delete(
            time_error_dict[method], noise_only_eeg_data
        )
        psnr_dict[method] = np.delete(psnr_dict[method], noise_only_eeg_data)
#####################################################################
#############################################################################


for method in methods:
    print(f" >>>>>>>>>>>>>>> Results method {method} <<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"mean time error: {time_error_dict[method].mean()*1e3} [ms]")
    print(f"mean localisation error: {loc_error_dict[method].mean()*1e3} [mm]")
    print(f"mean nmse at instant of max activity: {nmse_dict[method].mean():.4f}")
    print(f"psnr for total source distrib: {psnr_dict[method].mean():.4f} [dB]")
    print(f"auc: {auc_dict[method].mean():.4f}")


##################################################################################################"
#os.makedirs(f"{eval_results_path}/{dataset}/eval/{args.sfolder}", exist_ok=True)
#  Save into a csv file

for method in methods:
    if method == "cnn_1d":
        method_info = cnn_model_name
    elif method == "lstm":
        method_info = lstm_model_name
    elif method == "deep_sif":
        method_info = deep_sif_model_name
    else:
        method_info = "none"
    my_values = [
        {
            "simu_name": args.simu_name,
            "src_space": head_model.source_space.src_sampling,
            "method": method,
            "method_info": method_info,
            "valset": str(n_val_samples),
            "noise db": f"{args.eeg_snr}",
            "mean nmse": f"{nmse_dict[method].mean()}",
            "std nmse": f"{nmse_dict[method].std()}",
            "mean loc error": f"{loc_error_dict[method].mean()}",
            "std loc error": f"{loc_error_dict[method].std()}",
            "mean auc": f"{auc_dict[method].mean()}",
            "std auc": f"{auc_dict[method].std()}",
            "mean time error": f"{time_error_dict[method].mean()}",
            "std time error": f"{time_error_dict[method].std()}",
            "mean psnr": f"{psnr_dict[method].mean()}",
            "std psnr": f"{psnr_dict[method].std()}",
        }
    ]

    fields = [
        list(my_values[0].keys())[k] for k in range(len(list(my_values[0].keys())))
    ]

    import csv

    suffix_save_metrics = (
        f"train_simu_{args.train_simu_type}_"
        f"eval_simu_{args.eval_simu_type}_"
        f"method_{method}"
        f"_srcspace_{head_model.source_space.src_sampling}"
        f"_dataset{args.simu_name}"
        f"_n_train_{args.n_train_samples}"
        f"{args.save_suff}"
    )

    with open(
        f"{eval_results_path}/evaluation_metrics_{suffix_save_metrics}.csv",
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(my_values)

print(
    f">>>>>>> results saved in :{eval_results_path}/evaluation_metrics_{suffix_save_metrics}.csv"
)


################### save all values to then plot distribution ###############################
for method in methods:
    if method == "cnn_1d":
        method_info = cnn_model_name
    elif method == "lstm":
        method_info = lstm_model_name
    elif method == "deep_sif":
        method_info = deep_sif_model_name
    else:
        method_info = "none"
    my_values = {
        "nmse": np.squeeze(nmse_dict[method]),
        "loc error": np.squeeze(loc_error_dict[method]),
        "auc": np.squeeze(auc_dict[method]),
        "time error": np.squeeze(time_error_dict[method]),
        "psnr": np.squeeze(psnr_dict[method]),
    }

    df = pd.DataFrame(data=my_values)

    # fields = [
    #    list(my_values[0].keys())[k] for k in range(len(list(my_values[0].keys())))
    # ]

    suffix_save_metrics = (
        f"train_simu_{args.train_simu_type}_"
        f"eval_simu_{args.eval_simu_type}_"
        f"method_{method}"
        f"_srcspace_{head_model.source_space.src_sampling}"
        f"_dataset{args.simu_name}"
        f"_n_train_{args.n_train_samples}"
        f"{args.save_suff}"
    )

    df.to_csv(
        f"{eval_results_path}/evaluation_{suffix_save_metrics}.csv"
    )

    # with open(
    #    f"{home}/Documents/Results/{dataset}/eval/{args.sfolder}/evaluation_{suffix_save_metrics}.csv",
    #    "w",
    # ) as csvfile:
    # writer = csv.DictWriter(csvfile, fieldnames=fields)
    # writer.writeheader()
    # writer.writerows(my_values)

print(
    f">>>>>>> results saved in :{eval_results_path}/evaluation_{suffix_save_metrics}.csv"
)


###################################### plot distribution and save figs #######################################
# TODO
