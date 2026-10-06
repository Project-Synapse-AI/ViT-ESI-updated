import os
import time
import argparse

import numpy as np
from scipy.io import loadmat, savemat
import json
import utils

## PARAMETERS
home = os.path.expanduser('~')
root_folder = os.path.join(home, "Documents", "Data")

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

### @TODO : faire des subparser ou trouver une façon un peu mieux que ce gros parser là de passer les paramètres
parser.add_argument("-sin", "--simu_name", type=str, required=True, help="Name of the simulation to load")
parser.add_argument("-ne", "--n_examples", type=int, required=True, help="Number of samples to create")
# Head model parameters
parser.add_argument("-mk", "--montage_kind", type=str, required=True, help="Name of the electrode montage to use")
parser.add_argument("-ss", "--source_sampling", type=str, required=True, help="Name of the source space/source sampling to use")
parser.add_argument("-o", "--orientation", type=str, required=True, help="Orientation of the sources", choices=["constrained", "unconstrained"])
parser.add_argument("-sn", "--subject_name", type=str, required=True, help="name of the subject to use (ex: fsaverage)")

parser.add_argument("-v", "--volume", action='store_true', help="Volume source space")
parser.add_argument("-s", "--sphere", action="store_true", help="Spherical head model")
parser.add_argument("-rf", "--root_folder", type=str, default=root_folder, help="Root of simulation folder")
parser.add_argument(
    "-af", "--anatomy_folder",
    type=str,
    default=None,
    help="Path to anatomy folder (sources_fsav_994.mat, etc.). Default: {root_folder}/anatomy",
)
parser.add_argument(
    "--leadfield_mat",
    type=str,
    default=None,
    help="Optional path to a .mat leadfield to use instead of model/LF_<suf>.mat (e.g. anatomy/leadfield_75_20k.mat).",
)

# recodring/timeline parameters
parser.add_argument("-fs", "--fs", type=int, default=512, help="Sampling frequency [Hz]")
parser.add_argument("-d", "--duree", type=int, default=500, help="Signal duration, in ms")
parser.add_argument("-nt", "--n_trials", type=int, default="1", help="Number of trials in one signal")
# spatial pattern parameters
# patch = region of neighboring active sources
parser.add_argument("-m", "--margin", type=int, default=2, help="extension order of space between different patches")
parser.add_argument("-np_min", "--n_patch_min", type=int, default=1, help="minimum number of patches")
parser.add_argument("-np_max", "--n_patch_max", type=int, default=5, help="maximum number of patches")
parser.add_argument("-o_min", "--order_min", type=int, default=1, help="minimum order of a patch")
parser.add_argument("-o_max", "--order_max", type=int, default=5, help="maximum order of a patch")

## TEMPORAL PATTERN PARAMETERS 
parser.add_argument("-s_type", "--sig_type", type=str, default="erp", help="type of source signal")
parser.add_argument("-amp", "--amplitude", type=float, default=1., help="base amplitude of a source signal")
parser.add_argument("-c", "--center", type=int, default=250, help="center of the ERP signal, in ms" )
parser.add_argument("-w", "--width", type=int, default=50, help="width of the ERP signal, in ms")
# deviation
parser.add_argument("-itradev", "--intra_sample_dev", type=float, nargs=3, default=[0.7, 0.3, 0.1],
                    help="deviation parameters between patches of a given example - amplitude, center, width")
parser.add_argument("-interdev","--inter_sample_dev", type=float, nargs=3, default=[0.5, 0.5, 0.02],
                    help="deviation parameters between examples - amplitude, center, width")

# Other parameters
parser.add_argument("-ds", "--dont_save", action="store_true", help = "Do not save the data")
parser.add_argument("--seed", type=int, default=0,
                    help="numpy random seed (default 0 = the previously hard-coded seed, same output)")

args = parser.parse_args()
np.random.seed(args.seed)
root_folder = args.root_folder
print(f"root_folder: {root_folder}")
anatomy_folder = args.anatomy_folder if args.anatomy_folder is not None else os.path.join(root_folder, "anatomy")
######################################################
if args.orientation == "constrained" : 
    constrained_orientation = True # orientation of sources.
else : 
    constrained_orientation = False

if args.volume :
    suf = f"'vol_'{args.source_sampling}.0"
    constrained_orientation = False #if volume source space the sources are necessarily unconstrained
elif args.sphere :
    suf = f"sphere_{args.source_sampling}.0" 
    constrained_orientation = False
else :
    suf = f"{args.source_sampling}" 

########### CREATE DIRS 
model_path = os.path.join(root_folder, "simulation", args.subject_name, args.orientation, args.montage_kind, suf, "model") 
saving_folder = os.path.join(root_folder, "simulation", args.subject_name, args.orientation, args.montage_kind, suf, "simu") 
suffix_save = os.path.join( args.subject_name, args.orientation, args.montage_kind, suf, "simu", args.simu_name) 
if not args.dont_save :
    saving_folder = os.path.join(saving_folder, args.simu_name) 
    os.makedirs(saving_folder, exist_ok=True) 
    os.makedirs( f"{saving_folder}/sources/Jact", exist_ok=True ) 
    os.makedirs( f"{saving_folder}/sources/Jnoise" , exist_ok = True )
    os.makedirs( f"{saving_folder}/eeg/infdb", exist_ok=True ) 
    os.makedirs( f"{saving_folder}/md", exist_ok=True ) 
    os.makedirs( f"{saving_folder}/timeline", exist_ok=True ) 

#####################################################################
# "recording"/timeline parameterts
n_times     = args.fs*args.duree/1000 # Number of time samples

timeline    = {
    'n': args.n_trials,
    'srate': args.fs,
    'length': args.duree,
    'marker': 'event1',
    'prestim': 0
    }

print(f"Sampling frequency: {args.fs} \nTrial duration (ms): {args.duree} \nNumber of time samples: {n_times} \nNumber of trials: {args.n_trials} \n")

################ signal
erp_dev_intra_patch = {
    'ampl': 0, 
    'width' : 0,
    'center': 0  } 

# Variation between samples: 
# - variation in amplitude
# - variation in center position
# - variation in width
s_amplitude_dev   = args.inter_sample_dev[0]
s_center_dev      = args.inter_sample_dev[1]
s_width_dev       = args.inter_sample_dev[2]
base_amplitude  = args.amplitude
base_width      = args.width
base_center     = args.center

s_range_ampl = np.array( [base_amplitude - s_amplitude_dev*base_amplitude,
                         base_amplitude + s_amplitude_dev*base_amplitude] )
s_range_width = np.array( [base_width - s_width_dev*base_width, base_width + s_width_dev*base_width ])
s_range_center = np.array([ base_center - s_center_dev*base_center,
    base_center + s_center_dev*base_center ])

# Variation between patches of a same sample
p_amplitude_dev   = args.intra_sample_dev[0]
p_center_dev      = args.intra_sample_dev[1]
p_width_dev       = args.intra_sample_dev[2]


#####################################################################
## Load anatomy data
# get the data using the unpack_fwdModel function

def _load_leadfield_mat(mat_path: str) -> np.ndarray:
    m = loadmat(mat_path)
    # Common keys in this codebase: 'G' (leadfield), or 'fwd' (leadfield_75_20k.mat)
    if "G" in m:
        return m["G"]
    if "fwd" in m:
        return m["fwd"]
    # Fallback: first 2D numeric array
    for k, v in m.items():
        if k.startswith("__"):
            continue
        if isinstance(v, np.ndarray) and v.ndim == 2:
            return v
    raise KeyError(f"No 2D leadfield found in {mat_path}. Keys: {[k for k in m.keys() if not k.startswith('__')]}")


def _compute_knn_neighbors(positions: np.ndarray, k: int = 10) -> np.ndarray:
    """Simple kNN adjacency for region-level source spaces."""
    pos = np.asarray(positions)
    n = pos.shape[0]
    nbs = np.zeros((n, k), dtype=np.int64)
    # brute-force (n=994 is fine)
    for i in range(n):
        d = np.sum((pos - pos[i]) ** 2, axis=1)
        d[i] = np.inf
        nbs[i] = np.argsort(d)[:k]
    return nbs

# Leadfield + source positions
leadfield = None
spos = None

# Optional override leadfield (e.g. 75x994 regional leadfield)
if args.leadfield_mat is not None:
    leadfield = _load_leadfield_mat(args.leadfield_mat)
    # If regional (994), load matching region positions from anatomy folder.
    if leadfield.shape[1] == 994:
        src_994_path = os.path.join(anatomy_folder, "sources_fsav_994.mat")
        spos = loadmat(src_994_path)["positions"]
else:
    # Default: load source positions + leadfield from the head-model folder
    src = loadmat(os.path.join(model_path, f"sources_{suf}.mat"))
    spos = src["positions"]
    leadfield = loadmat(os.path.join(model_path, f"LF_{suf}.mat"))["G"]

## Compute neighbors from the mesh triangle data
#tlh     = loadmat(os.path.join(folder_path,f"tris_lh_{suf}.mat"))
#trh     = loadmat(os.path.join(folder_path, f"tris_rh_{suf}.mat"))
#verts = loadmat(os.path.join(folder_path, f"verts_{suf}.mat"))
#verts = {"lh": verts['verts_lh'], "rh": verts['verts_rh']} 
#tris = {"lh": tlh['tris_lh'] , "rh": trh['tris_rh']}
#tris = np.array([np.squeeze(tlh['tris_lh']), np.squeeze(trh['tris_rh'])])
#verts = np.array([np.squeeze(verts['lh']),np.squeeze(verts['rh']) ])
import mne

n_sources = leadfield.shape[1]
n_electrodes = leadfield.shape[0]

# If leadfield is the 994-region space, use the matching regional source positions
# from the repo-level anatomy folder and compute neighbors via kNN.
if n_sources == 994 and args.leadfield_mat is not None:
    # spos already loaded above
    neighbors = _compute_knn_neighbors(spos, k=10)
else:
    # Vertex/ico source spaces: use MNE forward for mesh neighbors
    fwd = mne.read_forward_solution(
        f"{model_path}/fwd_{args.source_sampling}-fwd.fif",
        verbose=False,
    )
    # constrain source orientation if necessary
    constrained = True
    if args.orientation == "unconstrained":
        constrained = False
    fwd = mne.convert_forward_solution(
        fwd,
        surf_ori=constrained,
        force_fixed=constrained,
        use_cps=True,
        verbose=0,
    )

    vertices = [fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]]
    # compute neighbors matrix:
    from utils import get_neighbors

    neighbors = get_neighbors([fwd["src"][0]["use_tris"], fwd["src"][1]["use_tris"]], vertices)
    if spos is None:
        src = loadmat(os.path.join(model_path, f"sources_{suf}.mat"))
        spos = src["positions"]
#######################################################################################
################## SIMULATION LOOP #########################################
match_dict ={}
tic = time.time()
for e in range(1,args.n_examples+1) :
    n_src = 0
    # to save data
    id      = e
    seeds   = [] 
    orders  = [] 
    patches = {}
    
    c_tot   = [] # total components 
    # For each sample
    # - randomlky choose number of patch
    # - choose amplitude-width... base value
    # for each patch 
    #   - randomly choose order of the patch
    #   - randomly choose seed of the patch **among available seeds**
    #   - randomly choose signal parameters for the patch 
    #   - -> get_component
    #   - remove activated sources from available sources
    #   - save info of the patch.
    
    # spatial
    
    n_patch     = np.random.randint(args.n_patch_min, args.n_patch_max,1).item() 
    # temporal
    ## base_amplitude  = s_range_ampl[0] + (s_range_ampl[1]-s_range_ampl[0])*np.random.rand(1)
    ## base_width      = s_range_width[0] + (s_range_width[1]-s_range_width[0])*np.random.rand(1)
    ## base_center     = s_range_center[0] + (s_range_center[1]-s_range_center[0])*np.random.rand(1) 

    base_amplitude = np.random.uniform( low=s_range_ampl[0], high=s_range_ampl[1], size=1 )
    base_width = np.random.uniform( low=s_range_width[0], high=s_range_width[1], size=1 )
    base_center = np.random.uniform( low=s_range_center[0], high=s_range_center[1], size=1 )


    p_range_ampl = np.array( [ base_amplitude - p_amplitude_dev*base_amplitude,
                              base_amplitude + p_amplitude_dev*base_amplitude ])
    p_range_width = np.array( [ base_width - p_width_dev*base_width,
                               base_width + p_width_dev*base_width ])
    p_range_center = np.array([ base_center - p_center_dev*base_center,
                               base_center + p_center_dev*base_center ])
    
    erp_per_patch = []
    to_remove = []
    available_sources = np.arange(0,n_sources,1)
    
    for p in range(1,n_patch+1):
        #spatial
        order               = np.random.randint(args.order_min, args.order_max)
        available_sources   = np.delete(available_sources, to_remove)
        seed                = int(np.random.choice(available_sources))


        #print(f"seed : {seed}, order : {order}")

        # temporal
        ## erp_params          = {
        ##     'ampl':   p_range_ampl[0] + (p_range_ampl[1]-p_range_ampl[0])*np.random.randint(1), 
        ##     'width':  np.ceil( p_range_width[0] + (p_range_width[1]-p_range_width[0])*np.random.randint(1) ), 
        ##     'center': np.ceil( p_range_center[0] + (p_range_center[1]-p_range_center[0])*np.random.randint(1))
        ## }

        erp_params          = {
            'ampl':   np.random.uniform(p_range_ampl[0], p_range_ampl[1], 1), 
            'width':  np.ceil( np.random.uniform(p_range_width[0] ,p_range_width[1], 1 ) ), 
            'center': np.ceil( np.random.uniform(p_range_center[0] ,p_range_center[1], 1) )
        }
        
        # get the components of the patch
        [c,patch, patch_dim] = utils.get_component_extended_src(order, seed, neighbors, spos,
                                erp_params, erp_dev_intra_patch, timeline )
        
        margin_sources = utils.get_patch( order+args.margin, seed, neighbors )
        to_remove = np.hstack([to_remove, np.squeeze(margin_sources).astype(int)]).astype(int)
        available_sources = np.arange(0,n_sources,1)
        
        c_tot += c

        
        if len(patch)>1 : 
            int_patch = []
            for pp in patch : 
                int_patch.append(int(pp))
            patches[f'patch_{p}'] = int_patch #list(np.squeeze(patch).astype(int)) 
        else : 
            patches[f'patch_{p}'] = [int(patch.item())]
        
        #orders = np.hstack([orders, int(np.squeeze(order))]) #orders.append(order)
        orders.append(int(order))
        seeds.append(seed) #seeds = np.hstack([seeds, int(seed)])
        erp_per_patch.append({k: float(np.squeeze(v)) for k, v in erp_params.items()})
        n_src += len(patches[f'patch_{p}'])    
        
    [X,source_data] = utils.generate_scalp_data(c_tot, leadfield, timeline)
    
    if not args.dont_save :
        
        act_src_file    = f"{id}_src_act.mat"
        noise_src_file  = f"{id}_src_noise.mat"
        md_json_file    = f"{id}_md_json_flie.json" 
        eeg_infdb_file  = f"{id}_eeg.mat"
        
        #Jact   = {'Jact': source_data } 
        #Jnoise = {'Jnoise': [] } 
        #eeg_data    = { 'EEG' : X }
 
        Jact   = {'Jact': {'Jact':source_data} } 
        Jnoise = {'Jnoise': {'Jnoise':[]} } 
        eeg_data    = { 'eeg_data' : {'EEG':X} }
        
        md_dict = {
            'id': id,
            'seeds': list(seeds),
            'orders': list(orders), 
            'n_patch': n_patch,
            'act_src': patches,
            # ERP timing (ms from window start): sample-level base values + each patch's
            # actual parameters (patch k <-> seeds[k]); a patch peaks at its 'center'.
            'base_center': float(np.squeeze(base_center)),
            'base_width': float(np.squeeze(base_width)),
            'base_amplitude': float(np.squeeze(base_amplitude)),
            'erp': erp_per_patch}
        
        match_dict[f'id_{id}'] ={
            'act_src_file_name': os.path.join(saving_folder, "sources", "Jact", act_src_file),
            'noise_src_file_name': os.path.join(saving_folder, "sources", "Jnoise", noise_src_file),
            'eeg_file_name': os.path.join(saving_folder, "eeg", "infdb", eeg_infdb_file), 
            'md_json_file_name': os.path.join(saving_folder, "md", md_json_file) 
        }

        act_src_file    = os.path.join( saving_folder, "sources", "Jact", act_src_file)
        noise_src_file  = os.path.join(saving_folder, "sources", "Jnoise", noise_src_file)
        
        savemat(act_src_file, Jact)
        savemat(noise_src_file, Jnoise)
        
        eeg_infdb_file = os.path.join(saving_folder, "eeg", "infdb", eeg_infdb_file)
        savemat(eeg_infdb_file, eeg_data); 

        md_json_file = os.path.join(saving_folder, "md", md_json_file) 
        with open(md_json_file, 'w') as f : 
            json.dump(md_dict,f)


simu_time = time.time() -tic
################################################


ids = np.arange(1,args.n_examples)
if not args.dont_save :
    match_json_file = '_match_json_file.json' 
    match_json_file = os.path.join( saving_folder, f"{args.simu_name}{args.source_sampling}{match_json_file}" )
    with open(match_json_file, 'w') as f : 
        json.dump(match_dict, f)
    
    electrode_space = {
            'n_electrodes': n_electrodes, 
            'electrode_montage': args.montage_kind} 
    source_space = {
        'n_sources': n_sources,
        'constrained_orientation': constrained_orientation,
        'src_sampling': args.source_sampling} 
    rec_info = {
        'fs': args.fs,
        'n_trials': args.n_trials,
        'n_times': int(n_times),
        'trial_ms_duree': args.duree} 

    int_ids = []
    for i in ids :
        int_ids.append(int(i)) 
    general_dict = {
        'electrode_space': electrode_space, 
        'source_space': source_space,
        'rec_info': rec_info,
        'ids': list(int_ids),
        'seed': args.seed,
        'generator_args': vars(args)}

    general_config_file = os.path.join( saving_folder, f"{args.simu_name}{args.source_sampling}_config.json" )
    with open(general_config_file, 'w') as f: 
        json.dump(general_dict,f)

print("___________________DONE___________________")
print(f"simulation took (s): {simu_time:.4}")

        



