"""
Step 2: Compute Beta Maps from W matrices using PCA (PC1 loadings) - Gump Dataset
LORO Version
"""

import os
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import nibabel as nib
from sklearn.decomposition import PCA


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURE_TYPE = "E"  # "E" for Error signals, "Ahat" for Predictions

# Base directory - LORO version
BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/modified_prednet/kitti_pretrain_result/LORO_PCAcap/layer4"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Layer configuration based on feature type
if FEATURE_TYPE == "Ahat":
    layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"] 
else:  # "E"
    layer_names = ["/E0", "/E1", "/E2", "/E3"]

num_layers = len(layer_names)

# Gump subject IDs
subjects = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15", 
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]


# =============================================================================
# HELPER FUNCTIONS (unchanged)
# =============================================================================

def load_W_matrix(filepath):
    """Load W matrix from HDF5 or MAT file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.h5':
        with h5py.File(filepath, 'r') as f:
            W = f['W'][:].astype(np.float32)
    elif filepath.suffix == '.mat':
        try:
            with h5py.File(filepath, 'r') as f:
                W = f['W'][:].astype(np.float32)
        except:
            data = sio.loadmat(filepath)
            W = data['W'].astype(np.float32)
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")
    
    return W


def compute_beta_pc1(W):
    """Compute beta map using PCA PC1 loadings."""
    n_features, n_voxels = W.shape
    W_T = W.T
    W_centered = W_T - W_T.mean(axis=0, keepdims=True)
    
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(W_centered)
    
    beta_pc1 = pc1_scores.flatten()
    explained_var = pca.explained_variance_ratio_[0]
    
    return beta_pc1, explained_var


def compute_beta_mean(W):
    """Alternative: Compute beta as mean weight across features."""
    return W.mean(axis=0)


def compute_beta_pc1_weight(W):
    """Alternative: Use the weight for PC1 of original features."""
    return W[0, :]


def save_cifti_dscalar(data, template_file, output_file, map_names=None):
    """Save as CIFTI dscalar with cortex only."""
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_gray_data = data.shape
    brain_axis = template.header.get_axis(1)
    
    cortex_indices = []
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX' in name:
            cortex_indices.extend(range(slc.start, slc.stop))
    
    n_cortex = len(cortex_indices)
    
    if n_gray_data != n_cortex:
        print(f"    WARNING: Data ({n_gray_data}) != cortex ({n_cortex})")
        return False
    
    cortex_brain_axis = brain_axis[np.array(cortex_indices)]
    
    if map_names is None:
        map_names = [f'Map_{i+1}' for i in range(n_maps)]
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    
    return True


# =============================================================================
# MAIN FUNCTION - LORO VERSION
# =============================================================================

def compute_subject_beta_maps_LORO(save_root, subject, layer_names, method='pc1'):
    """
    Compute beta maps for LORO - average W across folds first, then compute beta.
    
    LORO saves W matrices per fold: W_layer{lay}_fold{fold}.h5
    """
    subject_dir = Path(save_root) / subject
    n_layers = len(layer_names)
    n_folds = 8  # LORO with 8 runs
    
    beta_all_layers = []
    explained_vars = []
    
    for lay in range(n_layers):
        # Check if fold-wise W matrices exist (LORO)
        fold_w_path = subject_dir / f"W_layer{lay + 1}_fold1.h5"
        single_w_path = subject_dir / f"W_layer{lay + 1}.h5"
        
        if fold_w_path.exists():
            # LORO: Average W across folds
            W_folds = []
            for fold in range(1, n_folds + 1):
                w_path = subject_dir / f"W_layer{lay + 1}_fold{fold}.h5"
                if w_path.exists():
                    W_fold = load_W_matrix(w_path)
                    W_folds.append(W_fold)
            
            if len(W_folds) == 0:
                print(f"      [ERROR] No W matrices found for layer {lay + 1}")
                return None
            
            W = np.mean(np.stack(W_folds), axis=0)
            print(f"      Layer {lay + 1}: Averaged {len(W_folds)} folds, W shape = {W.shape}", end=" ")
        
        elif single_w_path.exists():
            # Single W matrix (non-LORO or already averaged)
            W = load_W_matrix(single_w_path)
            print(f"      Layer {lay + 1}: W shape = {W.shape}", end=" ")
        
        else:
            print(f"      [ERROR] W matrix not found for layer {lay + 1}")
            return None
        
        # Compute beta
        if method == 'pc1':
            beta, exp_var = compute_beta_pc1(W)
            explained_vars.append(exp_var)
            print(f"-> beta shape = {beta.shape}, PC1 var = {exp_var:.2%}")
        elif method == 'mean':
            beta = compute_beta_mean(W)
            print(f"-> beta shape = {beta.shape} (mean)")
        elif method == 'pc1_weight':
            beta = compute_beta_pc1_weight(W)
            print(f"-> beta shape = {beta.shape} (PC1 weight)")
        else:
            raise ValueError(f"Unknown method: {method}")
        
        beta_all_layers.append(beta)
    
    beta_all_layers = np.stack(beta_all_layers)
    
    if method == 'pc1' and explained_vars:
        print(f"      Mean PC1 variance explained: {np.mean(explained_vars):.2%}")
    
    return beta_all_layers


def run_step4_beta_maps(method='pc1'):
    """Main function to compute beta maps for all subjects."""
    print("\n" + "="*70)
    print(f" Step 4: Computing Beta Maps - Gump LORO ({FEATURE_TYPE})")
    print(f" Method: {method}")
    print("="*70)
    
    SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE
    
    if not SAVE_ROOT.exists():
        print(f"  [ERROR] Directory not found: {SAVE_ROOT}")
        return
    
    for subj_idx, subject in enumerate(subjects):
        print(f"\n  Subject: {subject} ({subj_idx + 1}/{len(subjects)})")
        
        subject_dir = SAVE_ROOT / subject
        
        if not subject_dir.exists():
            print(f"    [WARNING] Subject directory not found: {subject_dir}")
            continue
        
        # Skip if beta already computed
        beta_output = subject_dir / "beta_pc1_all_layers_LORO.npy"
        if beta_output.exists():
            print(f"    Beta maps already exist. Skipping.")
            continue
        
        # Compute beta maps
        beta_all_layers = compute_subject_beta_maps_LORO(
            SAVE_ROOT, subject, layer_names, method=method
        )
        
        if beta_all_layers is None:
            continue
        
        print(f"    Final beta shape: {beta_all_layers.shape}")
        
        # Save as numpy
        np.save(str(beta_output), beta_all_layers)
        print(f"    Saved: {beta_output.name}")
        
        # Save as .mat
        sio.savemat(
            str(subject_dir / "beta_pc1_all_layers_LORO.mat"),
            {"beta_all_layers": beta_all_layers, "method": "LORO"},
            do_compression=True
        )
        print(f"    Saved: beta_pc1_all_layers_LORO.mat")
        
        # Save as CIFTI
        map_names = [f"{FEATURE_TYPE}{i}_beta" for i in range(num_layers)]
        cifti_output = str(subject_dir / "beta_pc1_all_layers_LORO.dscalar.nii")
        success = save_cifti_dscalar(
            beta_all_layers, CIFTI_TEMPLATE, cifti_output, map_names
        )
        if success:
            print(f"    Saved: beta_pc1_all_layers_LORO.dscalar.nii")
        
        # Print summary
        print(f"\n    Beta Summary:")
        for i in range(num_layers):
            mean_b = np.mean(beta_all_layers[i])
            std_b = np.std(beta_all_layers[i])
            min_b = np.min(beta_all_layers[i])
            max_b = np.max(beta_all_layers[i])
            print(f"      Layer {i+1}: mean={mean_b:.4f}, std={std_b:.4f}, range=[{min_b:.4f}, {max_b:.4f}]")
    
    print(f"\n{'='*70}")
    print("=== Step 4 Complete ===")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gump Step 4: Beta Maps (LORO)")
    parser.add_argument("--method", type=str, default="pc1", 
                        choices=["pc1", "mean", "pc1_weight"],
                        help="Method for computing beta maps")
    args = parser.parse_args()
    
    run_step4_beta_maps(method=args.method)