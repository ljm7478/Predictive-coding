"""
Step 4: Compute Beta Maps from W matrices using PCA (PC1 loadings)

This script:
1. Loads W matrices for each subject and layer
2. Applies PCA to W matrix
3. Extracts PC1 loadings as "beta" for each voxel
4. Saves beta maps for each subject

Beta map interpretation:
- W shape: (n_features, n_voxels) = (500, 59412)
- Each voxel has 500 weights (one per PCA component)
- PC1 of W captures the dominant pattern across features
- PC1 loading for each voxel = how strongly that voxel follows the main encoding pattern
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

# Base directory containing subject folders
BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/longer_time/four_train_four_test/layer4"

CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

TEMPORAL_TYPES = ["indirect_1s_extrap", "indirect_2s_extrap", "indirect_5s_extrap", "indirect_10s_extrap"]

FEATURE_TYPE = "E"

if FEATURE_TYPE == "Ahat":
    layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"]
else:  # "E"
    layer_names = ["/E0", "/E1", "/E2", "/E3"]
    
num_layers = len(layer_names)

subjects = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15", 
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_W_matrix(filepath):
    """
    Load W matrix from HDF5 or MAT file.
    
    Returns:
        W: (n_features, n_voxels) array
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.h5':
        with h5py.File(filepath, 'r') as f:
            W = f['W'][:].astype(np.float32)
    elif filepath.suffix == '.mat':
        try:
            # Try loading as v7.3 (HDF5)
            with h5py.File(filepath, 'r') as f:
                W = f['W'][:].astype(np.float32)
        except:
            # Fall back to scipy for older .mat format
            data = sio.loadmat(filepath)
            W = data['W'].astype(np.float32)
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")
    
    return W


def compute_beta_pc1(W):
    """
    Compute beta map using PCA PC1 loadings.
    
    Args:
        W: (n_features, n_voxels) weight matrix
    
    Returns:
        beta_pc1: (n_voxels,) PC1 loading for each voxel
        explained_var: variance explained by PC1
    
    Method:
        1. Transpose W to (n_voxels, n_features)
        2. Apply PCA
        3. PC1 scores = beta for each voxel
    """
    n_features, n_voxels = W.shape
    
    # Transpose: each row = one voxel, each column = one feature weight
    W_T = W.T  # (n_voxels, n_features)
    
    # Center the data (mean across voxels for each feature)
    W_centered = W_T - W_T.mean(axis=0, keepdims=True)
    
    # Apply PCA
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(W_centered)  # (n_voxels, 1)
    
    beta_pc1 = pc1_scores.flatten()  # (n_voxels,)
    explained_var = pca.explained_variance_ratio_[0]
    
    return beta_pc1, explained_var


def compute_beta_mean(W):
    """
    Alternative: Compute beta as mean weight across features.
    
    Args:
        W: (n_features, n_voxels) weight matrix
    
    Returns:
        beta_mean: (n_voxels,) mean weight for each voxel
    """
    return W.mean(axis=0)


def compute_beta_pc1_weight(W):
    """
    Alternative: Use the weight for PC1 of original features.
    
    This takes the first row of W (weight for the first PCA component).
    
    Args:
        W: (n_features, n_voxels) weight matrix
    
    Returns:
        beta: (n_voxels,) weight for PC1
    """
    return W[0, :]


def save_cifti_dscalar(data, template_file, output_file, map_names=None):
    """
    Save as CIFTI dscalar with cortex only.
    
    Args:
        data: (n_maps, n_grayordinates) or (n_grayordinates,)
        template_file: CIFTI template path
        output_file: output path
        map_names: optional list of map names
    """
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_gray_data = data.shape
    
    # Get brain axis from template
    brain_axis = template.header.get_axis(1)
    
    # Find cortical indices only
    cortex_indices = []
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX' in name:
            cortex_indices.extend(range(slc.start, slc.stop))
    
    n_cortex = len(cortex_indices)
    
    if n_gray_data != n_cortex:
        print(f"    WARNING: Data ({n_gray_data}) != cortex ({n_cortex})")
        return False
    
    # Create cortex-only brain axis
    cortex_brain_axis = brain_axis[np.array(cortex_indices)]
    
    # Create scalar axis
    if map_names is None:
        map_names = [f'Map_{i+1}' for i in range(n_maps)]
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    
    # Create new header and image
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    
    return True


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compute_subject_beta_maps(save_root, subject, layer_names, method='pc1'):
    """
    Compute beta maps for a single subject.
    
    Args:
        save_root: directory containing subject folder
        subject: subject ID
        layer_names: list of layer names
        method: 'pc1', 'mean', or 'pc1_weight'
    
    Returns:
        beta_all_layers: (n_layers, n_voxels) array
    """
    subject_dir = Path(save_root) / subject
    n_layers = len(layer_names)
    
    beta_all_layers = []
    explained_vars = []
    
    for lay in range(n_layers):
        layer_name = layer_names[lay]
        
        # Try different W file formats
        w_path_h5 = subject_dir / f"W_layer{lay + 1}.h5"
        w_path_mat = subject_dir / f"W_layer{lay + 1}.mat"
        
        if w_path_h5.exists():
            W = load_W_matrix(w_path_h5)
        elif w_path_mat.exists():
            W = load_W_matrix(w_path_mat)
        else:
            print(f"      [ERROR] W matrix not found for layer {lay + 1}")
            return None
        
        print(f"      Layer {lay + 1}: W shape = {W.shape}", end=" ")
        
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
    
    beta_all_layers = np.stack(beta_all_layers)  # (n_layers, n_voxels)
    
    if method == 'pc1' and explained_vars:
        print(f"      Mean PC1 variance explained: {np.mean(explained_vars):.2%}")
    
    return beta_all_layers


def run_step4_beta_maps(method='pc1'):
    """
    Main function to compute beta maps for all subjects.
    
    Args:
        method: 'pc1' (PCA), 'mean' (average), or 'pc1_weight' (first component weight)
    """
    print("\n" + "="*70)
    print(f" Step 4: Computing Beta Maps (method={method})")
    print("="*70)
    
    for current_temp_type in TEMPORAL_TYPES[0:1]:
        print(f"\n{'='*70}")
        print(f" PROCESSING: {current_temp_type} ({FEATURE_TYPE})")
        print(f"{'='*70}")
        
        SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE / current_temp_type
        
        if not SAVE_ROOT.exists():
            print(f"  [ERROR] Directory not found: {SAVE_ROOT}")
            continue
        
        for subj_idx, subject in enumerate(subjects):
            print(f"\n  Subject: {subject} ({subj_idx + 1}/{len(subjects)})")
            
            subject_dir = SAVE_ROOT / subject
            
            # Check if W matrices exist
            if not subject_dir.exists():
                print(f"    [WARNING] Subject directory not found: {subject_dir}")
                continue
            
            # Skip if beta already computed
            beta_output = subject_dir / "beta_pc1_all_layers.npy"
            if beta_output.exists():
                print(f"    Beta maps already exist. Skipping.")
                continue
            
            # Compute beta maps
            beta_all_layers = compute_subject_beta_maps(
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
                str(subject_dir / "beta_pc1_all_layers.mat"),
                {"beta_all_layers": beta_all_layers},
                do_compression=True
            )
            print(f"    Saved: beta_pc1_all_layers.mat")
            
            # Save as CIFTI
            map_names = [f"Layer_{i+1}_beta" for i in range(num_layers)]
            cifti_output = str(subject_dir / "beta_pc1_all_layers.dscalar.nii")
            success = save_cifti_dscalar(
                beta_all_layers, CIFTI_TEMPLATE, cifti_output, map_names
            )
            if success:
                print(f"    Saved: beta_pc1_all_layers.dscalar.nii")
            
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


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_step4_beta_maps(method='pc1')