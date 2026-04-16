"""
Step 5: Group-Level Analysis for Correlation and Beta Maps

This script:
1. Loads individual subject correlation maps
2. Loads individual subject beta maps
3. Computes group mean and std
4. Saves group-level maps as CIFTI
"""

import os
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import nibabel as nib


# =============================================================================
# CONFIGURATION
# =============================================================================

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

def save_cifti_dscalar(data, template_file, output_file, map_names=None):
    """
    Save as CIFTI dscalar with cortex only.
    """
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


def load_subject_data(subject_dir, filename):
    """
    Load subject data from .npy or .mat file.
    
    Returns:
        data: numpy array or None if not found
    """
    npy_path = subject_dir / filename.replace('.mat', '.npy')
    mat_path = subject_dir / filename.replace('.npy', '.mat')
    
    if npy_path.exists():
        return np.load(str(npy_path))
    elif mat_path.exists():
        try:
            with h5py.File(mat_path, 'r') as f:
                # Get first key that's not metadata
                for key in f.keys():
                    if not key.startswith('#'):
                        return f[key][:].astype(np.float32)
        except:
            data = sio.loadmat(str(mat_path))
            for key in data.keys():
                if not key.startswith('_'):
                    return data[key].astype(np.float32)
    
    return None


# =============================================================================
# GROUP CORRELATION ANALYSIS
# =============================================================================

def compute_group_correlations(save_root, subjects, layer_names):
    """
    Compute group-level correlation maps.
    
    Returns:
        group_mean_corr: (n_layers, n_voxels)
        group_std_corr: (n_layers, n_voxels)
        all_subject_corrs: (n_subjects, n_layers, n_voxels)
    """
    print("\n  [Group Correlations]")
    
    all_subject_corrs = []
    valid_subjects = []
    
    for subject in subjects:
        subject_dir = Path(save_root) / subject
        
        # Try loading correlation data
        corr_data = load_subject_data(subject_dir, 'average_correlations.npy')
        
        if corr_data is not None:
            # Handle different shapes
            if corr_data.ndim == 1:
                corr_data = corr_data.reshape(1, -1)
            
            all_subject_corrs.append(corr_data)
            valid_subjects.append(subject)
            print(f"    Loaded {subject}: {corr_data.shape}")
        else:
            print(f"    WARNING: Missing correlation data for {subject}")
    
    if len(all_subject_corrs) == 0:
        print("    ERROR: No correlation data found!")
        return None, None, None
    
    # Stack and compute stats
    stacked_corrs = np.stack(all_subject_corrs, axis=0)
    print(f"    Stacked: {stacked_corrs.shape} (subjects × layers × voxels)")
    
    group_mean_corr = np.mean(stacked_corrs, axis=0)
    group_std_corr = np.std(stacked_corrs, axis=0)
    
    # Print summary
    print(f"\n    Group Mean Correlation Summary:")
    for i in range(group_mean_corr.shape[0]):
        mean_r = np.mean(group_mean_corr[i])
        max_r = np.max(group_mean_corr[i])
        print(f"      Layer {i+1}: Mean R = {mean_r:.4f}, Max R = {max_r:.4f}")
    
    return group_mean_corr, group_std_corr, stacked_corrs


# =============================================================================
# GROUP BETA ANALYSIS
# =============================================================================

def compute_group_betas(save_root, subjects, layer_names):
    """
    Compute group-level beta maps.
    
    Returns:
        group_mean_beta: (n_layers, n_voxels)
        group_std_beta: (n_layers, n_voxels)
        all_subject_betas: (n_subjects, n_layers, n_voxels)
    """
    print("\n  [Group Betas]")
    
    all_subject_betas = []
    valid_subjects = []
    
    for subject in subjects:
        subject_dir = Path(save_root) / subject
        
        # Try loading beta data
        beta_data = load_subject_data(subject_dir, 'beta_pc1_all_layers.npy')
        
        if beta_data is not None:
            if beta_data.ndim == 1:
                beta_data = beta_data.reshape(1, -1)
            
            all_subject_betas.append(beta_data)
            valid_subjects.append(subject)
            print(f"    Loaded {subject}: {beta_data.shape}")
        else:
            print(f"    WARNING: Missing beta data for {subject}")
    
    if len(all_subject_betas) == 0:
        print("    ERROR: No beta data found!")
        return None, None, None
    
    # Stack and compute stats
    stacked_betas = np.stack(all_subject_betas, axis=0)
    print(f"    Stacked: {stacked_betas.shape} (subjects × layers × voxels)")
    
    group_mean_beta = np.mean(stacked_betas, axis=0)
    group_std_beta = np.std(stacked_betas, axis=0)
    
    # Print summary
    print(f"\n    Group Mean Beta Summary:")
    for i in range(group_mean_beta.shape[0]):
        mean_b = np.mean(group_mean_beta[i])
        min_b = np.min(group_mean_beta[i])
        max_b = np.max(group_mean_beta[i])
        print(f"      Layer {i+1}: Mean = {mean_b:.4f}, Range = [{min_b:.4f}, {max_b:.4f}]")
    
    return group_mean_beta, group_std_beta, stacked_betas


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_step5_group_analysis():
    """
    Main function for group-level analysis.
    """
    print("\n" + "="*70)
    print(" Step 5: Group-Level Analysis")
    print("="*70)
    
    for current_temp_type in TEMPORAL_TYPES[0:1]:
        print(f"\n{'='*70}")
        print(f" PROCESSING: {current_temp_type} ({FEATURE_TYPE})")
        print(f"{'='*70}")
        
        SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE / current_temp_type
        
        if not SAVE_ROOT.exists():
            print(f"  [ERROR] Directory not found: {SAVE_ROOT}")
            continue
        
        # Create group directory
        group_dir = SAVE_ROOT / "group"
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # Group Correlation Maps
        # =====================================================================
        group_mean_corr, group_std_corr, all_subject_corrs = compute_group_correlations(
            SAVE_ROOT, subjects, layer_names
        )
        
        if group_mean_corr is not None:
            # Save numpy
            np.save(str(group_dir / "group_mean_correlations.npy"), group_mean_corr)
            np.save(str(group_dir / "group_std_correlations.npy"), group_std_corr)
            np.save(str(group_dir / "all_subject_correlations.npy"), all_subject_corrs)
            print(f"\n    Saved: group correlation numpy files")
            
            # Save CIFTI
            map_names = [f"Layer_{i+1}_corr" for i in range(num_layers)]
            save_cifti_dscalar(
                group_mean_corr, CIFTI_TEMPLATE,
                str(group_dir / "group_mean_correlations.dscalar.nii"),
                map_names
            )
            save_cifti_dscalar(
                group_std_corr, CIFTI_TEMPLATE,
                str(group_dir / "group_std_correlations.dscalar.nii"),
                map_names
            )
            print(f"    Saved: group correlation CIFTI files")
        
        # =====================================================================
        # Group Beta Maps
        # =====================================================================
        group_mean_beta, group_std_beta, all_subject_betas = compute_group_betas(
            SAVE_ROOT, subjects, layer_names
        )
        
        if group_mean_beta is not None:
            # Save numpy
            np.save(str(group_dir / "group_mean_betas.npy"), group_mean_beta)
            np.save(str(group_dir / "group_std_betas.npy"), group_std_beta)
            np.save(str(group_dir / "all_subject_betas.npy"), all_subject_betas)
            print(f"\n    Saved: group beta numpy files")
            
            # Save CIFTI
            map_names = [f"Layer_{i+1}_beta" for i in range(num_layers)]
            save_cifti_dscalar(
                group_mean_beta, CIFTI_TEMPLATE,
                str(group_dir / "group_mean_betas.dscalar.nii"),
                map_names
            )
            save_cifti_dscalar(
                group_std_beta, CIFTI_TEMPLATE,
                str(group_dir / "group_std_betas.dscalar.nii"),
                map_names
            )
            print(f"    Saved: group beta CIFTI files")
        
        print(f"\n    Group analysis saved to: {group_dir}")
    
    print(f"\n{'='*70}")
    print("=== Step 5 Complete ===")
    print("="*70)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    run_step5_group_analysis()