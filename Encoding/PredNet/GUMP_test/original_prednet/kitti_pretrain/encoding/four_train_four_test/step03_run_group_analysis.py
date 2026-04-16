"""
Step 3: Group-Level Analysis for Correlation and Beta Maps - Gump Dataset

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

FEATURE_TYPE = "E"  # "E" for Error signals, "Ahat" for Predictions

BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4"
FMRI_DATA_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new/"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Layer configuration based on feature type
if FEATURE_TYPE == "Ahat":
    layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"]
else:  # "E"
    layer_names = ["/E0", "/E1", "/E2", "/E3"]

num_layers = len(layer_names)

# Gump subject IDs (actual naming from your preprocessing script)
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
    print(f"    Stacked: {stacked_corrs.shape} (subjects x layers x voxels)")
    
    group_mean_corr = np.mean(stacked_corrs, axis=0)
    group_std_corr = np.std(stacked_corrs, axis=0)
    
    # Print summary
    print(f"\n    Group Mean Correlation Summary:")
    for i in range(group_mean_corr.shape[0]):
        mean_r = np.mean(group_mean_corr[i])
        max_r = np.max(group_mean_corr[i])
        print(f"      Layer {i+1} ({FEATURE_TYPE}{i}): Mean R = {mean_r:.4f}, Max R = {max_r:.4f}")
    
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
    print(f"    Stacked: {stacked_betas.shape} (subjects x layers x voxels)")
    
    group_mean_beta = np.mean(stacked_betas, axis=0)
    group_std_beta = np.std(stacked_betas, axis=0)
    
    # Print summary
    print(f"\n    Group Mean Beta Summary:")
    for i in range(group_mean_beta.shape[0]):
        mean_b = np.mean(group_mean_beta[i])
        min_b = np.min(group_mean_beta[i])
        max_b = np.max(group_mean_beta[i])
        print(f"      Layer {i+1} ({FEATURE_TYPE}{i}): Mean = {mean_b:.4f}, Range = [{min_b:.4f}, {max_b:.4f}]")
    
    return group_mean_beta, group_std_beta, stacked_betas


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_step5_group_analysis():
    """
    Main function for group-level analysis.
    """
    print("\n" + "="*70)
    print(f" Step 5: Group-Level Analysis - Gump ({FEATURE_TYPE})")
    print("="*70)
    
    SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE
    
    if not SAVE_ROOT.exists():
        print(f"  [ERROR] Directory not found: {SAVE_ROOT}")
        return
    
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
        
        # Save .mat
        sio.savemat(
            str(group_dir / "group_correlations.mat"),
            {
                "group_mean_corr": group_mean_corr,
                "group_std_corr": group_std_corr,
                "all_subject_corrs": all_subject_corrs
            },
            do_compression=True
        )
        print(f"    Saved: group_correlations.mat")
        
        # Save CIFTI
        map_names = [f"{FEATURE_TYPE}{i}_corr" for i in range(num_layers)]
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
        
        # Save .mat
        sio.savemat(
            str(group_dir / "group_betas.mat"),
            {
                "group_mean_beta": group_mean_beta,
                "group_std_beta": group_std_beta,
                "all_subject_betas": all_subject_betas
            },
            do_compression=True
        )
        print(f"    Saved: group_betas.mat")
        
        # Save CIFTI
        map_names = [f"{FEATURE_TYPE}{i}_beta" for i in range(num_layers)]
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
# LAYER-WISE RANGE MATCHING (Prediction vs Error)
# =============================================================================

def save_cifti_with_layerwise_palette(data, template_file, output_file, map_names, layer_pos_max, layer_neg_max):
    """
    Save CIFTI dscalar with per-layer palette ranges using wb_command.
    
    Args:
        data: (n_maps, n_voxels) array
        template_file: CIFTI template path
        output_file: output path
        map_names: list of map names
        layer_pos_max: list of positive max values for each layer
        layer_neg_max: list of negative max (most extreme negative) values for each layer
    """
    import subprocess
    
    # First save the basic CIFTI file
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
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    
    # Apply palette settings to each column using wb_command
    for i in range(len(layer_pos_max)):
        pos_max = layer_pos_max[i]
        neg_max = layer_neg_max[i]  # most extreme negative value
        
        # -neg-user <neg-min> <neg-max>: neg-min is closer to 0, neg-max is most negative
        cmd = [
            'wb_command', '-cifti-palette',
            output_file,
            'MODE_USER_SCALE',
            output_file,
            '-column', str(i + 1),
            '-pos-user', '0', str(pos_max),
            '-neg-user', '0', str(neg_max),  # Fixed: 0 first, then most negative value
            '-palette-name', 'ROY-BIG-BL'
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"      Layer {i+1}: palette set to [{neg_max:.6f}, {pos_max:.6f}]")
        except subprocess.CalledProcessError as e:
            print(f"    WARNING: wb_command failed for column {i+1}: {e.stderr.decode()}")
        except FileNotFoundError:
            print(f"    WARNING: wb_command not found, palette not set")
            break
    
    return True


def compute_layerwise_range_matched():
    """
    Load both Prediction and Error beta/correlation maps, compute layer-wise max range,
    and save range-matched CIFTI files for proper comparison.
    """
    print("\n" + "="*70)
    print(" Layer-wise Range Matching: Prediction vs Error")
    print("="*70)
    
    # Paths
    ahat_group_dir = Path(BASE_SAVE_DIR) / "Ahat" / "group"
    e_group_dir = Path(BASE_SAVE_DIR) / "E" / "group"
    combined_dir = Path(BASE_SAVE_DIR) / "combined_analysis"
    combined_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # BETA MAPS
    # =========================================================================
    print("\n" + "-"*50)
    print(" Processing BETA Maps")
    print("-"*50)
    
    # Load Prediction (Ahat) beta
    ahat_beta_path = ahat_group_dir / "group_mean_betas.npy"
    if not ahat_beta_path.exists():
        print(f"  [ERROR] Ahat beta not found: {ahat_beta_path}")
        ahat_beta = None
    else:
        ahat_beta = np.load(str(ahat_beta_path))
        print(f"  Loaded Ahat beta: {ahat_beta.shape}")
    
    # Load Error (E) beta
    e_beta_path = e_group_dir / "group_mean_betas.npy"
    if not e_beta_path.exists():
        print(f"  [ERROR] E beta not found: {e_beta_path}")
        e_beta = None
    else:
        e_beta = np.load(str(e_beta_path))
        print(f"  Loaded E beta: {e_beta.shape}")
    
    if ahat_beta is not None and e_beta is not None:
        # Compute layer-wise positive max and negative max (most extreme values)
        n_layers = ahat_beta.shape[0]
        beta_layer_pos_max = []
        beta_layer_neg_max = []
        
        print(f"\n  Layer-wise Beta Range Analysis:")
        print(f"  {'Layer':<8} {'Ahat Min':<12} {'Ahat Max':<12} {'E Min':<12} {'E Max':<12} {'Combined NegMax':<16} {'Combined PosMax':<16}")
        print(f"  {'-'*92}")
        
        for i in range(n_layers):
            ahat_min = ahat_beta[i].min()
            ahat_max = ahat_beta[i].max()
            e_min = e_beta[i].min()
            e_max = e_beta[i].max()
            
            # Combined: take the most extreme values from both
            combined_neg_max = min(ahat_min, e_min)  # most negative (extreme)
            combined_pos_max = max(ahat_max, e_max)  # most positive (extreme)
            
            beta_layer_neg_max.append(combined_neg_max)
            beta_layer_pos_max.append(combined_pos_max)
            
            print(f"  Layer {i+1:<3} {ahat_min:<12.6f} {ahat_max:<12.6f} {e_min:<12.6f} {e_max:<12.6f} {combined_neg_max:<16.6f} {combined_pos_max:<16.6f}")
        
        # Save Ahat beta with layer-wise palette
        print(f"\n  Saving Ahat beta with layer-wise palette...")
        ahat_map_names = [f"Ahat{i}" for i in range(n_layers)]
        save_cifti_with_layerwise_palette(
            ahat_beta, CIFTI_TEMPLATE,
            str(combined_dir / "Ahat_group_mean_betas_layerwise_range.dscalar.nii"),
            ahat_map_names, beta_layer_pos_max, beta_layer_neg_max
        )
        
        # Save E beta with layer-wise palette
        print(f"\n  Saving E beta with layer-wise palette...")
        e_map_names = [f"E{i}" for i in range(n_layers)]
        save_cifti_with_layerwise_palette(
            e_beta, CIFTI_TEMPLATE,
            str(combined_dir / "E_group_mean_betas_layerwise_range.dscalar.nii"),
            e_map_names, beta_layer_pos_max, beta_layer_neg_max
        )
    else:
        beta_layer_pos_max = None
        beta_layer_neg_max = None
    
    # =========================================================================
    # CORRELATION MAPS
    # =========================================================================
    print("\n" + "-"*50)
    print(" Processing CORRELATION Maps")
    print("-"*50)
    
    # Load Prediction (Ahat) correlation
    ahat_corr_path = ahat_group_dir / "group_mean_correlations.npy"
    if not ahat_corr_path.exists():
        print(f"  [ERROR] Ahat correlation not found: {ahat_corr_path}")
        ahat_corr = None
    else:
        ahat_corr = np.load(str(ahat_corr_path))
        print(f"  Loaded Ahat correlation: {ahat_corr.shape}")
    
    # Load Error (E) correlation
    e_corr_path = e_group_dir / "group_mean_correlations.npy"
    if not e_corr_path.exists():
        print(f"  [ERROR] E correlation not found: {e_corr_path}")
        e_corr = None
    else:
        e_corr = np.load(str(e_corr_path))
        print(f"  Loaded E correlation: {e_corr.shape}")
    
    if ahat_corr is not None and e_corr is not None:
        # Compute layer-wise positive max and negative max
        n_layers = ahat_corr.shape[0]
        corr_layer_pos_max = []
        corr_layer_neg_max = []
        
        print(f"\n  Layer-wise Correlation Range Analysis:")
        print(f"  {'Layer':<8} {'Ahat Min':<12} {'Ahat Max':<12} {'E Min':<12} {'E Max':<12} {'Combined NegMax':<16} {'Combined PosMax':<16}")
        print(f"  {'-'*92}")
        
        for i in range(n_layers):
            ahat_min = ahat_corr[i].min()
            ahat_max = ahat_corr[i].max()
            e_min = e_corr[i].min()
            e_max = e_corr[i].max()
            
            # Combined: take the most extreme values from both
            combined_neg_max = min(ahat_min, e_min)  # most negative (extreme)
            combined_pos_max = max(ahat_max, e_max)  # most positive (extreme)
            
            corr_layer_neg_max.append(combined_neg_max)
            corr_layer_pos_max.append(combined_pos_max)
            
            print(f"  Layer {i+1:<3} {ahat_min:<12.6f} {ahat_max:<12.6f} {e_min:<12.6f} {e_max:<12.6f} {combined_neg_max:<16.6f} {combined_pos_max:<16.6f}")
        
        # Save Ahat correlation with layer-wise palette
        print(f"\n  Saving Ahat correlation with layer-wise palette...")
        ahat_corr_map_names = [f"Ahat{i}_corr" for i in range(n_layers)]
        save_cifti_with_layerwise_palette(
            ahat_corr, CIFTI_TEMPLATE,
            str(combined_dir / "Ahat_group_mean_correlations_layerwise_range.dscalar.nii"),
            ahat_corr_map_names, corr_layer_pos_max, corr_layer_neg_max
        )
        
        # Save E correlation with layer-wise palette
        print(f"\n  Saving E correlation with layer-wise palette...")
        e_corr_map_names = [f"E{i}_corr" for i in range(n_layers)]
        save_cifti_with_layerwise_palette(
            e_corr, CIFTI_TEMPLATE,
            str(combined_dir / "E_group_mean_correlations_layerwise_range.dscalar.nii"),
            e_corr_map_names, corr_layer_pos_max, corr_layer_neg_max
        )
    else:
        corr_layer_pos_max = None
        corr_layer_neg_max = None
    
    # =========================================================================
    # SAVE RANGE INFORMATION
    # =========================================================================
    print("\n" + "-"*50)
    print(" Saving Range Information")
    print("-"*50)
    
    # Prepare data for saving
    range_data = {}
    mat_data = {}
    
    if beta_layer_pos_max is not None:
        range_data['beta_pos_max'] = np.array(beta_layer_pos_max)
        range_data['beta_neg_max'] = np.array(beta_layer_neg_max)
        mat_data['beta_layer_pos_max'] = np.array(beta_layer_pos_max)
        mat_data['beta_layer_neg_max'] = np.array(beta_layer_neg_max)
        mat_data['ahat_beta_layer_max'] = np.array([ahat_beta[i].max() for i in range(n_layers)])
        mat_data['ahat_beta_layer_min'] = np.array([ahat_beta[i].min() for i in range(n_layers)])
        mat_data['e_beta_layer_max'] = np.array([e_beta[i].max() for i in range(n_layers)])
        mat_data['e_beta_layer_min'] = np.array([e_beta[i].min() for i in range(n_layers)])
    
    if corr_layer_pos_max is not None:
        range_data['corr_pos_max'] = np.array(corr_layer_pos_max)
        range_data['corr_neg_max'] = np.array(corr_layer_neg_max)
        mat_data['corr_layer_pos_max'] = np.array(corr_layer_pos_max)
        mat_data['corr_layer_neg_max'] = np.array(corr_layer_neg_max)
        mat_data['ahat_corr_layer_max'] = np.array([ahat_corr[i].max() for i in range(n_layers)])
        mat_data['ahat_corr_layer_min'] = np.array([ahat_corr[i].min() for i in range(n_layers)])
        mat_data['e_corr_layer_max'] = np.array([e_corr[i].max() for i in range(n_layers)])
        mat_data['e_corr_layer_min'] = np.array([e_corr[i].min() for i in range(n_layers)])
    
    # Save npz
    np.savez(str(combined_dir / "layer_ranges.npz"), **range_data)
    print(f"  Saved: layer_ranges.npz")
    
    # Save mat
    sio.savemat(str(combined_dir / "layer_range_info.mat"), mat_data)
    print(f"  Saved: layer_range_info.mat")
    
    # Save summary text file
    summary_path = combined_dir / "visualization_ranges.txt"
    with open(summary_path, 'w') as f:
        f.write("Layer-wise Range Matching for Prediction vs Error Comparison\n")
        f.write("="*60 + "\n\n")
        f.write("Each layer has its own palette range set automatically.\n")
        f.write("Open the dscalar files in wb_view to see matched ranges.\n\n")
        
        if beta_layer_pos_max is not None:
            f.write("BETA RANGES:\n")
            f.write("-" * 40 + "\n")
            for i in range(len(beta_layer_pos_max)):
                f.write(f"  Layer {i+1}: [{beta_layer_neg_max[i]:.6f}, {beta_layer_pos_max[i]:.6f}]\n")
            f.write("\n")
        
        if corr_layer_pos_max is not None:
            f.write("CORRELATION RANGES:\n")
            f.write("-" * 40 + "\n")
            for i in range(len(corr_layer_pos_max)):
                f.write(f"  Layer {i+1}: [{corr_layer_neg_max[i]:.6f}, {corr_layer_pos_max[i]:.6f}]\n")
    
    print(f"  Saved: visualization_ranges.txt")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n" + "="*70)
    print(f"  Output files in: {combined_dir}")
    print(f"="*70)
    print(f"\n  BETA files:")
    print(f"    - Ahat_group_mean_betas_layerwise_range.dscalar.nii")
    print(f"    - E_group_mean_betas_layerwise_range.dscalar.nii")
    print(f"\n  CORRELATION files:")
    print(f"    - Ahat_group_mean_correlations_layerwise_range.dscalar.nii")
    print(f"    - E_group_mean_correlations_layerwise_range.dscalar.nii")
    print(f"\n  Range info:")
    print(f"    - layer_ranges.npz")
    print(f"    - layer_range_info.mat")
    print(f"    - visualization_ranges.txt")
    print("="*70)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gump Step 5: Group Analysis")
    parser.add_argument("--combined", action="store_true", 
                        help="Run layer-wise range matching for Ahat vs E comparison")
    args = parser.parse_args()
    
    if args.combined:
        compute_layerwise_range_matched()
    else:
        run_step5_group_analysis()