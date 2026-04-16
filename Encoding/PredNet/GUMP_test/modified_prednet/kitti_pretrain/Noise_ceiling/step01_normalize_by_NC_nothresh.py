"""
Create normalized maps WITHOUT threshold
- All voxels included (no ISC threshold)
- Just normalize by sqrt(ISC)
- FIXED: Dynamic path loading based on DATASET variable
"""

import numpy as np
import os
import nibabel as nib

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose dataset: 'budapest' or 'gump'
DATASET = 'gump'

# Common Root Path 
USER_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained"

if DATASET == 'budapest':
    PROJECT_DIR = f"{USER_ROOT}/Budapest_test"
    # Budapest´
    ENCODING_DIR = f"{PROJECT_DIR}/encoding_analysis/modified_prednet/kitti_pretrain_result/four_train_four_test/layer4"
    
    # Template for CIFTI (Budapest)
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
    
else:  # gump
    PROJECT_DIR = f"{USER_ROOT}/GUMP_test"
    # Gump´Â PCAcap 
    ENCODING_DIR = f"{PROJECT_DIR}/encoding_analysis/modified_prednet/kitti_pretrain_result/LORO_PCAcap/layer4"
    
    # Template for CIFTI (Gump)
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Set Input/Output Directories based on selection
INPUT_DIR_AHAT = f"{ENCODING_DIR}/Ahat/group"
INPUT_DIR_E = f"{ENCODING_DIR}/E/group"
INPUT_DIR_ISC = f"{ENCODING_DIR}/divergence_analysis/step0_noise_ceiling"

OUTPUT_DIR = f"{ENCODING_DIR}/divergence_analysis/step1_normalized/no_threshold_no_range_match"

print(f"Current Dataset: {DATASET}")
print(f"Encoding Dir: {ENCODING_DIR}")
print(f"Output Dir: {OUTPUT_DIR}")

# =============================================================================
# LOAD RAW DATA
# =============================================================================
print("\nLoading raw data...")

# Beta
ahat_beta_path = f"{INPUT_DIR_AHAT}/group_mean_betas.npy"
e_beta_path = f"{INPUT_DIR_E}/group_mean_betas.npy"

# ahat_beta_path = f"{INPUT_DIR_AHAT}/group_mean_betas_LORO.npy"
# e_beta_path = f"{INPUT_DIR_E}/group_mean_betas_LORO.npy"

if not os.path.exists(ahat_beta_path):
    raise FileNotFoundError(f"File not found: {ahat_beta_path}")

ahat_beta_raw = np.load(ahat_beta_path)  # (4, 59412)
e_beta_raw = np.load(e_beta_path)        # (4, 59412)

# Corr
ahat_corr_raw = np.load(f"{INPUT_DIR_AHAT}/group_mean_correlations.npy")  # (4, 59412)
e_corr_raw = np.load(f"{INPUT_DIR_E}/group_mean_correlations.npy")        # (4, 59412)

# ahat_corr_raw = np.load(f"{INPUT_DIR_AHAT}/group_mean_correlations_LORO.npy")  # (4, 59412)
# e_corr_raw = np.load(f"{INPUT_DIR_E}/group_mean_correlations_LORO.npy")        # (4, 59412)

# ISC (noise ceiling)
isc_path = f"{INPUT_DIR_ISC}/isc_map.npy"
if not os.path.exists(isc_path):
     raise FileNotFoundError(f"ISC map not found at: {isc_path}\nPlease ensure step0 has been run.")

isc = np.load(isc_path)  # (59412,)

print(f"Ahat beta shape: {ahat_beta_raw.shape}")
print(f"ISC shape: {isc.shape}")
print(f"ISC range: {isc.min():.4f} to {isc.max():.4f}")

# =============================================================================
# NORMALIZE WITHOUT THRESHOLD
# =============================================================================
print("\nNormalizing without threshold...")

# sqrt(ISC) - avoid division by zero
sqrt_isc = np.sqrt(np.clip(isc, 1e-6, None))  # clip small values, no NaN

# Normalize: value / sqrt(ISC)
# Shape: (4, 59412) / (59412,) -> broadcast
ahat_beta_norm = ahat_beta_raw / sqrt_isc[np.newaxis, :]
e_beta_norm = e_beta_raw / sqrt_isc[np.newaxis, :]
ahat_corr_norm = ahat_corr_raw / sqrt_isc[np.newaxis, :]
e_corr_norm = e_corr_raw / sqrt_isc[np.newaxis, :]

# Contrast
contrast_beta_norm = ahat_beta_norm - e_beta_norm
contrast_corr_norm = ahat_corr_norm - e_corr_norm

print(f"Ahat beta norm range: {np.nanmin(ahat_beta_norm):.4f} to {np.nanmax(ahat_beta_norm):.4f}")
print(f"E beta norm range: {np.nanmin(e_beta_norm):.4f} to {np.nanmax(e_beta_norm):.4f}")

# =============================================================================
# SAVE
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(f"{OUTPUT_DIR}/ahat_beta_norm_no_thresh.npy", ahat_beta_norm)
np.save(f"{OUTPUT_DIR}/e_beta_norm_no_thresh.npy", e_beta_norm)
np.save(f"{OUTPUT_DIR}/ahat_corr_norm_no_thresh.npy", ahat_corr_norm)
np.save(f"{OUTPUT_DIR}/e_corr_norm_no_thresh.npy", e_corr_norm)
np.save(f"{OUTPUT_DIR}/contrast_beta_norm_no_thresh.npy", contrast_beta_norm)
np.save(f"{OUTPUT_DIR}/contrast_corr_norm_no_thresh.npy", contrast_corr_norm)

print(f"\nSaved to: {OUTPUT_DIR}/")
print("Files saved: ahat/e beta & corr, contrast beta & corr (all normalized)")

# =============================================================================
# CREATE CIFTI FOR VISUALIZATION
# =============================================================================
print("\nCreating CIFTI files for wb_view...")

# Load Template
template = nib.load(CIFTI_TEMPLATE)

def save_dscalar(data, name, output_dir):
    """Save 4-layer data as dscalar.nii"""
    # data shape: (4, 59412)
    time_axis = nib.cifti2.ScalarAxis(['L1', 'L2', 'L3', 'L4'])
    
    # Get brain model axis from template (Axis 1 usually)
    brain_axis = template.header.get_axis(1)
    
    # Pad to match template shape if needed (e.g. 59k -> 91k)
    # The template typically has ~91282 vertices (grayordinates)
    n_vertices_template = brain_axis.name.shape[0] # or similar way to get size
    
    if data.shape[1] != n_vertices_template:
        # Assuming data is cortical surface only (~59k) and template includes subcortical (~91k)
        padded = np.zeros((4, n_vertices_template))
        # Be careful here: Ensure the mapping is correct (first N indices)
        # Usually valid for standard fs_LR 32k cifti files
        padded[:, :data.shape[1]] = data 
        final_data = padded
    else:
        final_data = data
    
    new_header = nib.Cifti2Header.from_axes((time_axis, brain_axis))
    new_img = nib.Cifti2Image(final_data.astype(np.float32), new_header)
    
    nib.save(new_img, f"{output_dir}/{name}.dscalar.nii")
    print(f"  Saved: {name}.dscalar.nii")

save_dscalar(ahat_beta_norm, "ahat_beta_norm_no_thresh", OUTPUT_DIR)
save_dscalar(e_beta_norm, "e_beta_norm_no_thresh", OUTPUT_DIR)
save_dscalar(ahat_corr_norm, "ahat_corr_norm_no_thresh", OUTPUT_DIR)
save_dscalar(e_corr_norm, "e_corr_norm_no_thresh", OUTPUT_DIR)
save_dscalar(contrast_beta_norm, "contrast_beta_norm_no_thresh", OUTPUT_DIR)
save_dscalar(contrast_corr_norm, "contrast_corr_norm_no_thresh", OUTPUT_DIR)

print("\nDone! Open in wb_view to visualize.")