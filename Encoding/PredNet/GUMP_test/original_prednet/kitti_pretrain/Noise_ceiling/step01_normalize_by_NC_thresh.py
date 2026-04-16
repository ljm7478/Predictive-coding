"""
Create normalized maps WITH ISC threshold
- ISC < threshold vertices are set to NaN (excluded)
- Normalize by sqrt(ISC) for reliable vertices only
- Prevents unstable values in low-SNR regions (e.g., Visuomotor)
"""

import numpy as np
import os
import nibabel as nib

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose dataset: 'budapest' or 'gump'
DATASET = 'gump'

# ISC Threshold - vertices with ISC below this are set to NaN
ISC_THRESHOLD = 0.1

# Common Root Path 
USER_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained"

if DATASET == 'budapest':
    PROJECT_DIR = f"{USER_ROOT}/Budapest_test"
    ENCODING_DIR = f"{PROJECT_DIR}/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
    
else:  # gump
    PROJECT_DIR = f"{USER_ROOT}/GUMP_test"
    ENCODING_DIR = f"{PROJECT_DIR}/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4"
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Set Input/Output Directories based on selection
INPUT_DIR_AHAT = f"{ENCODING_DIR}/Ahat/group"
INPUT_DIR_E = f"{ENCODING_DIR}/E/group"
INPUT_DIR_ISC = f"{ENCODING_DIR}/divergence_analysis/step0_noise_ceiling"

# Output directory with threshold info
OUTPUT_DIR = f"{ENCODING_DIR}/divergence_analysis/step1_normalized/thresh{ISC_THRESHOLD}_no_range_matched"

print("="*70)
print(f"NC Normalization WITH ISC Threshold")
print("="*70)
print(f"Dataset: {DATASET}")
print(f"ISC Threshold: {ISC_THRESHOLD}")
print(f"Encoding Dir: {ENCODING_DIR}")
print(f"Output Dir: {OUTPUT_DIR}")

# =============================================================================
# LOAD RAW DATA
# =============================================================================
print("\n[1] Loading raw data...")

# Beta
# ahat_beta_path = f"{INPUT_DIR_AHAT}/group_mean_betas_LORO.npy"
# e_beta_path = f"{INPUT_DIR_E}/group_mean_betas_LORO.npy"

ahat_beta_path = f"{INPUT_DIR_AHAT}/group_mean_betas.npy"
e_beta_path = f"{INPUT_DIR_E}/group_mean_betas.npy"

if not os.path.exists(ahat_beta_path):
    raise FileNotFoundError(f"File not found: {ahat_beta_path}")

ahat_beta_raw = np.load(ahat_beta_path)  # (4, 59412)
e_beta_raw = np.load(e_beta_path)        # (4, 59412)

# Corr
# ahat_corr_raw = np.load(f"{INPUT_DIR_AHAT}/group_mean_correlations_LORO.npy")  # (4, 59412)
# e_corr_raw = np.load(f"{INPUT_DIR_E}/group_mean_correlations_LORO.npy")        # (4, 59412)

ahat_corr_raw = np.load(f"{INPUT_DIR_AHAT}/group_mean_correlations.npy")  # (4, 59412)
e_corr_raw = np.load(f"{INPUT_DIR_E}/group_mean_correlations.npy")        # (4, 59412)

# ISC (noise ceiling)
isc_path = f"{INPUT_DIR_ISC}/isc_map.npy"
if not os.path.exists(isc_path):
     raise FileNotFoundError(f"ISC map not found at: {isc_path}\nPlease ensure step0 has been run.")

isc = np.load(isc_path)  # (59412,)

print(f"  Ahat beta shape: {ahat_beta_raw.shape}")
print(f"  ISC shape: {isc.shape}")
print(f"  ISC range: {isc.min():.4f} to {isc.max():.4f}")

# =============================================================================
# APPLY ISC THRESHOLD
# =============================================================================
print(f"\n[2] Applying ISC threshold ({ISC_THRESHOLD})...")

# Create mask for low ISC vertices
low_isc_mask = isc < ISC_THRESHOLD
n_excluded = np.sum(low_isc_mask)
n_total = len(isc)
pct_excluded = n_excluded / n_total * 100

print(f"  Vertices with ISC < {ISC_THRESHOLD}: {n_excluded:,} / {n_total:,} ({pct_excluded:.1f}%)")
print(f"  Vertices retained: {n_total - n_excluded:,} ({100 - pct_excluded:.1f}%)")

# ISC distribution info
print(f"\n  ISC distribution:")
print(f"    Min:    {np.min(isc):.4f}")
print(f"    10th:   {np.percentile(isc, 10):.4f}")
print(f"    25th:   {np.percentile(isc, 25):.4f}")
print(f"    Median: {np.median(isc):.4f}")
print(f"    75th:   {np.percentile(isc, 75):.4f}")
print(f"    90th:   {np.percentile(isc, 90):.4f}")
print(f"    Max:    {np.max(isc):.4f}")

# =============================================================================
# NORMALIZE WITH THRESHOLD
# =============================================================================
print(f"\n[3] Normalizing with threshold...")

# sqrt(ISC) for normalization
sqrt_isc = np.sqrt(isc)

# Set low ISC vertices to NaN BEFORE normalization
sqrt_isc_thresh = sqrt_isc.copy()
sqrt_isc_thresh[low_isc_mask] = np.nan

# Normalize: value / sqrt(ISC)
# Low ISC vertices will become NaN automatically
ahat_beta_norm = ahat_beta_raw / sqrt_isc_thresh[np.newaxis, :]
e_beta_norm = e_beta_raw / sqrt_isc_thresh[np.newaxis, :]
ahat_corr_norm = ahat_corr_raw / sqrt_isc_thresh[np.newaxis, :]
e_corr_norm = e_corr_raw / sqrt_isc_thresh[np.newaxis, :]

# Contrast
contrast_beta_norm = ahat_beta_norm - e_beta_norm
contrast_corr_norm = ahat_corr_norm - e_corr_norm

# Stats after thresholding
print(f"\n  After thresholding:")
print(f"    Ahat beta norm - valid: {np.sum(~np.isnan(ahat_beta_norm[0])):,}, NaN: {np.sum(np.isnan(ahat_beta_norm[0])):,}")
print(f"    Ahat beta norm range (valid): {np.nanmin(ahat_beta_norm):.4f} to {np.nanmax(ahat_beta_norm):.4f}")
print(f"    E beta norm range (valid): {np.nanmin(e_beta_norm):.4f} to {np.nanmax(e_beta_norm):.4f}")

# =============================================================================
# SAVE
# =============================================================================
print(f"\n[4] Saving results...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save numpy arrays
np.save(f"{OUTPUT_DIR}/ahat_beta_norm_thresh.npy", ahat_beta_norm)
np.save(f"{OUTPUT_DIR}/e_beta_norm_thresh.npy", e_beta_norm)
np.save(f"{OUTPUT_DIR}/ahat_corr_norm_thresh.npy", ahat_corr_norm)
np.save(f"{OUTPUT_DIR}/e_corr_norm_thresh.npy", e_corr_norm)
np.save(f"{OUTPUT_DIR}/contrast_beta_norm_thresh.npy", contrast_beta_norm)
np.save(f"{OUTPUT_DIR}/contrast_corr_norm_thresh.npy", contrast_corr_norm)

# Save ISC mask for reference
np.save(f"{OUTPUT_DIR}/isc_mask_excluded.npy", low_isc_mask)
np.save(f"{OUTPUT_DIR}/isc_threshold_used.npy", np.array([ISC_THRESHOLD]))

print(f"  Saved numpy arrays to: {OUTPUT_DIR}/")

# =============================================================================
# CREATE CIFTI FOR VISUALIZATION
# =============================================================================
print(f"\n[5] Creating CIFTI files for wb_view...")

# Load Template
template = nib.load(CIFTI_TEMPLATE)

def save_dscalar(data, name, output_dir):
    """Save 4-layer data as dscalar.nii"""
    # data shape: (4, 59412)
    time_axis = nib.cifti2.ScalarAxis(['L1', 'L2', 'L3', 'L4'])
    
    # Get brain model axis from template
    brain_axis = template.header.get_axis(1)
    
    # Get template size
    n_vertices_template = brain_axis.name.shape[0]
    
    if data.shape[1] != n_vertices_template:
        # Pad to match template shape
        padded = np.zeros((4, n_vertices_template))
        padded[:] = np.nan  # Fill with NaN first
        padded[:, :data.shape[1]] = data 
        final_data = padded
    else:
        final_data = data
    
    new_header = nib.Cifti2Header.from_axes((time_axis, brain_axis))
    new_img = nib.Cifti2Image(final_data.astype(np.float32), new_header)
    
    nib.save(new_img, f"{output_dir}/{name}.dscalar.nii")
    print(f"    Saved: {name}.dscalar.nii")

save_dscalar(ahat_beta_norm, "ahat_beta_norm_thresh", OUTPUT_DIR)
save_dscalar(e_beta_norm, "e_beta_norm_thresh", OUTPUT_DIR)
save_dscalar(ahat_corr_norm, "ahat_corr_norm_thresh", OUTPUT_DIR)
save_dscalar(e_corr_norm, "e_corr_norm_thresh", OUTPUT_DIR)
save_dscalar(contrast_beta_norm, "contrast_beta_norm_thresh", OUTPUT_DIR)
save_dscalar(contrast_corr_norm, "contrast_corr_norm_thresh", OUTPUT_DIR)

# Also save ISC mask as CIFTI for visualization
isc_mask_cifti = low_isc_mask.astype(np.float32)[np.newaxis, :]  # (1, 59412)
time_axis_1 = nib.cifti2.ScalarAxis(['ISC_excluded'])
brain_axis = template.header.get_axis(1)
n_vertices_template = brain_axis.name.shape[0]

if isc_mask_cifti.shape[1] != n_vertices_template:
    padded = np.zeros((1, n_vertices_template))
    padded[:, :isc_mask_cifti.shape[1]] = isc_mask_cifti
    isc_mask_cifti = padded

new_header = nib.Cifti2Header.from_axes((time_axis_1, brain_axis))
new_img = nib.Cifti2Image(isc_mask_cifti.astype(np.float32), new_header)
nib.save(new_img, f"{OUTPUT_DIR}/isc_excluded_mask.dscalar.nii")
print(f"    Saved: isc_excluded_mask.dscalar.nii")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"ISC Threshold: {ISC_THRESHOLD}")
print(f"Vertices excluded: {n_excluded:,} ({pct_excluded:.1f}%)")
print(f"Vertices retained: {n_total - n_excluded:,} ({100 - pct_excluded:.1f}%)")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nFiles created:")
print("  - ahat_beta_norm_thresh.npy / .dscalar.nii")
print("  - e_beta_norm_thresh.npy / .dscalar.nii")
print("  - contrast_beta_norm_thresh.npy / .dscalar.nii")
print("  - isc_excluded_mask.dscalar.nii (for visualization)")
print("\nNaN vertices in output = ISC < threshold (excluded from analysis)")
print("="*70)