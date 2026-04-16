"""
Create normalized maps WITHOUT threshold
- All voxels included (no ISC threshold)
- Just normalize by sqrt(ISC)
"""

import numpy as np
import os

BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis"

# =============================================================================
# LOAD RAW DATA
# =============================================================================
print("Loading raw data...")

# Beta
ahat_beta_raw = np.load("/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/Ahat/group/group_mean_betas.npy")  # (4, 59412)
e_beta_raw = np.load("/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/E/group/group_mean_betas.npy")        # (4, 59412)

# Corr
ahat_corr_raw = np.load("/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/Ahat/group/group_mean_correlations.npy")  # (4, 59412)
e_corr_raw = np.load("/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/E/group/group_mean_correlations.npy")        # (4, 59412)

# ISC (noise ceiling)
isc = np.load(f"{BASE_DIR}/step0_noise_ceiling/isc_map.npy")  # (59412,)

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
output_dir = f"{BASE_DIR}/step1_normalized_no_threshold"
os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/ahat_beta_norm_no_thresh.npy", ahat_beta_norm)
np.save(f"{output_dir}/e_beta_norm_no_thresh.npy", e_beta_norm)
np.save(f"{output_dir}/ahat_corr_norm_no_thresh.npy", ahat_corr_norm)
np.save(f"{output_dir}/e_corr_norm_no_thresh.npy", e_corr_norm)
np.save(f"{output_dir}/contrast_beta_norm_no_thresh.npy", contrast_beta_norm)
np.save(f"{output_dir}/contrast_corr_norm_no_thresh.npy", contrast_corr_norm)

print(f"\nSaved to: {output_dir}/")
print("Files:")
print("  - ahat_beta_norm_no_thresh.npy")
print("  - e_beta_norm_no_thresh.npy")
print("  - ahat_corr_norm_no_thresh.npy")
print("  - e_corr_norm_no_thresh.npy")
print("  - contrast_beta_norm_no_thresh.npy")
print("  - contrast_corr_norm_no_thresh.npy")

# =============================================================================
# CREATE CIFTI FOR VISUALIZATION
# =============================================================================
print("\nCreating CIFTI files for wb_view...")

import nibabel as nib

# Template CIFTI
template_path = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
template = nib.load(template_path)

def save_dscalar(data, name, output_dir):
    """Save 4-layer data as dscalar.nii"""
    # data shape: (4, 59412)
    time_axis = nib.cifti2.ScalarAxis(['L1', 'L2', 'L3', 'L4'])
    brain_axis = template.header.get_axis(1)
    
    new_header = nib.Cifti2Header.from_axes((time_axis, brain_axis))
    
    # Pad to match template shape if needed
    if data.shape[1] != 91282:
        padded = np.zeros((4, 91282))
        padded[:, :59412] = data
        data = padded
    
    new_img = nib.Cifti2Image(data.astype(np.float32), new_header)
    nib.save(new_img, f"{output_dir}/{name}.dscalar.nii")
    print(f"  Saved: {name}.dscalar.nii")

save_dscalar(ahat_beta_norm, "ahat_beta_norm_no_thresh", output_dir)
save_dscalar(e_beta_norm, "e_beta_norm_no_thresh", output_dir)
save_dscalar(ahat_corr_norm, "ahat_corr_norm_no_thresh", output_dir)
save_dscalar(e_corr_norm, "e_corr_norm_no_thresh", output_dir)
save_dscalar(contrast_beta_norm, "contrast_beta_norm_no_thresh", output_dir)
save_dscalar(contrast_corr_norm, "contrast_corr_norm_no_thresh", output_dir)

print("\nDone! Open in wb_view to visualize.")