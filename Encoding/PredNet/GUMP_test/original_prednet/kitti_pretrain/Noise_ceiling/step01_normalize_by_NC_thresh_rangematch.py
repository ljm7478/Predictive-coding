"""
Create normalized maps WITHOUT threshold + Global Robust Range
- Structure: Single dscalar file per signal type (containing 4 layers)
- Normalization: Raw Beta / sqrt(ISC)
- Visualization: Global 98th percentile applied to the 4-layer file directly
"""

import numpy as np
import os
import nibabel as nib
import subprocess

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose dataset: 'budapest' or 'gump'
DATASET = 'gump'

# Common Root Path
USER_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained"

if DATASET == 'budapest':
    PROJECT_DIR = "{}/Budapest_test".format(USER_ROOT)
    ENCODING_DIR = "{}/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4".format(PROJECT_DIR)
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
    
else:  # gump
    PROJECT_DIR = "{}/GUMP_test".format(USER_ROOT)
    ENCODING_DIR = "{}/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4".format(PROJECT_DIR)
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

INPUT_DIR_AHAT = "{}/Ahat/group".format(ENCODING_DIR)
INPUT_DIR_E = "{}/E/group".format(ENCODING_DIR)
INPUT_DIR_ISC = "{}/divergence_analysis/step0_noise_ceiling".format(ENCODING_DIR)

# Output directory
OUTPUT_DIR = "{}/divergence_analysis/step1_normalized/thresh_range_global_merged".format(ENCODING_DIR)

print("Dataset: {}".format(DATASET))
print("Output:  {}".format(OUTPUT_DIR))

# =============================================================================
# LOAD & NORMALIZE
# =============================================================================
print("\nLoading and Normalizing...")

# Load
ahat_raw = np.load("{}/group_mean_betas.npy".format(INPUT_DIR_AHAT))
e_raw = np.load("{}/group_mean_betas.npy".format(INPUT_DIR_E))
isc = np.load("{}/isc_map.npy".format(INPUT_DIR_ISC))

# ahat_raw = np.load("{}/group_mean_betas_LORO.npy".format(INPUT_DIR_AHAT))
# e_raw = np.load("{}/group_mean_betas_LORO.npy".format(INPUT_DIR_E))
# isc = np.load("{}/isc_map.npy".format(INPUT_DIR_ISC))

# Normalize
sqrt_isc = np.sqrt(np.clip(isc, 1e-6, None))
sqrt_isc[isc < 0.1] = np.nan ## threshold apply

ahat_norm = ahat_raw / sqrt_isc[np.newaxis, :]
e_norm = e_raw / sqrt_isc[np.newaxis, :]
contrast_norm = ahat_norm - e_norm

print("Data Shape: {}".format(ahat_norm.shape)) # Should be (4, 59412)

# =============================================================================
# COMPUTE GLOBAL RANGE
# =============================================================================
print("\nComputing Global Range (98th percentile)...")

stat_mask = isc > 0.1
combined_vals = np.concatenate([
    ahat_norm[:, stat_mask].flatten(),
    e_norm[:, stat_mask].flatten()
])

global_max = np.percentile(np.abs(combined_vals), 98)
print("Global Max: +/- {:.4f}".format(global_max))

# =============================================================================
# SAVE MERGED FILES & APPLY PALETTE
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
template = nib.load(CIFTI_TEMPLATE)

def save_and_palette(data, filename, global_range):
    """
    Saves a (4, N) array as a single dscalar file and applies the palette.
    """
    # 1. Prepare Header (4 Layers)
    map_names = ['Layer1', 'Layer2', 'Layer3', 'Layer4']
    time_axis = nib.cifti2.ScalarAxis(map_names)
    brain_axis = template.header.get_axis(1)
    
    # 2. Handle Padding if vertices don't match template
    n_template = brain_axis.name.shape[0]
    if data.shape[1] != n_template:
        final_data = np.full((4, n_template), np.nan)
        final_data[:, :data.shape[1]] = data
    else:
        final_data = data
        
    # 3. Create Image
    header = nib.Cifti2Header.from_axes((time_axis, brain_axis))
    img = nib.Cifti2Image(final_data.astype(np.float32), header)
    
    # 4. Save Initial File
    out_path = "{}/{}.dscalar.nii".format(OUTPUT_DIR, filename)
    nib.save(img, out_path)
    
    # 5. Apply Palette (Overwrite)
    subprocess.run([
        "wb_command", "-cifti-palette", out_path,
        "MODE_USER_SCALE", out_path,
        "-pos-user", "0", str(global_range),
        "-neg-user", "0", str(-global_range),
        "-palette-name", "ROY-BIG-BL",
        "-disp-pos", "true", "-disp-neg", "true"
    ], check=True)
    
    print("Saved & Paletted: {}".format(filename))

print("\nSaving 4-layer dscalar files...")

# Process 3 main files
save_and_palette(ahat_norm, "ahat_beta_norm_global", global_max)
save_and_palette(e_norm, "e_beta_norm_global", global_max)
save_and_palette(contrast_norm, "contrast_beta_norm_global", global_max)

# Save range info
with open("{}/range_info.txt".format(OUTPUT_DIR), 'w') as f:
    f.write("Global Range: +/- {:.4f}\n".format(global_max))

print("\nDone! 3 files generated in total.")