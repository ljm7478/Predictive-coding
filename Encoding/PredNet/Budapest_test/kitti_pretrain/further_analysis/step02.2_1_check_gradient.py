"""
Step 2.2-Pre: Validate Principal Gradient Map (FIXED)
===================================================
Fix: Apply Medial Wall Mask using Indices from the Cifti Template.
     (32k Full Surface -> ~29k Cortical Surface)
"""

import numpy as np
import nibabel as nib
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis"
OUTPUT_DIR = f"{BASE_DIR}/step2.2_gradient_analysis"

# Gradient Source Files
GRADIENT_DIR = "/combinelab/03_user/jungmin/01_project/03_Continual_learning/01_code/02_Hierarchy/00_original/Margulies_gradient/gradient_analysis_orig/gradient_data/templates"
GRADIENT_LH = f"{GRADIENT_DIR}/hcp.embed.all.179.lh.dscalar.nii"
GRADIENT_RH = f"{GRADIENT_DIR}/hcp.embed.all.179.rh.dscalar.nii"

# Template (Reference for Valid Vertices)
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"

# =============================================================================
# MAIN SCRIPT
# =============================================================================
print("Validating Gradient 1 with Medial Wall Masking...")

# 1. Get Valid Vertex Indices from Template
#    The template knows which vertices (out of 32k) are valid cortex.
print("Extracting Medial Wall mask from template...")
template = nib.load(CIFTI_TEMPLATE)
brain_axis = template.header.get_axis(1)

valid_idx_lh = None
valid_idx_rh = None

for name, slc, bm in brain_axis.iter_structures():
    if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
        valid_idx_lh = bm.vertex  # Array of valid vertex indices (e.g., [0, 1, 2, ... 32491])
        print(f"  LH Valid Cortex: {len(valid_idx_lh)} vertices")
    elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
        valid_idx_rh = bm.vertex
        print(f"  RH Valid Cortex: {len(valid_idx_rh)} vertices")

# 2. Load Raw Gradient Data
#    These are likely 32,492 vertices each (Full surface including medial wall)
print("\nLoading Raw Gradients...")
img_lh = nib.load(GRADIENT_LH)
data_lh = img_lh.get_fdata()
if data_lh.ndim > 1: data_lh = data_lh[0, :] # Get Component 0

img_rh = nib.load(GRADIENT_RH)
data_rh = img_rh.get_fdata()
if data_rh.ndim > 1: data_rh = data_rh[0, :] # Get Component 0

print(f"  Raw Input Shapes -> LH: {data_lh.shape}, RH: {data_rh.shape}")

# 3. Apply Masking (The Critical Fix)
#    We assume the input data is on the standard fs_LR_32k mesh (0 to 32491).
#    We use the indices from the template to pick only the valid ones.

# Safety Check: Input must be large enough to contain the indices
if data_lh.shape[0] < np.max(valid_idx_lh) or data_rh.shape[0] < np.max(valid_idx_rh):
    print("ERROR: Input data is smaller than the vertex indices requested.")
    print("The input file might already be masked or is in a different resolution.")
else:
    print("\nApplying Medial Wall Mask...")
    g1_lh_masked = data_lh[valid_idx_lh]
    g1_rh_masked = data_rh[valid_idx_rh]
    
    print(f"  Masked Shapes -> LH: {g1_lh_masked.shape}, RH: {g1_rh_masked.shape}")

    # 4. Concatenate
    gradient_combined = np.concatenate([g1_lh_masked, g1_rh_masked])
    print(f"  Final Combined Shape: {gradient_combined.shape}") # Should be ~59412

    # 5. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = f"{OUTPUT_DIR}/gradient1_validation_masked.dscalar.nii"

    # Create Header
    # Note: brain_axis matches the template, which matches our masked data perfectly now.
    scalar_axis = nib.cifti2.ScalarAxis(["Principal_Gradient"])
    
    # Check if sizes match exactly (Robustness)
    if gradient_combined.shape[0] == brain_axis.size:
        final_data = gradient_combined.reshape(1, -1)
    else:
        # If template includes subcortical or other structures, pad with 0
        print(f"  Note: Template has {brain_axis.size} voxels, Data has {gradient_combined.shape[0]}. Padding/Trimming...")
        final_data = np.zeros((1, brain_axis.size), dtype=np.float32)
        final_data[0, :gradient_combined.shape[0]] = gradient_combined

    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    new_img = nib.Cifti2Image(final_data, header=new_header)

    nib.save(new_img, save_path)
    print(f"\n[SUCCESS] Saved masked validation file: {save_path}")
    print("ACTION: Check in wb_view. The striping should be gone.")