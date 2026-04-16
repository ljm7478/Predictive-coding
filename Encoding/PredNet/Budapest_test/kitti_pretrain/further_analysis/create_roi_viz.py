import numpy as np
import nibabel as nib

BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis"

SCHAEFER_PATH = "/combinelab/02_data/99_parcellation/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"

# CIFTI template (59412 vertices)
cifti_template = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
template = nib.load(cifti_template)
brain_axis = template.header.get_axis(1)

# Get vertex indices for LH and RH from our CIFTI
lh_vertices = None
rh_vertices = None
for name, slc, bm in brain_axis.iter_structures():
    if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
        lh_vertices = bm.vertex.copy()  # 29696 indices into 32492
        lh_slice = slc
    elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
        rh_vertices = bm.vertex.copy()  # 29716 indices into 32492
        rh_slice = slc

print(f"LH vertices: {len(lh_vertices)} (indices into 32492)")
print(f"RH vertices: {len(rh_vertices)} (indices into 32492)")

# Load Schaefer (64984 = 32492 LH + 32492 RH)
schaefer_img = nib.load(SCHAEFER_PATH)
schaefer_full = schaefer_img.get_fdata().flatten()  # (64984,)

schaefer_lh = schaefer_full[:32492]   # LH parcels
schaefer_rh = schaefer_full[32492:]   # RH parcels

print(f"\nSchaefer LH shape: {schaefer_lh.shape}")
print(f"Schaefer RH shape: {schaefer_rh.shape}")

# Map Schaefer to our 59412 vertices
# LH: use lh_vertices to index into schaefer_lh
# RH: use rh_vertices to index into schaefer_rh
schaefer_59k = np.zeros(59412)
schaefer_59k[:29696] = schaefer_lh[lh_vertices]
schaefer_59k[29696:59412] = schaefer_rh[rh_vertices]

print(f"\nSchaefer 59k shape: {schaefer_59k.shape}")
print(f"Schaefer 59k unique: {len(np.unique(schaefer_59k))}")
print(f"Schaefer 59k non-zero: {np.sum(schaefer_59k != 0)}")

# Label names
label_axis = schaefer_img.header.get_axis(0)
label_names = {}
for map_idx, label_dict in enumerate(label_axis.label):
    for lid, (name, rgba) in label_dict.items():
        if lid > 0:
            label_names[lid] = name

# =============================================================================
# ROI Definitions
# =============================================================================
roi_definitions = {
    'Visual': (['VisCent', 'VisPeri'], 'Error'),
    'STS_Temporal': (['TempPar'], 'Error'),
    'Superior_Parietal': (['DorsAttn'], 'Pred'),
    'DMN_Precuneus': (['Default'], 'Pred'),
    'Frontal': (['Cont'], 'Pred'),
    'SomatoMotor': (['SomMot'], 'Neutral'),
}

# =============================================================================
# Create ROI Masks (using schaefer_59k!)
# =============================================================================
print("\n=== Creating ROI Masks ===")

roi_masks = {}
roi_combined = np.zeros(59412)

for roi_idx, (roi_name, (keywords, expected)) in enumerate(roi_definitions.items()):
    mask = np.zeros(59412, dtype=bool)
    
    # Find matching label IDs
    matched_lids = []
    for lid, name in label_names.items():
        for kw in keywords:
            if kw in name:
                matched_lids.append(lid)
                break
    
    # Create mask using schaefer_59k (not schaefer_full!)
    for lid in matched_lids:
        mask[schaefer_59k == lid] = True
    
    roi_masks[roi_name] = mask
    roi_combined[mask] = roi_idx + 1
    print(f"  {roi_idx + 1}: {roi_name}: {np.sum(mask):,} voxels")

# =============================================================================
# Save as CIFTI
# =============================================================================
print("\n=== Saving CIFTI ===")

import os
os.makedirs(f"{BASE_DIR}/step2_analysis", exist_ok=True)

# Combined ROI map
scalar_axis = nib.cifti2.ScalarAxis(['ROI_combined'])
padded = np.zeros((1, 91282))
padded[0, :59412] = roi_combined

new_header = nib.Cifti2Header.from_axes((scalar_axis, brain_axis))
new_img = nib.Cifti2Image(padded.astype(np.float32), new_header)
nib.save(new_img, f"{BASE_DIR}/step2_analysis/roi_mask_combined_fixed.dscalar.nii")
print(f"Saved: roi_mask_combined_fixed.dscalar.nii")

# Individual masks
roi_names = list(roi_definitions.keys())
individual_maps = np.zeros((6, 91282))
for i, roi_name in enumerate(roi_names):
    individual_maps[i, :59412] = roi_masks[roi_name].astype(float)

scalar_axis = nib.cifti2.ScalarAxis(roi_names)
new_header = nib.Cifti2Header.from_axes((scalar_axis, brain_axis))
new_img = nib.Cifti2Image(individual_maps.astype(np.float32), new_header)
nib.save(new_img, f"{BASE_DIR}/step2_analysis/roi_mask_individual_fixed.dscalar.nii")
print(f"Saved: roi_mask_individual_fixed.dscalar.nii")

# Also save schaefer_59k for future use
np.save(f"{BASE_DIR}/step2_analysis/schaefer_59k.npy", schaefer_59k)
print(f"Saved: schaefer_59k.npy")

print("\n=== Summary ===")
print("""
ROI values in wb_view:
  1 = Visual (Expected: Error)
  2 = STS_Temporal (Expected: Error)  
  3 = Superior_Parietal (Expected: Pred)
  4 = DMN_Precuneus (Expected: Pred)
  5 = Frontal (Expected: Pred)
  6 = SomatoMotor (Expected: Neutral)
""")