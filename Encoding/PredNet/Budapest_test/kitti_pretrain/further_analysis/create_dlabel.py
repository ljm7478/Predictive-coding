"""
Step 2-Bonus: Create Grouped ROI Label File (The Golden Combo Fix)
==================================================================
Strategy: 
1. Manually build 'Cifti2LabelTable' object (Bypasses parsing errors).
2. Manually initialize 'LabelAxis' with the table object.
This avoids both wb_command text parsing issues and nibabel dict conversion bugs.
"""

import numpy as np
import nibabel as nib
import os
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis"
SCHAEFER_PATH = "/combinelab/02_data/99_parcellation/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
OUTPUT_DIR = f"{BASE_DIR}/step2_analysis"

# ROI Definitions (Color: R, G, B, A)
roi_definitions_ordered = {
    'Visual':       (['Vis'], (0, 0, 1, 1)),          # Blue
    'SomatoMotor':  (['SomMot'], (0, 1, 1, 1)),       # Cyan
    'Dorsal_Attn':  (['DorsAttn'], (0, 0.5, 0, 1)),   # Dark Green
    'SalVent_Attn': (['SalVentAttn'], (1, 0, 1, 1)),  # Magenta
    'Control':      (['Cont'], (1, 0.65, 0, 1)),      # Orange
    'Default':      (['Default'], (1, 0, 0, 1)),      # Red
}

# =============================================================================
# MAIN SCRIPT
# =============================================================================
print("Creating Grouped ROI Label file...")

# 1. Load Templates
template = nib.load(CIFTI_TEMPLATE)
brain_axis = template.header.get_axis(1)

# Get Vertex Mapping
lh_vertices, rh_vertices = None, None
for name, slc, bm in brain_axis.iter_structures():
    if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
        lh_vertices = bm.vertex.copy()
    elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
        rh_vertices = bm.vertex.copy()

# Load Schaefer Atlas
schaefer_img = nib.load(SCHAEFER_PATH)
schaefer_full = schaefer_img.get_fdata().flatten()
schaefer_axis = schaefer_img.header.get_axis(0)

schaefer_59k = np.zeros(59412)
schaefer_59k[:29696] = schaefer_full[:32492][lh_vertices]
schaefer_59k[29696:59412] = schaefer_full[32492:][rh_vertices]

# Extract Schaefer Label Names
schaefer_labels = {}
for map_idx, label_dict in enumerate(schaefer_axis.label):
    for lid, (name, rgba) in label_dict.items():
        if lid > 0:
            schaefer_labels[lid] = name

# 2. Create Data Map & Manual Label Table
grouped_map = np.zeros((1, 59412), dtype=np.int32)

# === KEY FIX: Create Cifti2LabelTable Object Manually ===
label_table = nib.cifti2.Cifti2LabelTable()

# Add Background (Key 0)
# Cifti2Label(key, name, r, g, b, a)
label_table[0] = nib.cifti2.Cifti2Label(0, "???", 0, 0, 0, 0)

print("\nMapping ROIs...")
for idx, (roi_name, (keywords, rgba)) in enumerate(roi_definitions_ordered.items(), start=1):
    
    # Fill Data Map
    matched_lids = [lid for lid, name in schaefer_labels.items() if any(kw in name for kw in keywords)]
    count = 0
    for lid in matched_lids:
        mask = (schaefer_59k == lid)
        grouped_map[0, mask] = idx
        count += np.sum(mask)
        
    # Add to Label Table Object
    # Unpack RGBA tuple: *rgba -> r, g, b, a
    label_table[idx] = nib.cifti2.Cifti2Label(idx, roi_name, *rgba)
    
    print(f"  ID {idx}: {roi_name:<15} ({count:,} voxels) -> Color: {rgba}")

# 3. Create Cifti Header
map_name = "ROI_Groups"

# === KEY FIX: Pass the Object in a Dictionary ===
# LabelAxis([names], {name: label_table_object})
# Passing the object prevents nibabel from trying to 'convert' a dict and failing.
label_axis = nib.cifti2.LabelAxis([map_name], {map_name: label_table})

# Create Header
new_header = nib.cifti2.Cifti2Header.from_axes((label_axis, brain_axis))

# 4. Save
final_data = np.zeros((1, 91282), dtype=np.int32)
final_data[0, :59412] = grouped_map

output_path = f"{OUTPUT_DIR}/roi_groups_6.dlabel.nii"
os.makedirs(OUTPUT_DIR, exist_ok=True)

new_img = nib.Cifti2Image(final_data, header=new_header)
nib.save(new_img, output_path)

print(f"\n[SUCCESS] Saved Grouped Label File: {output_path}")
print("Open in wb_view -> Overlay -> Toolbox -> Drawing Type: 'Outline Colors'")