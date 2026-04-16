"""
Step 2: Layer-wise Contrast Pattern (Hierarchical order) - FINAL v2
===================================================================
Updates:
1. Fixed Legend Position (Moved up to avoid overlap)
2. Added ROI Mask Validation (Saves .dscalar.nii of defined ROIs)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
from matplotlib.patches import Patch

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis"

# Input Paths
CONTRAST_PATH = f"{BASE_DIR}/step1_normalized_no_threshold/contrast_beta_norm_no_thresh.npy"
ISC_PATH = f"{BASE_DIR}/step0_noise_ceiling/isc_map.npy" 

# Templates
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
SCHAEFER_PATH = "/combinelab/02_data/99_parcellation/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"

# Analysis Parameters
ISC_THRESHOLD = 0.1  # Only analyze voxels with meaningful signal
MIN_VOXELS = 10      # Minimum voxels required to compute stats

# ROI definitions - Ordered by Functional Hierarchy
roi_definitions_ordered = {
    'Visual':       (['Vis'], 'Error'),           
    'SomMot':  (['SomMot'], 'Neutral'),      
    'Dorsal_Attn':  (['DorsAttn'], 'Error'),      
    'SalVent_Attn': (['SalVentAttn'], 'Pred'),    
    # 'Control':      (['Cont'], 'Pred'),           
    'Default':      (['Default'], 'Pred'),        
}

# =============================================================================
# HELPER: SAVE CIFTI
# =============================================================================
def save_dscalar(data_matrix, map_names, template_path, output_path):
    """Saves a (n_maps, n_voxels) matrix as dscalar.nii"""
    template = nib.load(template_path)
    brain_axis = template.header.get_axis(1) # BrainModelAxis
    
    # 1. Create Scalar Axis (Map Names)
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    
    # 2. Check shape and pad if necessary (91282 standard)
    n_brain_voxels = brain_axis.size
    if data_matrix.shape[1] < n_brain_voxels:
        padded = np.zeros((data_matrix.shape[0], n_brain_voxels), dtype=np.float32)
        padded[:, :data_matrix.shape[1]] = data_matrix
        data_matrix = padded
    
    # 3. Create Header and Image
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    new_img = nib.Cifti2Image(data_matrix.astype(np.float32), header=new_header)
    
    nib.save(new_img, output_path)
    print(f"Saved CIFTI: {output_path}")

# =============================================================================
# 1. LOAD DATA & MAP SCHAEFER
# =============================================================================
print("Loading data...")

contrast = np.load(CONTRAST_PATH)
isc_map = np.load(ISC_PATH)

# Reliability Mask
reliability_mask = isc_map > ISC_THRESHOLD
print(f"ISC Mask (> {ISC_THRESHOLD}): {np.sum(reliability_mask):,} voxels")

# Load Template and Map Schaefer
template = nib.load(CIFTI_TEMPLATE)
brain_axis = template.header.get_axis(1)

lh_vertices, rh_vertices = None, None
for name, slc, bm in brain_axis.iter_structures():
    if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
        lh_vertices = bm.vertex.copy()
    elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
        rh_vertices = bm.vertex.copy()

schaefer_img = nib.load(SCHAEFER_PATH)
schaefer_full = schaefer_img.get_fdata().flatten()

schaefer_59k = np.zeros(59412)
schaefer_59k[:29696] = schaefer_full[:32492][lh_vertices]
schaefer_59k[29696:59412] = schaefer_full[32492:][rh_vertices]

# Extract Label Names
label_axis = schaefer_img.header.get_axis(0)
label_names = {}
for map_idx, label_dict in enumerate(label_axis.label):
    for lid, (name, rgba) in label_dict.items():
        if lid > 0:
            label_names[lid] = name

# =============================================================================
# 2. VALIDATION: SAVE ROI MASKS
# =============================================================================
print("\n[Validation] Creating and saving ROI masks...")

roi_keys = list(roi_definitions_ordered.keys())
n_rois = len(roi_keys)
roi_mask_matrix = np.zeros((n_rois, 59412))

for i, roi_name in enumerate(roi_keys):
    keywords = roi_definitions_ordered[roi_name][0]
    
    # Find matching labels
    matched_lids = [lid for lid, name in label_names.items() if any(kw in name for kw in keywords)]
    
    # Mark voxels
    for lid in matched_lids:
        roi_mask_matrix[i, schaefer_59k == lid] = 1.0

# Save to file
output_dir = f"{BASE_DIR}/step2.1_spatial_spread_analysis"
os.makedirs(output_dir, exist_ok=True)
mask_output_path = f"{output_dir}/roi_validation_masks.dscalar.nii"

save_dscalar(roi_mask_matrix, roi_keys, CIFTI_TEMPLATE, mask_output_path)
print("  -> Use wb_view to check if 'Visual', 'Default' etc. cover correct areas.")


# =============================================================================
# 3. COMPUTE STATISTICS (With ISC Intersection)
# =============================================================================
print("\nComputing statistics...")
results = []

for i, roi_name in enumerate(roi_keys):
    expected = roi_definitions_ordered[roi_name][1]
    
    # Get the anatomical mask we just created
    roi_mask_anat = roi_mask_matrix[i, :] > 0
    
    # INTERSECT with Reliability Mask
    final_mask = roi_mask_anat & reliability_mask
    
    if np.sum(final_mask) < MIN_VOXELS:
        continue

    for layer_idx in range(4):
        vals = contrast[layer_idx, final_mask]
        vals = vals[~np.isnan(vals) & np.isfinite(vals)]
        
        if len(vals) > 0:
            mean_val = np.mean(vals)
            sem_val = np.std(vals) / np.sqrt(len(vals))
            t_stat, p_val = stats.ttest_1samp(vals, 0)
        else:
            mean_val, sem_val, p_val = 0, 0, 1.0
            
        results.append({
            'ROI': roi_name,
            'Layer': layer_idx + 1,
            'Contrast_mean': mean_val,
            'Contrast_sem': sem_val,
            'p_value': p_val
        })

df = pd.DataFrame(results)

# =============================================================================
# 4. VISUALIZATION (Fixed Legend)
# =============================================================================
print("Generating Plots...")
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

all_vals = df['Contrast_mean'].values
all_sems = df['Contrast_sem'].values
y_max = np.nanmax(all_vals + all_sems) * 1.4
y_min = np.nanmin(all_vals - all_sems) * 1.4

roi_labels_plot = [r.replace('_', '\n') for r in roi_keys]

for layer_idx in range(4):
    ax = axes[layer_idx]
    layer_df = df[df['Layer'] == layer_idx + 1]
    
    x = np.arange(len(roi_keys))
    vals, sems, pvals = [], [], []
    
    for roi in roi_keys:
        row = layer_df[layer_df['ROI'] == roi]
        if not row.empty:
            vals.append(row['Contrast_mean'].values[0])
            sems.append(row['Contrast_sem'].values[0])
            pvals.append(row['p_value'].values[0])
        else:
            vals.append(0); sems.append(0); pvals.append(1.0)
            
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in vals]
    ax.bar(x, vals, yerr=sems, color=colors, alpha=0.85, capsize=4, edgecolor='black', linewidth=0.8, width=0.7)
    ax.axhline(0, color='black', linewidth=1)
    
    # Stars
    for i, (val, sem, p) in enumerate(zip(vals, sems, pvals)):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        if sig:
            offset = (y_max - y_min) * 0.03
            y_pos = val + sem + offset if val > 0 else val - sem - offset*1.5
            ax.text(x[i], y_pos, sig, ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_title(f'Layer {layer_idx + 1}', fontsize=16, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels_plot, fontsize=11)
    ax.set_ylim([y_min, y_max])
    if layer_idx == 0: ax.set_ylabel('Contrast (Pred - Error)', fontsize=13)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

# Legend moved UP
legend_elements = [
    Patch(facecolor='#d62728', edgecolor='black', alpha=0.85, label='Prediction Dominant (+)'),
    Patch(facecolor='#1f77b4', edgecolor='black', alpha=0.85, label='Error Dominant (-)'),
]
# bbox_to_anchor adjusted: (x, y) -> y > 1.0 moves it above the axes
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 1.08), 
           fontsize=12, frameon=True, framealpha=0.9)

plt.suptitle(f'Functional Hierarchy of Prediction vs. Error (ISC > {ISC_THRESHOLD})', 
             fontsize=18, fontweight='bold', y=1.02) # slightly lower than legend
plt.tight_layout()

save_path = f'{output_dir}/layer_contrast_hierarchical_FINAL_v2.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved Plot: {save_path}")