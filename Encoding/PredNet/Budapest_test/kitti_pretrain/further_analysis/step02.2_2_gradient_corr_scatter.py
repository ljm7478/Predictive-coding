"""
Step 2.2 (Refined): Clean Scatter Plot (6 Core Networks Only)
=============================================================
Goal: Visualize the functional shift without noise.
      - Exclude: Limbic, Other
      - Include: Visual, SomMot, DorsAttn, SalVent, Control, Default
      - X-axis: Sensory (Left) -> Association (Right) [No Numbers]
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import os
from scipy.stats import pearsonr

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis"
OUTPUT_DIR = f"{BASE_DIR}/step2.2_gradient_analysis"

CONTRAST_PATH = f"{BASE_DIR}/step1_normalized_no_threshold/contrast_beta_norm_no_thresh.npy"
ISC_PATH = f"{BASE_DIR}/step0_noise_ceiling/isc_map.npy"
GRADIENT_LH = "/combinelab/03_user/jungmin/01_project/03_Continual_learning/01_code/02_Hierarchy/00_original/Margulies_gradient/gradient_analysis_orig/gradient_data/templates/hcp.embed.all.179.lh.dscalar.nii"
GRADIENT_RH = "/combinelab/03_user/jungmin/01_project/03_Continual_learning/01_code/02_Hierarchy/00_original/Margulies_gradient/gradient_analysis_orig/gradient_data/templates/hcp.embed.all.179.rh.dscalar.nii"
SCHAEFER_PATH_17 = "/combinelab/02_data/99_parcellation/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
ISC_THRESHOLD = 0.1

# === KEY CHANGE: Only Define 6 Core Networks ===
NET_COLORS = {
    'Visual': '#781286',   # Purple
    'SomMot': '#4682B4',   # Steel Blue (Key!)
    'DorsAttn': '#00760E', # Green
    'SalVent': '#C43AFA',  # Violet
    # 'Control': '#E69422',  # Orange
    'Default': '#CD3E4E'   # Red
    # Limbic and Other are removed
}

TARGET_NETWORKS = list(NET_COLORS.keys())

# =============================================================================
# 1. LOAD & PREPARE DATA
# =============================================================================
print("Loading Data...")
contrast = np.load(CONTRAST_PATH)
isc_map = np.load(ISC_PATH)
reliability_mask = isc_map > ISC_THRESHOLD

# Load Gradient & Mask
template = nib.load(CIFTI_TEMPLATE)
brain_axis = template.header.get_axis(1)
valid_idx_lh, valid_idx_rh = None, None
for name, slc, bm in brain_axis.iter_structures():
    if name == 'CIFTI_STRUCTURE_CORTEX_LEFT': valid_idx_lh = bm.vertex
    elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT': valid_idx_rh = bm.vertex

g_lh = nib.load(GRADIENT_LH).get_fdata(); g_rh = nib.load(GRADIENT_RH).get_fdata()
if g_lh.ndim > 1: g_lh = g_lh[0, :]
if g_rh.ndim > 1: g_rh = g_rh[0, :]
gradient_map = np.concatenate([g_lh[valid_idx_lh], g_rh[valid_idx_rh]])

# === INVERT GRADIENT ===
gradient_map = gradient_map * -1

# Load Atlas
schaefer = nib.load(SCHAEFER_PATH_17).get_fdata().flatten()
schaefer_59k = np.zeros(59412)
schaefer_59k[:29696] = schaefer[:32492][valid_idx_lh]
schaefer_59k[29696:59412] = schaefer[32492:][valid_idx_rh]

# Map Labels
atlas_axis = nib.load(SCHAEFER_PATH_17).header.get_axis(0)
label_to_net = {}
for idx, (name, rgba) in atlas_axis.label[0].items():
    if 'Vis' in name: label_to_net[idx] = 'Visual'
    elif 'SomMot' in name: label_to_net[idx] = 'SomMot'
    elif 'DorsAttn' in name: label_to_net[idx] = 'DorsAttn'
    elif 'SalVent' in name: label_to_net[idx] = 'SalVent'
    elif 'Cont' in name: label_to_net[idx] = 'Control'
    elif 'Default' in name: label_to_net[idx] = 'Default'
    # Limbic/Other will not be in label_to_net keys effectively for coloring, or mapped to None

voxel_nets = []
for val in schaefer_59k:
    net = label_to_net.get(int(val), 'Exclude')
    voxel_nets.append(net)
voxel_nets = np.array(voxel_nets)

# =============================================================================
# 2. FILTER & PLOT
# =============================================================================
print("\nCreating Clean 6-Network Scatter Plot...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
layers_to_plot = [0, 3]
titles = ["Layer 1 (Pixel / Visual)", "Layer 4 (Object / Semantic)"]

# Mask: Reliable AND Valid Gradient AND belong to Target Networks
core_mask = reliability_mask & (gradient_map != 0) & (~np.isnan(gradient_map)) & np.isin(voxel_nets, TARGET_NETWORKS)

for i, ax in enumerate(axes):
    l_idx = layers_to_plot[i]
    
    # Apply Mask
    x = gradient_map[core_mask]
    y = contrast[l_idx, core_mask]
    nets = voxel_nets[core_mask]
    
    # Map colors
    c = [NET_COLORS[n] for n in nets]
    
    # Scatter
    ax.scatter(x, y, c=c, alpha=0.35, s=12, edgecolors='none') # Slightly larger points
    
    # Regression Line (Calculated only on 6 networks)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color='black', linewidth=2.5, linestyle='--')
    
    # Stats
    corr = np.corrcoef(x, y)[0,1]
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, 
            fontsize=15, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Formatting
    ax.set_title(titles[i], fontsize=17, fontweight='bold')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xticks([]) # Remove numbers
    
    # Axis Text
    ax.text(0.02, -0.05, "Sensory\n(Visual/Motor)", transform=ax.transAxes, ha='left', va='top', fontsize=13, fontweight='bold', color='#333333')
    ax.text(0.98, -0.05, "Association\n(DMN/Control)", transform=ax.transAxes, ha='right', va='top', fontsize=13, fontweight='bold', color='#333333')
    ax.set_xlabel("Principal Gradient Axis", fontsize=14, labelpad=20)

axes[0].set_ylabel("PredNet Contrast (P - E)", fontsize=14)

# Legend (Only 6 items)
handles = [mpatches.Patch(color=NET_COLORS[net], label=net) for net in TARGET_NETWORKS]
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=6, fontsize=13)

plt.tight_layout()
save_path = f"{OUTPUT_DIR}/scatter_clean_5networks.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {save_path}")
print("Done! Only the 6 core networks are plotted.")