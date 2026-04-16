"""
Analysis 3.1: Layer-wise Z-Scored Magnitude (Robust Comparison)
===============================================================
Goal: Compare Prediction vs Error magnitude across networks and layers
Method: Z-Score standardization (without mean centering) to preserve
        absolute magnitude information while normalizing across subjects.

Key difference from centered version:
- Centered: Shows only relative difference (P-E), redundant with contrast
- This version: Shows actual Z-scored values, complementary to contrast plot
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from pathlib import Path
import nibabel as nib

# =============================================================================
# [Step 0] CONFIGURATION
# =============================================================================
BASE_DATA_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"
OUTPUT_DIR = f"{BASE_DATA_DIR}/divergence_analysis/step2.3_network_interaction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ISC_PER_SUBJECT_PATH = f"{BASE_DATA_DIR}/divergence_analysis/step0_noise_ceiling/isc_per_subject.npy"
SCHAEFER_PATH_17 = "/combinelab/02_data/99_parcellation/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"

SUBJECTS = [
    "sub-sid000005", "sub-sid000007", "sub-sid000009", "sub-sid000010", "sub-sid000013",
    "sub-sid000020", "sub-sid000021", "sub-sid000024", "sub-sid000025", "sub-sid000029",
    "sub-sid000030", "sub-sid000034", "sub-sid000050", "sub-sid000052", "sub-sid000055",
    "sub-sid000114", "sub-sid000120", "sub-sid000134", "sub-sid000142", "sub-sid000278",
    "sub-sid000416", "sub-sid000499", "sub-sid000522", "sub-sid000535", "sub-sid000560"
]

TARGET_NETS = ['Visual', 'SomMot', 'DorsAttn', 'SalVent', 'Default']
COLORS = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
ISC_THRESHOLD = 0.1

# =============================================================================
# [Step 1] DATA LOADING FUNCTIONS
# =============================================================================
def load_subject_beta(subject_dir):
    """Load beta_pc1 for all layers from a subject directory."""
    npy_path = subject_dir / 'beta_pc1_all_layers.npy'
    if npy_path.exists():
        return np.load(str(npy_path))
    return None

def load_all_subjects_raw(base_dir, subjects, feature_type):
    """Load raw beta data for all subjects."""
    save_root = Path(base_dir) / feature_type
    all_data = []
    print(f"Loading {feature_type}...")
    for subject in subjects:
        data = load_subject_beta(save_root / subject)
        if data is not None:
            all_data.append(data)
    return np.stack(all_data, axis=0) if all_data else None

# Load datasets
print("Loading Dataset...")
ahat_raw = load_all_subjects_raw(BASE_DATA_DIR, SUBJECTS, "Ahat")
e_raw = load_all_subjects_raw(BASE_DATA_DIR, SUBJECTS, "E")
isc_per_subject = np.load(ISC_PER_SUBJECT_PATH)

# Align subject counts
n_min = min(ahat_raw.shape[0], isc_per_subject.shape[0])
ahat_raw = ahat_raw[:n_min]
e_raw = e_raw[:n_min]
isc_per_subject = isc_per_subject[:n_min]
n_subjects = n_min

print(f"Loaded {n_subjects} subjects")

# =============================================================================
# [Step 2] NETWORK MAPPING
# =============================================================================
print("Mapping Networks...")
schaefer = nib.load(SCHAEFER_PATH_17).get_fdata().flatten()
if len(schaefer) > ahat_raw.shape[2]:
    schaefer = schaefer[:ahat_raw.shape[2]]

# Extract network labels from atlas
atlas_axis = nib.load(SCHAEFER_PATH_17).header.get_axis(0)
label_dict = atlas_axis.label[0]
voxel_net_group = []

for val in schaefer:
    if val == 0:
        voxel_net_group.append("None")
        continue
    full_name = label_dict.get(int(val), ("",))[0]
    group = "Other"
    for target in TARGET_NETS:
        check_name = target if target != 'Visual' else 'Vis'
        if check_name in full_name:
            group = target
            break
    voxel_net_group.append(group)
voxel_net_group = np.array(voxel_net_group)

# =============================================================================
# [Step 3] CALCULATE Z-SCORED MAGNITUDES (NO MEAN CENTERING)
# =============================================================================
print("Calculating Layer-wise Z-Scored Magnitudes...")

# Storage arrays: (Subject, Network, Layer)
pred_mag_layer = np.zeros((n_subjects, len(TARGET_NETS), 4)) * np.nan
err_mag_layer = np.zeros((n_subjects, len(TARGET_NETS), 4)) * np.nan

for s in range(n_subjects):
    curr_p = ahat_raw[s]
    curr_e = e_raw[s]
    
    # Ensure shape is (Voxels, Layers)
    if curr_p.shape[0] == 4:
        curr_p = curr_p.T
    if curr_e.shape[0] == 4:
        curr_e = curr_e.T
    
    # Get ISC mask for reliable voxels
    curr_isc = isc_per_subject[s]
    valid_mask = (curr_isc > ISC_THRESHOLD)
    
    if np.sum(valid_mask) == 0:
        continue
    
    # Extract valid voxels only
    p_valid = curr_p[valid_mask]  # shape: (n_valid_voxels, 4)
    e_valid = curr_e[valid_mask]
    
    # -------------------------------------------------------------------------
    # Z-Score Normalization (Subject-wise)
    # Normalize P and E together to preserve their relative scale
    # This prevents subjects with extreme values from dominating group stats
    # -------------------------------------------------------------------------
    combined_data = np.concatenate([p_valid.flatten(), e_valid.flatten()])
    subj_mean = np.mean(combined_data)
    subj_std = np.std(combined_data)
    
    p_z = (p_valid - subj_mean) / (subj_std + 1e-8)
    e_z = (e_valid - subj_mean) / (subj_std + 1e-8)
    
    # Map back to full array for network masking
    norm_p_full = np.full_like(curr_p, np.nan)
    norm_e_full = np.full_like(curr_e, np.nan)
    norm_p_full[valid_mask] = p_z
    norm_e_full[valid_mask] = e_z
    
    # -------------------------------------------------------------------------
    # Network Aggregation (NO CENTERING - key difference!)
    # Store actual Z-scored values to show absolute magnitude
    # -------------------------------------------------------------------------
    for n_idx, net in enumerate(TARGET_NETS):
        net_mask = (voxel_net_group == net) & valid_mask
        if np.sum(net_mask) < 10:
            continue
        
        # Mean Z-score across voxels within this network
        p_profile = np.nanmean(norm_p_full[net_mask], axis=0)  # shape: (4,)
        e_profile = np.nanmean(norm_e_full[net_mask], axis=0)
        
        # Direct assignment without centering
        # This preserves the actual magnitude information
        pred_mag_layer[s, n_idx, :] = p_profile
        err_mag_layer[s, n_idx, :] = e_profile

# =============================================================================
# [Step 4] VISUALIZATION
# =============================================================================
print("Generating Z-Scored Magnitude Plot...")

fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=True)
layers = ['Layer 1 (Low)', 'Layer 2', 'Layer 3', 'Layer 4 (High)']
x = np.arange(len(TARGET_NETS))
width = 0.35

for l in range(4):
    ax = axes[l]
    
    # Calculate group statistics
    mean_p = np.nanmean(pred_mag_layer[:, :, l], axis=0)
    mean_e = np.nanmean(err_mag_layer[:, :, l], axis=0)
    sem_p = np.nanstd(pred_mag_layer[:, :, l], axis=0) / np.sqrt(n_subjects)
    sem_e = np.nanstd(err_mag_layer[:, :, l], axis=0) / np.sqrt(n_subjects)
    
    # Grouped bar plot
    ax.bar(x - width/2, mean_p, width, label='Prediction',
           color='white', edgecolor=COLORS, hatch='//', linewidth=1.5,
           yerr=sem_p, capsize=3)
    ax.bar(x + width/2, mean_e, width, label='Error',
           color=COLORS, alpha=0.6, linewidth=0,
           yerr=sem_e, capsize=3)
    
    # Styling
    ax.set_title(layers[l], fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_NETS, rotation=20, fontsize=10)
    ax.axhline(0, color='k', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    if l == 0:
        ax.set_ylabel("Z-Scored Beta (a.u.)", fontsize=12)
        ax.legend(loc='upper left', fontsize=11)

plt.suptitle("Layer-wise Signal Magnitude (Z-Scored)", fontsize=18, y=1.02)
plt.tight_layout()

save_path = f"{OUTPUT_DIR}/Layerwise_Signal_Magnitude_Zscore.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {save_path}")

# =============================================================================
# [Step 5] PRINT SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*60)
print("SUMMARY: Mean Z-Scored Beta by Network and Layer")
print("="*60)

for n_idx, net in enumerate(TARGET_NETS):
    print(f"\n{net}:")
    print(f"  {'Layer':<10} {'Prediction':>12} {'Error':>12} {'Diff (P-E)':>12}")
    print(f"  {'-'*46}")
    for l in range(4):
        p_val = np.nanmean(pred_mag_layer[:, n_idx, l])
        e_val = np.nanmean(err_mag_layer[:, n_idx, l])
        diff = p_val - e_val
        print(f"  Layer {l+1:<3} {p_val:>12.4f} {e_val:>12.4f} {diff:>12.4f}")