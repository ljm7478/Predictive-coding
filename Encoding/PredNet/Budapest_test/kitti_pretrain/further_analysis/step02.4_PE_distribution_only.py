"""
Analysis 3: Network-specific Optimal Layer Mapping (Complete Pipeline)
======================================================================
Goal: Visualize optimal layer distribution within each network
      For BOTH Prediction and Error

Steps:
1. Load beta data & compute optimal layer
2. Network-wise analysis
3. Generate comparison figure
4. Save CIFTI for flatmap
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
import os
from pathlib import Path
from nibabel import cifti2 as ci

# =============================================================================
# [Step 0] CONFIGURATION
# =============================================================================
BASE_DATA_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"
OUTPUT_DIR = f"{BASE_DATA_DIR}/divergence_analysis/step2.5_optimal_layer"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input files
ROI_MASK_PATH = f"{BASE_DATA_DIR}/divergence_analysis/step2.1_spatial_spread_analysis/roi_validation_masks.dscalar.nii"
ISC_PER_SUBJECT_PATH = f"{BASE_DATA_DIR}/divergence_analysis/step0_noise_ceiling/isc_per_subject.npy"
TEMPLATE_DSCALAR = f"{BASE_DATA_DIR}/Ahat/sub-sid000005/beta_pc1_all_layers.dscalar.nii"

SUBJECTS = [
    "sub-sid000005", "sub-sid000007", "sub-sid000009", "sub-sid000010", "sub-sid000013",
    "sub-sid000020", "sub-sid000021", "sub-sid000024", "sub-sid000025", "sub-sid000029",
    "sub-sid000030", "sub-sid000034", "sub-sid000050", "sub-sid000052", "sub-sid000055",
    "sub-sid000114", "sub-sid000120", "sub-sid000134", "sub-sid000142", "sub-sid000278",
    "sub-sid000416", "sub-sid000499", "sub-sid000522", "sub-sid000535", "sub-sid000560"
]

ISC_THRESHOLD = 0.1

# Network order (hierarchy: low to high)
NETWORK_ORDER = ['Visual', 'SomMot', 'Dorsal_Attn', 'SalVent_Attn', 'Default']
NETWORK_LABELS = ['Visual', 'SomMot', 'DorsAttn', 'SalVent', 'Default']

# Colors
LAYER_COLORS = ['#313695', '#4575b4', '#f46d43', '#a50026']

# =============================================================================
# [Step 1] LOAD RAW BETA DATA
# =============================================================================
print("="*70)
print("STEP 1: Loading raw beta data...")
print("="*70)

def load_subject_beta(subject_dir):
    npy_path = subject_dir / 'beta_pc1_all_layers.npy'
    if npy_path.exists():
        return np.load(str(npy_path))
    return None

def load_all_subjects_raw(base_dir, subjects, feature_type):
    save_root = Path(base_dir) / feature_type
    all_data = []
    for subject in subjects:
        data = load_subject_beta(save_root / subject)
        if data is not None:
            all_data.append(data)
    return np.stack(all_data, axis=0) if all_data else None

# Load Prediction and Error
ahat_raw = load_all_subjects_raw(BASE_DATA_DIR, SUBJECTS, "Ahat")
e_raw = load_all_subjects_raw(BASE_DATA_DIR, SUBJECTS, "E")
isc_per_subject = np.load(ISC_PER_SUBJECT_PATH)

# Align subject counts
n_min = min(ahat_raw.shape[0], isc_per_subject.shape[0])
ahat_raw = ahat_raw[:n_min]
e_raw = e_raw[:n_min]
isc_per_subject = isc_per_subject[:n_min]

n_subjects = n_min
n_voxels = ahat_raw.shape[2] if ahat_raw.shape[1] == 4 else ahat_raw.shape[1]

print(f"Loaded {n_subjects} subjects, {n_voxels} voxels")
print(f"Ahat raw shape: {ahat_raw.shape}")
print(f"E raw shape: {e_raw.shape}")

# =============================================================================
# [Step 2] COMPUTE GROUP-LEVEL BETA (ISC Normalized)
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Computing group-level beta (ISC normalized)...")
print("="*70)

# Storage
ahat_group = np.zeros((n_voxels, 4))
e_group = np.zeros((n_voxels, 4))

# Average ISC for masking
isc_mean = np.mean(isc_per_subject, axis=0)
valid_mask = isc_mean > ISC_THRESHOLD

for s in range(n_subjects):
    curr_p = ahat_raw[s]
    curr_e = e_raw[s]
    
    # Ensure shape is (n_voxels, 4)
    if curr_p.shape[0] == 4:
        curr_p = curr_p.T
    if curr_e.shape[0] == 4:
        curr_e = curr_e.T
    
    # ISC normalization
    curr_isc = isc_per_subject[s]
    denom = curr_isc[:, None] + 1e-8
    
    # Accumulate
    ahat_group += curr_p / denom
    e_group += curr_e / denom

# Average
ahat_group /= n_subjects
e_group /= n_subjects

print(f"Ahat group shape: {ahat_group.shape}")
print(f"E group shape: {e_group.shape}")
print(f"Valid voxels (ISC > {ISC_THRESHOLD}): {np.sum(valid_mask)}")

# =============================================================================
# [Step 3] COMPUTE OPTIMAL LAYER
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Computing optimal layer for each voxel...")
print("="*70)

# argmax across layers (1-indexed)
ahat_optimal = np.argmax(ahat_group, axis=1) + 1
e_optimal = np.argmax(e_group, axis=1) + 1

# Convert to float and mask invalid
ahat_optimal = ahat_optimal.astype(float)
e_optimal = e_optimal.astype(float)
ahat_optimal[~valid_mask] = 0
e_optimal[~valid_mask] = 0

# Print distribution
print("\nPrediction optimal layer distribution:")
for l in range(1, 5):
    count = np.sum(ahat_optimal == l)
    pct = count / np.sum(valid_mask) * 100
    print(f"  Layer {l}: {count} voxels ({pct:.1f}%)")

print("\nError optimal layer distribution:")
for l in range(1, 5):
    count = np.sum(e_optimal == l)
    pct = count / np.sum(valid_mask) * 100
    print(f"  Layer {l}: {count} voxels ({pct:.1f}%)")

# Save intermediate files
np.save(f"{OUTPUT_DIR}/Ahat_optimal_layer.npy", ahat_optimal)
np.save(f"{OUTPUT_DIR}/E_optimal_layer.npy", e_optimal)
np.save(f"{OUTPUT_DIR}/Ahat_group_beta.npy", ahat_group)
np.save(f"{OUTPUT_DIR}/E_group_beta.npy", e_group)
np.save(f"{OUTPUT_DIR}/valid_mask.npy", valid_mask)
print(f"\nSaved intermediate files to: {OUTPUT_DIR}")

# =============================================================================
# [Step 4] LOAD ROI MASKS
# =============================================================================
print("\n" + "="*70)
print("STEP 4: Loading ROI masks...")
print("="*70)

roi_img = nib.load(ROI_MASK_PATH)
roi_masks_full = roi_img.get_fdata()
roi_masks = roi_masks_full[:, :59412]  # Cortex only
print(f"ROI masks shape: {roi_masks.shape}")

# =============================================================================
# [Step 5] COMPUTE NETWORK STATISTICS
# =============================================================================
print("\n" + "="*70)
print("STEP 5: Computing network-wise statistics...")
print("="*70)

def compute_network_stats(optimal_layer, group_beta, roi_masks, valid_mask, network_order):
    """Compute layer distribution and profile for each network"""
    
    layer_distributions = {}
    layer_profiles = {}
    
    for i, net_name in enumerate(network_order):
        roi_mask = roi_masks[i, :] > 0
        combined_mask = roi_mask & valid_mask
        
        if np.sum(combined_mask) < 10:
            continue
        
        # Layer distribution
        opt_layers = optimal_layer[combined_mask]
        counts = [np.sum(opt_layers == l) for l in range(1, 5)]
        total = np.sum(counts)
        pcts = [c / total * 100 for c in counts]
        
        layer_distributions[net_name] = {
            'counts': counts,
            'pcts': pcts,
            'total': total
        }
        
        # Layer profile
        betas = group_beta[combined_mask, :]
        mean_profile = np.mean(betas, axis=0)
        sem_profile = np.std(betas, axis=0) / np.sqrt(np.sum(combined_mask))
        
        layer_profiles[net_name] = {
            'mean': mean_profile,
            'sem': sem_profile
        }
    
    return layer_distributions, layer_profiles

# Compute for Prediction
ahat_dist, ahat_prof = compute_network_stats(ahat_optimal, ahat_group, roi_masks, valid_mask, NETWORK_ORDER)

# Compute for Error
e_dist, e_prof = compute_network_stats(e_optimal, e_group, roi_masks, valid_mask, NETWORK_ORDER)

# Print distributions
print("\n--- Prediction ---")
for net in NETWORK_ORDER:
    if net in ahat_dist:
        pcts = ahat_dist[net]['pcts']
        print(f"  {net}: L1={pcts[0]:.1f}%, L2={pcts[1]:.1f}%, L3={pcts[2]:.1f}%, L4={pcts[3]:.1f}%")

print("\n--- Error ---")
for net in NETWORK_ORDER:
    if net in e_dist:
        pcts = e_dist[net]['pcts']
        print(f"  {net}: L1={pcts[0]:.1f}%, L2={pcts[1]:.1f}%, L3={pcts[2]:.1f}%, L4={pcts[3]:.1f}%")

# =============================================================================
# [Step 6] VISUALIZATION - Prediction vs Error Comparison
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Generating comparison figure...")
print("="*70)

fig, axes = plt.subplots(4, 5, figsize=(20, 14))

for i, (net_name, net_label) in enumerate(zip(NETWORK_ORDER, NETWORK_LABELS)):
    
    # ----- Row 0: Prediction Distribution -----
    ax = axes[0, i]
    if net_name in ahat_dist:
        pcts = ahat_dist[net_name]['pcts']
        left = 0
        for l, (pct, color) in enumerate(zip(pcts, LAYER_COLORS)):
            ax.barh(0, pct, left=left, color=color, edgecolor='white', height=0.6)
            if pct > 8:
                ax.text(left + pct/2, 0, f'L{l+1}\n{pct:.0f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            left += pct
        ax.set_xlim([0, 100])
        ax.set_ylim([-0.5, 0.5])
        ax.axis('off')
        if i == 0:
            ax.text(-5, 0, 'Pred\nDist', ha='right', va='center', fontsize=10, fontweight='bold')
        ax.set_title(f'{net_label}\n(n={ahat_dist[net_name]["total"]})', fontsize=12, fontweight='bold')
    
    # ----- Row 1: Prediction Profile -----
    ax = axes[1, i]
    if net_name in ahat_prof:
        profile = ahat_prof[net_name]
        x = np.arange(1, 5)
        for l in range(4):
            ax.scatter(x[l], profile['mean'][l], c=LAYER_COLORS[l], s=120, zorder=3, edgecolor='black', linewidth=1)
        ax.plot(x, profile['mean'], 'k-', linewidth=2, zorder=2)
        ax.errorbar(x, profile['mean'], yerr=profile['sem'], fmt='none', color='gray', capsize=3, zorder=1)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['L1', 'L2', 'L3', 'L4'], fontsize=9)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Pred Beta', fontsize=10, fontweight='bold')
    
    # ----- Row 2: Error Distribution -----
    ax = axes[2, i]
    if net_name in e_dist:
        pcts = e_dist[net_name]['pcts']
        left = 0
        for l, (pct, color) in enumerate(zip(pcts, LAYER_COLORS)):
            ax.barh(0, pct, left=left, color=color, edgecolor='white', height=0.6)
            if pct > 8:
                ax.text(left + pct/2, 0, f'L{l+1}\n{pct:.0f}%', 
                       ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            left += pct
        ax.set_xlim([0, 100])
        ax.set_ylim([-0.5, 0.5])
        ax.axis('off')
        if i == 0:
            ax.text(-5, 0, 'Error\nDist', ha='right', va='center', fontsize=10, fontweight='bold')
    
    # ----- Row 3: Error Profile -----
    ax = axes[3, i]
    if net_name in e_prof:
        profile = e_prof[net_name]
        x = np.arange(1, 5)
        for l in range(4):
            ax.scatter(x[l], profile['mean'][l], c=LAYER_COLORS[l], s=120, zorder=3, edgecolor='black', linewidth=1)
        ax.plot(x, profile['mean'], 'k-', linewidth=2, zorder=2)
        ax.errorbar(x, profile['mean'], yerr=profile['sem'], fmt='none', color='gray', capsize=3, zorder=1)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['L1', 'L2', 'L3', 'L4'], fontsize=9)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        if i == 0:
            ax.set_ylabel('Error Beta', fontsize=10, fontweight='bold')

# Colorbar
cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
for l, color in enumerate(LAYER_COLORS):
    cbar_ax.barh(0, 1, left=l, color=color, edgecolor='white')
    cbar_ax.text(l + 0.5, 0, f'L{l+1}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
cbar_ax.set_xlim([0, 4])
cbar_ax.axis('off')

plt.suptitle('Network-specific Optimal Layer: Prediction vs Error', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.04, 1, 0.96])

save_path = f"{OUTPUT_DIR}/Network_Optimal_Layer_Comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"Saved: {save_path}")

# =============================================================================
# [Step 7] SAVE CIFTI FOR FLATMAP
# =============================================================================
print("\n" + "="*70)
print("STEP 7: Saving CIFTI files for flatmap visualization...")
print("="*70)

template = nib.load(TEMPLATE_DSCALAR)
template_axes = [template.header.get_axis(i) for i in range(2)]

# ----- Save Prediction network maps -----
scalar_axis_ahat = ci.ScalarAxis([f'Ahat_{net}' for net in NETWORK_ORDER])
header_ahat = ci.Cifti2Header.from_axes((scalar_axis_ahat, template_axes[1]))

ahat_maps = np.zeros((5, 59412), dtype=np.float32)
for i, net_name in enumerate(NETWORK_ORDER):
    roi_mask = roi_masks[i, :] > 0
    combined_mask = roi_mask & valid_mask
    ahat_maps[i, combined_mask] = ahat_optimal[combined_mask]

ahat_cifti = ci.Cifti2Image(ahat_maps, header_ahat)
ahat_cifti_path = f"{OUTPUT_DIR}/Ahat_network_optimal_layer.dscalar.nii"
nib.save(ahat_cifti, ahat_cifti_path)
print(f"Saved: {ahat_cifti_path}")

# ----- Save Error network maps -----
scalar_axis_e = ci.ScalarAxis([f'E_{net}' for net in NETWORK_ORDER])
header_e = ci.Cifti2Header.from_axes((scalar_axis_e, template_axes[1]))

e_maps = np.zeros((5, 59412), dtype=np.float32)
for i, net_name in enumerate(NETWORK_ORDER):
    roi_mask = roi_masks[i, :] > 0
    combined_mask = roi_mask & valid_mask
    e_maps[i, combined_mask] = e_optimal[combined_mask]

e_cifti = ci.Cifti2Image(e_maps, header_e)
e_cifti_path = f"{OUTPUT_DIR}/E_network_optimal_layer.dscalar.nii"
nib.save(e_cifti, e_cifti_path)
print(f"Saved: {e_cifti_path}")

# ----- Save whole-brain optimal layer maps -----
scalar_axis_all = ci.ScalarAxis(['Ahat_whole', 'E_whole'])
header_all = ci.Cifti2Header.from_axes((scalar_axis_all, template_axes[1]))

all_maps = np.zeros((2, 59412), dtype=np.float32)
all_maps[0, :] = ahat_optimal
all_maps[1, :] = e_optimal

all_cifti = ci.Cifti2Image(all_maps, header_all)
all_cifti_path = f"{OUTPUT_DIR}/Whole_brain_optimal_layer.dscalar.nii"
nib.save(all_cifti, all_cifti_path)
print(f"Saved: {all_cifti_path}")

# =============================================================================
# [Step 8] SUMMARY & INSTRUCTIONS
# =============================================================================
print("\n" + "="*70)
print("DONE! SUMMARY")
print("="*70)

print(f"""
OUTPUT FILES:
  1. {OUTPUT_DIR}/Network_Optimal_Layer_Comparison.png
     ¡æ Prediction vs Error comparison figure
  
  2. {ahat_cifti_path}
     ¡æ Prediction optimal layer per network (for wb_view)
  
  3. {e_cifti_path}
     ¡æ Error optimal layer per network (for wb_view)
  
  4. {all_cifti_path}
     ¡æ Whole-brain optimal layer (Ahat & E)

FLATMAP VISUALIZATION (wb_view):
  wb_view {all_cifti_path}
  
  Colormap settings:
    - Palette: ROY-BIG-BL or custom discrete
    - Range: 1 to 4
    - L1 = Blue, L2 = Cyan, L3 = Orange, L4 = Red
""")