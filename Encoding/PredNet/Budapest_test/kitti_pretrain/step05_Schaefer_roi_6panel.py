"""
Step 2: 6 Network ROI Analysis - Prediction vs Error per Panel
==============================================================
Same layout as original but showing Ahat vs E separately in each panel
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats
import h5py
import scipy.io as sio

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose dataset: 'budapest' or 'gump'
DATASET = 'budapest'

if DATASET == 'budapest':
    BASE_DATA_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"
    OUTPUT_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis/step2_schaefer_roi"
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/16_GrandBudapestHotel/jm_processing/ciftify_outputs/ciftify/sub-sid000005/sub-sid000005/MNINonLinear/Results/task-movie_run-01/task-movie_run-01_Atlas_s3.dtseries.nii"
    SUBJECTS = [
        "sub-sid000005", "sub-sid000007", "sub-sid000009", "sub-sid000010", "sub-sid000013",
        "sub-sid000020", "sub-sid000021", "sub-sid000024", "sub-sid000025", "sub-sid000029",
        "sub-sid000030", "sub-sid000034", "sub-sid000050", "sub-sid000052", "sub-sid000055",
        "sub-sid000114", "sub-sid000120", "sub-sid000134", "sub-sid000142", "sub-sid000278",
        "sub-sid000416", "sub-sid000499", "sub-sid000522", "sub-sid000535", "sub-sid000560"
    ]
else:  # gump
    BASE_DATA_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"
    OUTPUT_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4/divergence_analysis/step2_schaefer_roi"
    CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"
    SUBJECTS = [
        "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
        "sub-06", "sub-09", "sub-10", "sub-14", "sub-15", 
        "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
    ]

SCHAEFER_PATH = "/combinelab/02_data/99_parcellation/Schaefer2018/HCP/fslr32k/cifti/Schaefer2018_1000Parcels_17Networks_order.dlabel.nii"

LAYER_NAMES = ['L1', 'L2', 'L3', 'L4']

# =============================================================================
# 6 ROI DEFINITIONS (same as original)
# =============================================================================

ROI_DEFINITIONS = {
    'Visual': {
        'keywords': ['Vis'],
        'color': '#2E86AB',
    },
    'SomatoMotor': {
        'keywords': ['SomMot'],
        'color': '#27AE60', 
    },
    'DorsalAttn': {
        'keywords': ['DorsAttn'],
        'color': '#8E44AD',
    },
    'VentralAttn': {
        'keywords': ['SalVentAttn'],
        'color': '#E91E63',
    },
    'Control': {
        'keywords': ['Cont'],
        'color': '#F39C12',
    },
    'Default': {
        'keywords': ['Default'],
        'color': '#E94F37',
    },
}

# Colors for Ahat and E
AHAT_COLOR = '#E94F37'  # Red for Prediction
E_COLOR = '#2E86AB'     # Blue for Error


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_schaefer_parcellation(path):
    img = nib.load(path)
    data = img.get_fdata().flatten()
    
    header = img.header
    label_axis = header.get_axis(0)
    
    label_names = {}
    if hasattr(label_axis, 'label'):
        for map_idx, label_dict in enumerate(label_axis.label):
            for lid, (name, rgba) in label_dict.items():
                if lid > 0:
                    label_names[lid] = name
    
    return data.astype(int), label_names


def get_network_mask(parcel_data, label_names, keywords):
    matching_labels = []
    for lid, name in label_names.items():
        for kw in keywords:
            if kw in name:
                matching_labels.append(lid)
                break
    
    mask = np.isin(parcel_data, matching_labels)
    return mask, matching_labels


def load_subject_beta(subject_dir):
    npy_path = subject_dir / 'beta_pc1_all_layers.npy'
    mat_path = subject_dir / 'beta_pc1_all_layers.mat'
    
    if npy_path.exists():
        return np.load(str(npy_path))
    elif mat_path.exists():
        try:
            with h5py.File(mat_path, 'r') as f:
                for key in f.keys():
                    if not key.startswith('#'):
                        return f[key][:].astype(np.float32)
        except:
            data = sio.loadmat(str(mat_path))
            for key in data.keys():
                if not key.startswith('_'):
                    return data[key].astype(np.float32)
    return None


def load_all_subjects(base_dir, subjects, feature_type):
    save_root = Path(base_dir) / feature_type
    all_data = []
    
    for subject in subjects:
        subject_dir = save_root / subject
        data = load_subject_beta(subject_dir)
        if data is not None:
            all_data.append(data)
    
    if len(all_data) == 0:
        return None
    
    return np.stack(all_data, axis=0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print(f" 6 NETWORK ROI: PREDICTION vs ERROR ({DATASET.upper()})")
    print("="*70)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load Schaefer Parcellation
    # =========================================================================
    print("\n[1] Loading Schaefer parcellation...")
    
    schaefer_data, label_names = load_schaefer_parcellation(SCHAEFER_PATH)
    n_voxels_schaefer = len(schaefer_data)
    
    # =========================================================================
    # Create ROI masks
    # =========================================================================
    print("\n[2] Creating ROI masks...")
    
    roi_masks = {}
    for roi_name, roi_info in ROI_DEFINITIONS.items():
        mask, _ = get_network_mask(schaefer_data, label_names, roi_info['keywords'])
        roi_masks[roi_name] = {
            'mask': mask,
            'n_verts': np.sum(mask),
            'color': roi_info['color'],
        }
        print(f"    {roi_name}: {np.sum(mask):,} vertices")
    
    # =========================================================================
    # Load Beta Data
    # =========================================================================
    print("\n[3] Loading beta data...")
    
    ahat_beta = load_all_subjects(BASE_DATA_DIR, SUBJECTS, "Ahat")
    e_beta = load_all_subjects(BASE_DATA_DIR, SUBJECTS, "E")
    
    if ahat_beta is None:
        print("ERROR: Data not found!")
        return
    
    n_subjects, n_layers, n_voxels_data = ahat_beta.shape
    print(f"    Subjects: {n_subjects}")
    print(f"    Voxels: {n_voxels_data}")
    
    # Adjust mask size
    if n_voxels_schaefer != n_voxels_data:
        for roi_name in roi_masks:
            roi_masks[roi_name]['mask'] = roi_masks[roi_name]['mask'][:n_voxels_data]
            roi_masks[roi_name]['n_verts'] = np.sum(roi_masks[roi_name]['mask'])
    
    # =========================================================================
    # Extract ROI Profiles
    # =========================================================================
    print("\n[4] Extracting ROI profiles...")
    
    roi_profiles = {}
    
    for roi_name, roi_info in roi_masks.items():
        mask = roi_info['mask']
        
        ahat_profile = np.zeros((n_subjects, n_layers))
        e_profile = np.zeros((n_subjects, n_layers))
        
        for subj in range(n_subjects):
            for layer in range(n_layers):
                ahat_profile[subj, layer] = np.nanmean(ahat_beta[subj, layer, mask])
                e_profile[subj, layer] = np.nanmean(e_beta[subj, layer, mask])
        
        roi_profiles[roi_name] = {
            'ahat': ahat_profile,
            'e': e_profile,
            'color': roi_info['color'],
            'n_verts': roi_info['n_verts'],
        }
    
    # =========================================================================
    # MAIN FIGURE: 6-Panel L1 vs L4 with Prediction vs Error
    # =========================================================================
    print("\n[5] Creating main figure...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    roi_names = list(roi_profiles.keys())
    
    # Get global y-limits for consistency
    all_vals = []
    for roi_name in roi_names:
        all_vals.extend(roi_profiles[roi_name]['ahat'].flatten())
        all_vals.extend(roi_profiles[roi_name]['e'].flatten())
    y_min, y_max = np.percentile(all_vals, [2, 98])
    y_range = y_max - y_min
    ylim = (y_min - 0.15 * y_range, y_max + 0.15 * y_range)
    
    for idx, roi_name in enumerate(roi_names):
        ax = axes[idx]
        profile = roi_profiles[roi_name]
        
        ahat_data = profile['ahat']  # (n_subjects, 4)
        e_data = profile['e']
        
        # Individual subjects (light lines)
        for subj in range(n_subjects):
            ax.plot([0, 1], [ahat_data[subj, 0], ahat_data[subj, 3]],
                   color=AHAT_COLOR, alpha=0.12, linewidth=0.8)
            ax.plot([0, 1], [e_data[subj, 0], e_data[subj, 3]],
                   color=E_COLOR, alpha=0.12, linewidth=0.8)
        
        # Group mean with error bars
        ahat_l1_mean = np.mean(ahat_data[:, 0])
        ahat_l4_mean = np.mean(ahat_data[:, 3])
        ahat_l1_sem = stats.sem(ahat_data[:, 0])
        ahat_l4_sem = stats.sem(ahat_data[:, 3])
        
        e_l1_mean = np.mean(e_data[:, 0])
        e_l4_mean = np.mean(e_data[:, 3])
        e_l1_sem = stats.sem(e_data[:, 0])
        e_l4_sem = stats.sem(e_data[:, 3])
        
        # Plot Prediction (Ahat)
        ax.errorbar([0, 1], [ahat_l1_mean, ahat_l4_mean],
                   yerr=[ahat_l1_sem, ahat_l4_sem],
                   marker='o', markersize=12, linewidth=3, capsize=6,
                   color=AHAT_COLOR, label='Prediction', zorder=10)
        
        # Plot Error (E)
        ax.errorbar([0, 1], [e_l1_mean, e_l4_mean],
                   yerr=[e_l1_sem, e_l4_sem],
                   marker='s', markersize=12, linewidth=3, capsize=6,
                   color=E_COLOR, label='Error', zorder=10)
        
        # Formatting
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['L1', 'L4'], fontsize=12)
        ax.set_ylabel('Beta Weight', fontsize=11)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        
        # Title with vertex count
        ax.set_title(f'{roi_name}\n({profile["n_verts"]:,} verts)', 
                    fontsize=13, fontweight='bold', color=profile['color'])
        
        # Legend only in first panel
        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)
        
        # Add slope annotations
        ahat_slope = ahat_l4_mean - ahat_l1_mean
        e_slope = e_l4_mean - e_l1_mean
        
        textstr = f'Pred: {ahat_slope:+.4f}\nErr: {e_slope:+.4f}'
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    
    plt.suptitle(f'Prediction vs Error: L1 ¡æ L4 by Network ({DATASET.upper()})',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_6panel_prediction_vs_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig_6panel_prediction_vs_error.png")
    
    # =========================================================================
    # FIGURE 2: Full Layer Profile (L1-L4) per ROI
    # =========================================================================
    print("\n[6] Creating full layer profile figure...")
    
    x = np.arange(4)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, roi_name in enumerate(roi_names):
        ax = axes[idx]
        profile = roi_profiles[roi_name]
        
        ahat_mean = np.mean(profile['ahat'], axis=0)
        ahat_sem = stats.sem(profile['ahat'], axis=0)
        e_mean = np.mean(profile['e'], axis=0)
        e_sem = stats.sem(profile['e'], axis=0)
        
        ax.errorbar(x, ahat_mean, yerr=ahat_sem,
                   marker='o', markersize=10, linewidth=2.5, capsize=5,
                   color=AHAT_COLOR, label='Prediction')
        ax.errorbar(x, e_mean, yerr=e_sem,
                   marker='s', markersize=10, linewidth=2.5, capsize=5,
                   color=E_COLOR, label='Error')
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_NAMES, fontsize=11)
        ax.set_xlabel('PredNet Layer', fontsize=11)
        ax.set_ylabel('Beta Weight', fontsize=11)
        ax.set_title(f'{roi_name}', fontsize=13, fontweight='bold', color=profile['color'])
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=10)
    
    plt.suptitle(f'Layer Profile: Prediction vs Error ({DATASET.upper()})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_6panel_layer_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig_6panel_layer_profile.png")
    
    # =========================================================================
    # FIGURE 3: Original style - Contrast only (for comparison)
    # =========================================================================
    print("\n[7] Creating contrast-only figure (original style)...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, roi_name in enumerate(roi_names):
        ax = axes[idx]
        profile = roi_profiles[roi_name]
        
        contrast = profile['ahat'] - profile['e']
        
        # Individual subjects
        for subj in range(n_subjects):
            ax.plot([0, 1], [contrast[subj, 0], contrast[subj, 3]],
                   color=profile['color'], alpha=0.15, linewidth=0.8)
        
        # Group mean
        l1_mean = np.mean(contrast[:, 0])
        l4_mean = np.mean(contrast[:, 3])
        l1_sem = stats.sem(contrast[:, 0])
        l4_sem = stats.sem(contrast[:, 3])
        
        ax.errorbar([0, 1], [l1_mean, l4_mean],
                   yerr=[l1_sem, l4_sem],
                   marker='o', markersize=12, linewidth=3, capsize=6,
                   color=profile['color'], zorder=10)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['L1', 'L4'], fontsize=12)
        ax.set_ylabel('Contrast (Ahat - E)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        ax.set_title(f'{roi_name}\n({profile["n_verts"]:,} verts)', 
                    fontsize=13, fontweight='bold')
        
        slope = l4_mean - l1_mean
        ax.text(0.98, 0.98, f'Slope: {slope:+.4f}', transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    plt.suptitle(f'Contrast L1 vs L4 by Network ({DATASET.upper()})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_6panel_contrast_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig_6panel_contrast_only.png")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    print("\n[8] Computing statistics...")
    
    results = []
    results.append("="*80)
    results.append(f" {DATASET.upper()}: 6 NETWORK ROI - PREDICTION vs ERROR")
    results.append("="*80)
    
    results.append(f"\n[1] PREDICTION (Ahat) Slope: L4 - L1")
    results.append("-"*70)
    results.append(f"{'Network':<15} {'Verts':>8} {'L1':>10} {'L4':>10} {'Slope':>10} {'t':>8} {'p':>10}")
    
    for roi_name, profile in roi_profiles.items():
        ahat = profile['ahat']
        l1, l4 = np.mean(ahat[:, 0]), np.mean(ahat[:, 3])
        slope_vals = ahat[:, 3] - ahat[:, 0]
        slope = np.mean(slope_vals)
        t_stat, p_val = stats.ttest_1samp(slope_vals, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        results.append(f"{roi_name:<15} {profile['n_verts']:>8} {l1:>+10.4f} {l4:>+10.4f} {slope:>+10.4f} {t_stat:>8.2f} {p_val:>10.4f} {sig}")
    
    results.append(f"\n[2] ERROR (E) Slope: L4 - L1")
    results.append("-"*70)
    results.append(f"{'Network':<15} {'Verts':>8} {'L1':>10} {'L4':>10} {'Slope':>10} {'t':>8} {'p':>10}")
    
    for roi_name, profile in roi_profiles.items():
        e = profile['e']
        l1, l4 = np.mean(e[:, 0]), np.mean(e[:, 3])
        slope_vals = e[:, 3] - e[:, 0]
        slope = np.mean(slope_vals)
        t_stat, p_val = stats.ttest_1samp(slope_vals, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        results.append(f"{roi_name:<15} {profile['n_verts']:>8} {l1:>+10.4f} {l4:>+10.4f} {slope:>+10.4f} {t_stat:>8.2f} {p_val:>10.4f} {sig}")
    
    results.append(f"\n[3] CROSSOVER TEST: Prediction slope vs Error slope (paired t-test)")
    results.append("-"*70)
    results.append(f"{'Network':<15} {'Pred_slope':>12} {'Err_slope':>12} {'Diff':>10} {'t':>8} {'p':>10}")
    
    for roi_name, profile in roi_profiles.items():
        ahat_slope = profile['ahat'][:, 3] - profile['ahat'][:, 0]
        e_slope = profile['e'][:, 3] - profile['e'][:, 0]
        
        diff = np.mean(ahat_slope) - np.mean(e_slope)
        t_stat, p_val = stats.ttest_rel(ahat_slope, e_slope)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        results.append(f"{roi_name:<15} {np.mean(ahat_slope):>+12.4f} {np.mean(e_slope):>+12.4f} {diff:>+10.4f} {t_stat:>8.2f} {p_val:>10.4f} {sig}")
    
    with open(output_dir / 'statistical_results_6roi.txt', 'w') as f:
        f.write("\n".join(results))
    print("    statistical_results_6roi.txt")
    
    # Print summary
    print("\n" + "="*70)
    print(" STATISTICAL SUMMARY")
    print("="*70)
    for line in results:
        print(line)
    
    print(f"\n    Output: {output_dir}")


if __name__ == "__main__":
    main()