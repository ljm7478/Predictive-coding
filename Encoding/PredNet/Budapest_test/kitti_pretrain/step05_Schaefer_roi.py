"""
Step 2: Analyze Contrast by Schaefer 1000 Parcels 17 Networks
==============================================================
ROIs: Visual, VentralAttn, IPL, IPS, PFC, SomatoMotor
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
# ROI DEFINITIONS based on Schaefer 17 Networks
# Network names in Schaefer: VisCent, VisPeri, SomMotA, SomMotB, DorsAttnA, DorsAttnB,
# SalVentAttnA, SalVentAttnB, LimbicA, LimbicB, ContA, ContB, ContC,
# DefaultA, DefaultB, DefaultC, TempPar
# =============================================================================

# Keywords to identify networks in parcel names
ROI_DEFINITIONS = {
    'Visual': {
        'keywords': ['Vis'],
        'color': '#2E86AB',
        'expected': 'negative',  # Prediction suppressed
    },
    'SomatoMotor': {
        'keywords': ['SomMot'],
        'color': '#27AE60', 
        'expected': 'neutral',
    },
    'DorsalAttn': {
        'keywords': ['DorsAttn'],
        'color': '#8E44AD',
        'expected': 'positive',  # Contains IPS
    },
    'VentralAttn': {
        'keywords': ['SalVentAttn'],
        'color': '#E91E63',
        'expected': 'positive',  # Salience/Ventral Attention
    },
    'Control': {
        'keywords': ['Cont'],
        'color': '#F39C12',
        'expected': 'positive',  # PFC regions
    },
    'Default': {
        'keywords': ['Default'],
        'color': '#E94F37',
        'expected': 'positive',  # Precuneus, mPFC
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_schaefer_parcellation(path):
    """Load Schaefer parcellation and extract network info"""
    img = nib.load(path)
    data = img.get_fdata().flatten()
    
    # Get label names
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
    """Get mask for parcels containing any of the keywords"""
    matching_labels = []
    for lid, name in label_names.items():
        for kw in keywords:
            if kw in name:
                matching_labels.append(lid)
                break
    
    mask = np.isin(parcel_data, matching_labels)
    return mask, matching_labels


def load_subject_beta(subject_dir):
    """Load beta_pc1_all_layers.npy"""
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
    """Load all subjects' beta data"""
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


def save_cifti_dscalar(data, template_file, output_file, map_names):
    """Save as CIFTI dscalar"""
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_voxels = data.shape
    brain_axis = template.header.get_axis(1)
    n_brain = brain_axis.size
    
    if n_voxels != n_brain:
        if n_voxels > n_brain:
            data = data[:, :n_brain]
        else:
            padded = np.zeros((n_maps, n_brain), dtype=np.float32)
            padded[:, :n_voxels] = data
            data = padded
    
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    nib.save(new_img, str(output_file))
    print(f"  Saved: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print(f" STEP 2: SCHAEFER 17 NETWORK ROI ANALYSIS ({DATASET.upper()})")
    print("="*70)
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Load Schaefer Parcellation
    # =========================================================================
    print("\n[1] Loading Schaefer 1000 parcels 17 networks...")
    
    schaefer_data, label_names = load_schaefer_parcellation(SCHAEFER_PATH)
    n_voxels_schaefer = len(schaefer_data)
    print(f"    Parcellation shape: {n_voxels_schaefer}")
    print(f"    Number of parcels: {len(label_names)}")
    
    # Print some example labels
    print(f"\n    Example labels:")
    for i, (lid, name) in enumerate(list(label_names.items())[:5]):
        print(f"      {lid}: {name}")
    
    # =========================================================================
    # Create ROI masks
    # =========================================================================
    print("\n[2] Creating ROI masks...")
    
    roi_masks = {}
    for roi_name, roi_info in ROI_DEFINITIONS.items():
        mask, matching_labels = get_network_mask(schaefer_data, label_names, roi_info['keywords'])
        roi_masks[roi_name] = {
            'mask': mask,
            'n_verts': np.sum(mask),
            'n_parcels': len(matching_labels),
            'color': roi_info['color'],
            'expected': roi_info['expected'],
        }
        print(f"    {roi_name}: {np.sum(mask):,} vertices, {len(matching_labels)} parcels")
    
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
    print(f"    Layers: {n_layers}")
    print(f"    Voxels: {n_voxels_data}")
    
    # Adjust Schaefer mask if sizes differ
    if n_voxels_schaefer != n_voxels_data:
        print(f"\n    [INFO] Adjusting Schaefer mask: {n_voxels_schaefer} -> {n_voxels_data}")
        schaefer_data = schaefer_data[:n_voxels_data]
        for roi_name in roi_masks:
            roi_masks[roi_name]['mask'] = roi_masks[roi_name]['mask'][:n_voxels_data]
            roi_masks[roi_name]['n_verts'] = np.sum(roi_masks[roi_name]['mask'])
    
    # =========================================================================
    # Compute Contrast per Layer
    # =========================================================================
    print("\n[4] Computing contrast per layer...")
    
    # Group mean
    ahat_group = np.nanmean(ahat_beta, axis=0)  # (4, n_voxels)
    e_group = np.nanmean(e_beta, axis=0)
    contrast_group = ahat_group - e_group
    
    # Subject-level contrast
    contrast_subj = ahat_beta - e_beta  # (n_subjects, 4, n_voxels)
    
    # =========================================================================
    # Extract ROI Layer Profiles
    # =========================================================================
    print("\n[5] Extracting ROI layer profiles...")
    
    roi_profiles = {}
    
    for roi_name, roi_info in roi_masks.items():
        mask = roi_info['mask']
        
        # Subject-level profiles
        ahat_profile = np.zeros((n_subjects, n_layers))
        e_profile = np.zeros((n_subjects, n_layers))
        contrast_profile = np.zeros((n_subjects, n_layers))
        
        for subj in range(n_subjects):
            for layer in range(n_layers):
                ahat_profile[subj, layer] = np.nanmean(ahat_beta[subj, layer, mask])
                e_profile[subj, layer] = np.nanmean(e_beta[subj, layer, mask])
                contrast_profile[subj, layer] = np.nanmean(contrast_subj[subj, layer, mask])
        
        roi_profiles[roi_name] = {
            'ahat': ahat_profile,
            'e': e_profile,
            'contrast': contrast_profile,
            'color': roi_info['color'],
            'expected': roi_info['expected'],
            'n_verts': roi_info['n_verts'],
        }
        
        # Print summary
        contrast_mean = np.mean(contrast_profile, axis=0)
        print(f"    {roi_name}: L1={contrast_mean[0]:+.4f}, L4={contrast_mean[3]:+.4f}, "
              f"slope={contrast_mean[3]-contrast_mean[0]:+.4f}")
    
    # =========================================================================
    # Figure 1: Contrast Layer Profile (All ROIs)
    # =========================================================================
    print("\n[6] Creating figures...")
    
    x = np.arange(n_layers)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for roi_name, profile in roi_profiles.items():
        contrast_mean = np.mean(profile['contrast'], axis=0)
        contrast_sem = stats.sem(profile['contrast'], axis=0)
        
        ax.errorbar(x, contrast_mean, yerr=contrast_sem,
                   marker='o', markersize=10, linewidth=2.5, capsize=5,
                   label=f"{roi_name} ({profile['n_verts']:,} verts)",
                   color=profile['color'])
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, fontsize=12)
    ax.set_xlabel('PredNet Layer', fontsize=14)
    ax.set_ylabel('Contrast (Ahat - E)', fontsize=14)
    ax.set_title(f'Contrast Layer Profile by Network ({DATASET.upper()})\n'
                 f'Positive = Prediction Enhanced, Negative = Error Enhanced',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_contrast_layer_profile.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig1_contrast_layer_profile.png")
    
    # =========================================================================
    # Figure 2: Ahat and E Separate (2 panels)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Ahat
    ax = axes[0]
    for roi_name, profile in roi_profiles.items():
        ahat_mean = np.mean(profile['ahat'], axis=0)
        ahat_sem = stats.sem(profile['ahat'], axis=0)
        ax.errorbar(x, ahat_mean, yerr=ahat_sem,
                   marker='o', markersize=9, linewidth=2, capsize=4,
                   label=roi_name, color=profile['color'])
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, fontsize=11)
    ax.set_xlabel('PredNet Layer', fontsize=12)
    ax.set_ylabel('Beta Weight', fontsize=12)
    ax.set_title('Prediction (Ahat)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # E
    ax = axes[1]
    for roi_name, profile in roi_profiles.items():
        e_mean = np.mean(profile['e'], axis=0)
        e_sem = stats.sem(profile['e'], axis=0)
        ax.errorbar(x, e_mean, yerr=e_sem,
                   marker='s', markersize=9, linewidth=2, capsize=4,
                   label=roi_name, color=profile['color'])
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, fontsize=11)
    ax.set_xlabel('PredNet Layer', fontsize=12)
    ax.set_ylabel('Beta Weight', fontsize=12)
    ax.set_title('Error (E)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Beta Layer Profile by Network ({DATASET.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_ahat_e_separate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig2_ahat_e_separate.png")
    
    # =========================================================================
    # Figure 3: Hierarchy Effect (L4 - L1) Bar Plot
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    roi_names = list(roi_profiles.keys())
    slopes = []
    colors = []
    sems = []
    
    for roi_name in roi_names:
        profile = roi_profiles[roi_name]
        slope_vals = profile['contrast'][:, 3] - profile['contrast'][:, 0]
        slopes.append(np.mean(slope_vals))
        sems.append(stats.sem(slope_vals))
        colors.append(profile['color'])
    
    bars = ax.bar(roi_names, slopes, color=colors, edgecolor='black', linewidth=1.5,
                  yerr=sems, capsize=5)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add value labels
    for bar, val, sem in zip(bars, slopes, sems):
        y_pos = bar.get_height() + sem + 0.002 if val >= 0 else bar.get_height() - sem - 0.01
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{val:+.4f}',
               ha='center', va='bottom' if val >= 0 else 'top', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Contrast Change (L4 - L1)', fontsize=12)
    ax.set_title(f'Hierarchy Effect: L4 - L1 Contrast Change ({DATASET.upper()})\n'
                 f'Negative = More Prediction Suppression in L4, Positive = More Enhancement',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_hierarchy_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig3_hierarchy_effect.png")
    
    # =========================================================================
    # Figure 4: L1 vs L4 Crossover (per ROI)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (roi_name, profile) in enumerate(roi_profiles.items()):
        if i >= 6:
            break
        ax = axes[i]
        
        # Individual subjects
        for subj in range(n_subjects):
            ax.plot([0, 1], [profile['contrast'][subj, 0], profile['contrast'][subj, 3]],
                   color=profile['color'], alpha=0.15, linewidth=0.8)
        
        # Group mean
        l1_mean = np.mean(profile['contrast'][:, 0])
        l4_mean = np.mean(profile['contrast'][:, 3])
        l1_sem = stats.sem(profile['contrast'][:, 0])
        l4_sem = stats.sem(profile['contrast'][:, 3])
        
        ax.errorbar([0, 1], [l1_mean, l4_mean], yerr=[l1_sem, l4_sem],
                   marker='o', markersize=12, linewidth=3, capsize=6,
                   color=profile['color'], zorder=10)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['L1', 'L4'], fontsize=12)
        ax.set_ylabel('Contrast (Ahat - E)', fontsize=11)
        ax.set_title(f"{roi_name}\n({profile['n_verts']:,} verts)", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add slope annotation
        slope = l4_mean - l1_mean
        ax.text(0.5, 0.95, f'Slope: {slope:+.4f}', transform=ax.transAxes,
               ha='center', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Contrast L1 vs L4 by Network ({DATASET.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_L1vsL4_crossover.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig4_L1vsL4_crossover.png")
    
    # =========================================================================
    # Figure 5: Ahat vs E within each ROI (6 panels)
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (roi_name, profile) in enumerate(roi_profiles.items()):
        if i >= 6:
            break
        ax = axes[i]
        
        ahat_mean = np.mean(profile['ahat'], axis=0)
        ahat_sem = stats.sem(profile['ahat'], axis=0)
        e_mean = np.mean(profile['e'], axis=0)
        e_sem = stats.sem(profile['e'], axis=0)
        
        ax.errorbar(x, ahat_mean, yerr=ahat_sem,
                   marker='o', markersize=10, linewidth=2.5, capsize=5,
                   label='Prediction (Ahat)', color='#E94F37')
        ax.errorbar(x, e_mean, yerr=e_sem,
                   marker='s', markersize=10, linewidth=2.5, capsize=5,
                   label='Error (E)', color='#2E86AB')
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_NAMES, fontsize=11)
        ax.set_xlabel('PredNet Layer', fontsize=11)
        ax.set_ylabel('Beta Weight', fontsize=11)
        ax.set_title(f"{roi_name}", fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Ahat vs E per Network ({DATASET.upper()})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_ahat_vs_e_per_roi.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("    fig5_ahat_vs_e_per_roi.png")
    
    # =========================================================================
    # Statistics
    # =========================================================================
    print("\n[7] Computing statistics...")
    
    results = []
    results.append("="*80)
    results.append(f" {DATASET.upper()}: SCHAEFER 17 NETWORK ROI ANALYSIS")
    results.append("="*80)
    
    results.append(f"\n[1] CONTRAST LAYER PROFILE")
    results.append("-"*70)
    results.append(f"{'Network':<15} {'Verts':>8} {'L1':>10} {'L4':>10} {'Slope':>10} {'t':>8} {'p':>10}")
    
    for roi_name, profile in roi_profiles.items():
        contrast = profile['contrast']
        l1, l4 = np.mean(contrast[:, 0]), np.mean(contrast[:, 3])
        slope_vals = contrast[:, 3] - contrast[:, 0]
        slope = np.mean(slope_vals)
        t_stat, p_val = stats.ttest_1samp(slope_vals, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        n_verts = profile['n_verts']
        results.append(f"{roi_name:<15} {n_verts:>8} {l1:>+10.4f} {l4:>+10.4f} {slope:>+10.4f} {t_stat:>8.2f} {p_val:>10.4f} {sig}")
    
    results.append(f"\n[2] AHAT LAYER PROFILE (L4 - L1)")
    results.append("-"*70)
    results.append(f"{'Network':<15} {'L1':>10} {'L4':>10} {'Slope':>10} {'t':>8} {'p':>10}")
    
    for roi_name, profile in roi_profiles.items():
        ahat = profile['ahat']
        l1, l4 = np.mean(ahat[:, 0]), np.mean(ahat[:, 3])
        slope_vals = ahat[:, 3] - ahat[:, 0]
        slope = np.mean(slope_vals)
        t_stat, p_val = stats.ttest_1samp(slope_vals, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        results.append(f"{roi_name:<15} {l1:>+10.4f} {l4:>+10.4f} {slope:>+10.4f} {t_stat:>8.2f} {p_val:>10.4f} {sig}")
    
    results.append(f"\n[3] ERROR LAYER PROFILE (L4 - L1)")
    results.append("-"*70)
    results.append(f"{'Network':<15} {'L1':>10} {'L4':>10} {'Slope':>10} {'t':>8} {'p':>10}")
    
    for roi_name, profile in roi_profiles.items():
        e = profile['e']
        l1, l4 = np.mean(e[:, 0]), np.mean(e[:, 3])
        slope_vals = e[:, 3] - e[:, 0]
        slope = np.mean(slope_vals)
        t_stat, p_val = stats.ttest_1samp(slope_vals, 0)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        results.append(f"{roi_name:<15} {l1:>+10.4f} {l4:>+10.4f} {slope:>+10.4f} {t_stat:>8.2f} {p_val:>10.4f} {sig}")
    
    with open(output_dir / 'statistical_results.txt', 'w') as f:
        f.write("\n".join(results))
    print("    statistical_results.txt")
    
    # Print summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    for line in results:
        print(line)
    
    # =========================================================================
    # Save ROI mask CIFTI
    # =========================================================================
    print("\n[8] Saving ROI mask...")
    
    roi_mask = np.zeros(n_voxels_data, dtype=np.float32)
    for i, (roi_name, roi_info) in enumerate(roi_masks.items()):
        mask = roi_info['mask']
        if len(mask) > n_voxels_data:
            mask = mask[:n_voxels_data]
        roi_mask[mask] = i + 1
    
    save_cifti_dscalar(roi_mask, CIFTI_TEMPLATE,
                       str(output_dir / "schaefer_roi_mask.dscalar.nii"),
                       ["_".join([f"{i+1}={name}" for i, name in enumerate(roi_masks.keys())])])
    
    print(f"\n    Output directory: {output_dir}")


if __name__ == "__main__":
    main()