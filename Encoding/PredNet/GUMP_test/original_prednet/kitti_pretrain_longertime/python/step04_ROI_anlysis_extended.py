"""
Step 6: V1-DMN Divergence Analysis - Three DMN Definitions Comparison

Creates separate output folders for:
1. divergence_analysis_narrow/  (6 labels)
2. divergence_analysis_medium/  (18 labels) - RECOMMENDED
3. divergence_analysis_wide/    (40+ labels)

Each folder contains identical figure sets for direct comparison.
"""

import os
import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"

GLASSER_PARCELLATION = "/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"

CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

subjects = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15", 
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]

# =============================================================================
# ROI DEFINITIONS
# =============================================================================

# V1/V2 (Same for all)
V1_LABELS = [1, 181]
V2_LABELS = [4, 184]
EARLY_VISUAL_LABELS = V1_LABELS + V2_LABELS

# Hierarchy ROIs (same for all)
V3_LABELS = [5, 185]
V4_LABELS = [6, 186]
MID_VISUAL_LABELS = V3_LABELS + V4_LABELS

MT_LABELS = [23, 203]
MST_LABELS = [24, 204]
V4t_LABELS = [156, 336]
FST_LABELS = [157, 337]
MT_COMPLEX_LABELS = MT_LABELS + MST_LABELS + V4t_LABELS + FST_LABELS

LIP_LABELS = [26, 27, 206, 207]
VIP_LABELS = [28, 208]
AIP_LABELS = [165, 345]
MIP_LABELS = [164, 344]
AREA7_LABELS = [16, 17, 18, 19, 196, 197, 198, 199]
IPS_LABELS = LIP_LABELS + VIP_LABELS + AIP_LABELS + MIP_LABELS + AREA7_LABELS

STS_LABELS = [128, 129, 130, 131, 308, 309, 310, 311]
STG_LABELS = [123, 303]
TE_LABELS = [132, 133, 134, 135, 136, 312, 313, 314, 315, 316]
TEMPORAL_LABELS = STS_LABELS + STG_LABELS + TE_LABELS

# =============================================================================
# THREE DMN DEFINITIONS
# =============================================================================

# --- NARROW (6 labels) ---
DMN_NARROW = {
    'name': 'Narrow',
    'n_labels': 6,
    'folder': 'narrow',
    'labels': [31, 211,     # PCC: POS1 only
               10, 190,     # mPFC: 10v only
               39, 219],    # AG: PGi only
    'description': 'POS1 + 10v + PGi only'
}

# --- MEDIUM (18 labels) - RECOMMENDED ---
DMN_MEDIUM = {
    'name': 'Medium',
    'n_labels': 18,
    'folder': 'medium',
    'labels': [10, 11, 190, 191,                    # mPFC: 10v, 10r
               31, 26, 14, 30, 72,                   # PCC Left: POS1, POS2, RSC, 7m, 31pv
               211, 206, 194, 210, 252,              # PCC Right
               39, 40, 219, 220],                    # AG: PGi, PGs
    'description': 'Core DMN: 10v/10r + POS1/POS2/RSC/7m/31pv + PGi/PGs'
}

# --- WIDE (40+ labels) ---
DMN_WIDE = {
    'name': 'Wide',
    'n_labels': 40,
    'folder': 'wide',
    'labels': list(set(
        [31, 26, 14, 74, 73, 72, 71, 30, 75,         # PCC extended
         211, 206, 194, 254, 253, 252, 251, 210, 255] +
        [10, 11, 12, 14, 15, 47, 48,                  # mPFC extended
         190, 191, 192, 194, 195, 227, 228] +
        [39, 40, 41, 219, 220, 221] +                 # AG extended
        [139, 140, 132, 319, 320, 312]                # Lateral temporal
    )),
    'description': 'Extended DMN + Lateral Temporal'
}

DMN_DEFINITIONS = [DMN_NARROW, DMN_MEDIUM, DMN_WIDE]

LAYER_NAMES = ['L1', 'L2', 'L3', 'L4']
NUM_LAYERS = 4


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_glasser_parcellation(parcellation_path, n_cortex=59412):
    img = nib.load(parcellation_path)
    data = img.get_fdata()
    if data.ndim > 1:
        data = data.flatten()
    if len(data) > n_cortex:
        data = data[:n_cortex]
    return data.astype(int)


def create_roi_mask(labels, roi_label_list):
    return np.isin(labels, roi_label_list)


def load_subject_data(subject_dir, filename):
    npy_path = subject_dir / filename.replace('.mat', '.npy')
    mat_path = subject_dir / filename.replace('.npy', '.mat')
    
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


def extract_roi_mean(beta_map, roi_mask):
    if beta_map.ndim == 1:
        return np.nanmean(beta_map[roi_mask])
    else:
        return np.nanmean(beta_map[:, roi_mask], axis=1)


def load_all_subject_betas(base_dir, subjects, feature_type):
    save_root = Path(base_dir) / feature_type
    all_betas = []
    valid_subjects = []
    
    for subject in subjects:
        subject_dir = save_root / subject
        beta_data = load_subject_data(subject_dir, 'beta_pc1_all_layers.npy')
        
        if beta_data is not None:
            if beta_data.ndim == 1:
                beta_data = beta_data.reshape(1, -1)
            all_betas.append(beta_data)
            valid_subjects.append(subject)
    
    if len(all_betas) == 0:
        return None, []
    
    return np.stack(all_betas, axis=0), valid_subjects


def compute_roi_profiles(all_betas, glasser_labels, roi_labels):
    n_subjects, n_layers, n_voxels = all_betas.shape
    mask = create_roi_mask(glasser_labels, roi_labels)
    
    profiles = np.zeros((n_subjects, n_layers))
    for subj_idx in range(n_subjects):
        profiles[subj_idx] = extract_roi_mean(all_betas[subj_idx], mask)
    
    return profiles


def compute_hierarchy_profiles(all_betas, glasser_labels, dmn_labels):
    hierarchy_order = [
        ('V1/V2', EARLY_VISUAL_LABELS),
        ('V3/V4', MID_VISUAL_LABELS),
        ('MT+', MT_COMPLEX_LABELS),
        ('IPS', IPS_LABELS),
        ('STS', TEMPORAL_LABELS),
        ('DMN', dmn_labels),
    ]
    
    n_subjects, n_layers, n_voxels = all_betas.shape
    n_rois = len(hierarchy_order)
    
    hierarchy_profiles = np.zeros((n_subjects, n_layers, n_rois))
    roi_names = []
    
    for roi_idx, (roi_name, roi_labels) in enumerate(hierarchy_order):
        mask = create_roi_mask(glasser_labels, roi_labels)
        roi_names.append(roi_name)
        
        for subj_idx in range(n_subjects):
            hierarchy_profiles[subj_idx, :, roi_idx] = extract_roi_mean(all_betas[subj_idx], mask)
    
    return hierarchy_profiles, roi_names


def compute_dissociation_index(v1_profiles, dmn_profiles):
    return dmn_profiles[:, 3] - v1_profiles[:, 3]


def split_groups(dissociation_indices):
    threshold = np.median(dissociation_indices)
    high_idx = np.where(dissociation_indices > threshold)[0]
    low_idx = np.where(dissociation_indices <= threshold)[0]
    return high_idx, low_idx


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_layer_profiles(v1_ahat, dmn_ahat, v1_e, dmn_e, dmn_def, output_dir):
    """Figure 1: V1 vs DMN layer profiles."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(NUM_LAYERS)
    v1_color = '#2E86AB'
    dmn_color = '#E94F37'
    
    # Ahat
    ax = axes[0]
    v1_mean = np.mean(v1_ahat, axis=0)
    v1_sem = stats.sem(v1_ahat, axis=0)
    dmn_mean = np.mean(dmn_ahat, axis=0)
    dmn_sem = stats.sem(dmn_ahat, axis=0)
    
    ax.errorbar(x, v1_mean, yerr=v1_sem, marker='o', markersize=10, 
                linewidth=2, capsize=5, label='V1/V2', color=v1_color)
    ax.errorbar(x, dmn_mean, yerr=dmn_sem, marker='s', markersize=10, 
                linewidth=2, capsize=5, label=f'DMN ({dmn_def["name"]})', color=dmn_color)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, fontsize=12)
    ax.set_xlabel('PredNet Layer', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title(f'Prediction (A): DMN {dmn_def["name"]} ({dmn_def["n_labels"]} labels)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if dmn_mean[3] > v1_mean[3]:
        ax.annotate('Explaining\nAway', xy=(3.15, (v1_mean[3] + dmn_mean[3])/2),
                   fontsize=11, ha='left', color='green', fontweight='bold')
    
    # Error
    ax = axes[1]
    v1_mean = np.mean(v1_e, axis=0)
    v1_sem = stats.sem(v1_e, axis=0)
    dmn_mean = np.mean(dmn_e, axis=0)
    dmn_sem = stats.sem(dmn_e, axis=0)
    
    ax.errorbar(x, v1_mean, yerr=v1_sem, marker='o', markersize=10, 
                linewidth=2, capsize=5, label='V1/V2', color=v1_color)
    ax.errorbar(x, dmn_mean, yerr=dmn_sem, marker='s', markersize=10, 
                linewidth=2, capsize=5, label=f'DMN ({dmn_def["name"]})', color=dmn_color)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, fontsize=12)
    ax.set_xlabel('PredNet Layer', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title('Error (E)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_v1_dmn_layer_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_group_comparison(v1_ahat, dmn_ahat, dissociation_indices, high_idx, low_idx, dmn_def, output_dir):
    """Figure 2: Strong vs Weak Dissociation groups."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(NUM_LAYERS)
    v1_color = '#2E86AB'
    dmn_color = '#E94F37'
    
    # Strong Dissociation (was High-Fidelity)
    ax = axes[0]
    v1_mean = np.mean(v1_ahat[high_idx], axis=0)
    v1_sem = stats.sem(v1_ahat[high_idx], axis=0)
    dmn_mean = np.mean(dmn_ahat[high_idx], axis=0)
    dmn_sem = stats.sem(dmn_ahat[high_idx], axis=0)
    
    ax.errorbar(x, v1_mean, yerr=v1_sem, marker='o', markersize=10, 
                linewidth=2.5, capsize=5, label='V1/V2', color=v1_color)
    ax.errorbar(x, dmn_mean, yerr=dmn_sem, marker='s', markersize=10, 
                linewidth=2.5, capsize=5, label='DMN', color=dmn_color)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, fontsize=12)
    ax.set_xlabel('PredNet Layer', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title(f'Strong Dissociation (n={len(high_idx)})\n- Clear Explaining Away', 
                fontsize=14, fontweight='bold', color='darkgreen')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Weak Dissociation (was Low-Fidelity)
    ax = axes[1]
    v1_mean = np.mean(v1_ahat[low_idx], axis=0)
    v1_sem = stats.sem(v1_ahat[low_idx], axis=0)
    dmn_mean = np.mean(dmn_ahat[low_idx], axis=0)
    dmn_sem = stats.sem(dmn_ahat[low_idx], axis=0)
    
    ax.errorbar(x, v1_mean, yerr=v1_sem, marker='o', markersize=10, 
                linewidth=2.5, capsize=5, label='V1/V2', color=v1_color)
    ax.errorbar(x, dmn_mean, yerr=dmn_sem, marker='s', markersize=10, 
                linewidth=2.5, capsize=5, label='DMN', color=dmn_color)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYER_NAMES, fontsize=12)
    ax.set_xlabel('PredNet Layer', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title(f'Weak Dissociation (n={len(low_idx)})\n Mixed/Weak Pattern', 
                fontsize=14, fontweight='bold', color='darkred')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Distribution
    ax = axes[2]
    median_val = np.median(dissociation_indices)
    n, bins, patches = ax.hist(dissociation_indices, bins=15, edgecolor='black', alpha=0.7)
    
    for i, patch in enumerate(patches):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center > median_val:
            patch.set_facecolor('darkgreen')
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor('darkred')
            patch.set_alpha(0.6)
    
    ax.axvline(median_val, color='black', linestyle='--', linewidth=2, 
               label=f'Median = {median_val:.3f}')
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.5)
    ax.set_xlabel('Dissociation Index', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title('Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    
    plt.suptitle(f'DMN {dmn_def["name"]} ({dmn_def["n_labels"]} labels)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_group_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_crossover(v1_ahat, dmn_ahat, v1_e, dmn_e, dmn_def, output_dir):
    """Figure 3: Crossover interaction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = [0, 1]
    v1_color = '#2E86AB'
    dmn_color = '#E94F37'
    
    # Ahat
    ax = axes[0]
    v1_l1, v1_l4 = v1_ahat[:, 0], v1_ahat[:, 3]
    dmn_l1, dmn_l4 = dmn_ahat[:, 0], dmn_ahat[:, 3]
    
    for i in range(len(v1_l1)):
        ax.plot(x, [v1_l1[i], v1_l4[i]], color=v1_color, alpha=0.1, linewidth=0.5)
        ax.plot(x, [dmn_l1[i], dmn_l4[i]], color=dmn_color, alpha=0.1, linewidth=0.5)
    
    ax.errorbar(x, [np.mean(v1_l1), np.mean(v1_l4)], 
               yerr=[stats.sem(v1_l1), stats.sem(v1_l4)],
               marker='o', markersize=14, linewidth=3, capsize=8,
               label='V1/V2', color=v1_color, zorder=10)
    ax.errorbar(x, [np.mean(dmn_l1), np.mean(dmn_l4)], 
               yerr=[stats.sem(dmn_l1), stats.sem(dmn_l4)],
               marker='s', markersize=14, linewidth=3, capsize=8,
               label='DMN', color=dmn_color, zorder=10)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['L1', 'L4'], fontsize=14)
    ax.set_xlabel('PredNet Layer', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title('Prediction (A): Crossover', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Error
    ax = axes[1]
    v1_l1, v1_l4 = v1_e[:, 0], v1_e[:, 3]
    dmn_l1, dmn_l4 = dmn_e[:, 0], dmn_e[:, 3]
    
    for i in range(len(v1_l1)):
        ax.plot(x, [v1_l1[i], v1_l4[i]], color=v1_color, alpha=0.1, linewidth=0.5)
        ax.plot(x, [dmn_l1[i], dmn_l4[i]], color=dmn_color, alpha=0.1, linewidth=0.5)
    
    ax.errorbar(x, [np.mean(v1_l1), np.mean(v1_l4)], 
               yerr=[stats.sem(v1_l1), stats.sem(v1_l4)],
               marker='o', markersize=14, linewidth=3, capsize=8,
               label='V1/V2', color=v1_color, zorder=10)
    ax.errorbar(x, [np.mean(dmn_l1), np.mean(dmn_l4)], 
               yerr=[stats.sem(dmn_l1), stats.sem(dmn_l4)],
               marker='s', markersize=14, linewidth=3, capsize=8,
               label='DMN', color=dmn_color, zorder=10)
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['L1', 'L4'], fontsize=14)
    ax.set_xlabel('PredNet Layer', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title('Error (E): Parallel', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'DMN {dmn_def["name"]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_crossover_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_individual_dissociation(dissociation_indices, dmn_def, output_dir):
    """Figure 4: Individual subject ranking."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    sorted_order = np.argsort(dissociation_indices)[::-1]
    sorted_indices = dissociation_indices[sorted_order]
    median_val = np.median(dissociation_indices)
    
    colors = ['darkgreen' if idx > median_val else 'darkred' for idx in sorted_indices]
    ax.bar(range(len(sorted_indices)), sorted_indices, color=colors, edgecolor='black', alpha=0.7)
    
    ax.axhline(median_val, color='black', linestyle='--', linewidth=2, 
               label=f'Median = {median_val:.3f}')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Subject (ranked)', fontsize=14)
    ax.set_ylabel('Dissociation Index', fontsize=14)
    ax.set_title(f'Individual Variability - DMN {dmn_def["name"]}', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_individual_dissociation.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hierarchy_gradient(ahat_hierarchy, e_hierarchy, roi_names, dmn_def, output_dir):
    """Figure 5: Hierarchical gradient."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    n_rois = len(roi_names)
    x = np.arange(n_rois)
    layer_colors = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
    
    # Ahat
    ax = axes[0]
    for layer_idx in range(NUM_LAYERS):
        layer_data = ahat_hierarchy[:, layer_idx, :]
        mean_vals = np.mean(layer_data, axis=0)
        sem_vals = stats.sem(layer_data, axis=0)
        ax.errorbar(x, mean_vals, yerr=sem_vals, marker='o', markersize=8, 
                   linewidth=2, capsize=4, label=f'L{layer_idx+1}', color=layer_colors[layer_idx])
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11, rotation=15, ha='right')
    ax.set_xlabel('Visual Hierarchy', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title('Prediction (A)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Error
    ax = axes[1]
    for layer_idx in range(NUM_LAYERS):
        layer_data = e_hierarchy[:, layer_idx, :]
        mean_vals = np.mean(layer_data, axis=0)
        sem_vals = stats.sem(layer_data, axis=0)
        ax.errorbar(x, mean_vals, yerr=sem_vals, marker='o', markersize=8, 
                   linewidth=2, capsize=4, label=f'L{layer_idx+1}', color=layer_colors[layer_idx])
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=11, rotation=15, ha='right')
    ax.set_xlabel('Visual Hierarchy', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title('Error (E)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Hierarchical Gradient - DMN {dmn_def["name"]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_hierarchy_gradient.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hierarchy_l4(ahat_hierarchy, e_hierarchy, roi_names, dmn_def, output_dir):
    """Figure 6: L4 comparison across hierarchy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_rois = len(roi_names)
    x = np.arange(n_rois)
    
    ahat_l4 = ahat_hierarchy[:, 3, :]
    e_l4 = e_hierarchy[:, 3, :]
    
    ax.errorbar(x, np.mean(ahat_l4, axis=0), yerr=stats.sem(ahat_l4, axis=0), 
               marker='o', markersize=12, linewidth=3, capsize=6,
               label='Prediction (A) L4', color='#2171b5')
    ax.errorbar(x, np.mean(e_l4, axis=0), yerr=stats.sem(e_l4, axis=0), 
               marker='s', markersize=12, linewidth=3, capsize=6,
               label='Error (E) L4', color='#cb181d')
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_names, fontsize=12, rotation=15, ha='right')
    ax.set_xlabel('Visual Hierarchy', fontsize=14)
    ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
    ax.set_title(f'L4: Prediction vs Error - DMN {dmn_def["name"]}', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fig6_hierarchy_l4.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_four_panel_all_subjects(v1_ahat, dmn_ahat, v1_e, dmn_e, dmn_def, output_dir):
    """Figure 7: 4-panel plot showing individual subjects."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    x = np.arange(NUM_LAYERS)
    
    # Get shared y-limits
    all_vals = np.concatenate([v1_ahat.flatten(), dmn_ahat.flatten(), 
                                v1_e.flatten(), dmn_e.flatten()])
    y_min, y_max = np.percentile(all_vals, [2, 98])
    y_range = y_max - y_min
    ylim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    panels = [
        (axes[0, 0], v1_ahat, 'Prediction (A) - V1/V2', '#2E86AB'),
        (axes[0, 1], dmn_ahat, 'Prediction (A) - DMN', '#E94F37'),
        (axes[1, 0], v1_e, 'Error (E) - V1/V2', '#2E86AB'),
        (axes[1, 1], dmn_e, 'Error (E) - DMN', '#E94F37'),
    ]
    
    for ax, data, title, color in panels:
        mean = np.mean(data, axis=0)
        sem = stats.sem(data, axis=0)
        
        # Individual subjects (light lines)
        for subj_data in data:
            ax.plot(x, subj_data, color=color, alpha=0.15, linewidth=0.8)
        
        # Group mean
        ax.errorbar(x, mean, yerr=sem, marker='o', markersize=12, 
                   linewidth=3, capsize=6, color=color, zorder=10)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_NAMES, fontsize=12)
        ax.set_xlabel('PredNet Layer', fontsize=14)
        ax.set_ylabel('Beta Weight (PC1)', fontsize=14)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Individual Subject Variability - DMN {dmn_def["name"]}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fig7_four_panel_all_subjects.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_statistics(v1_ahat, dmn_ahat, v1_e, dmn_e, dissociation_indices, 
                   high_idx, low_idx, ahat_hierarchy, roi_names, dmn_def, output_dir):
    """Save statistical results."""
    results = []
    results.append("="*70)
    results.append(f"STATISTICAL RESULTS - DMN {dmn_def['name'].upper()} ({dmn_def['n_labels']} labels)")
    results.append("="*70)
    results.append(f"\nDescription: {dmn_def['description']}\n")
    
    # Dissociation Index
    results.append("[1] DISSOCIATION INDEX")
    results.append("-"*40)
    results.append(f"    Mean:   {np.mean(dissociation_indices):.4f}")
    results.append(f"    Std:    {np.std(dissociation_indices):.4f}")
    results.append(f"    Median: {np.median(dissociation_indices):.4f}")
    
    t_stat, p_val = stats.ttest_1samp(dissociation_indices, 0)
    results.append(f"    t-test vs 0: t={t_stat:.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
    results.append("")
    
    # Layer x Region
    results.append("[2] AHAT - LAYER - REGION")
    results.append("-"*40)
    for layer_idx, layer_name in enumerate(LAYER_NAMES):
        t_stat, p_val = stats.ttest_rel(v1_ahat[:, layer_idx], dmn_ahat[:, layer_idx])
        diff = np.mean(dmn_ahat[:, layer_idx] - v1_ahat[:, layer_idx])
        sig = "*" if p_val < 0.05 else ""
        results.append(f"    {layer_name}: V1={np.mean(v1_ahat[:, layer_idx]):+.4f}, "
                      f"DMN={np.mean(dmn_ahat[:, layer_idx]):+.4f}, "
                      f"-={diff:+.4f}, p={p_val:.4f} {sig}")
    results.append("")
    
    # Interaction
    results.append("[3] INTERACTION CONTRAST")
    results.append("-"*40)
    v1_slope = v1_ahat[:, 3] - v1_ahat[:, 0]
    dmn_slope = dmn_ahat[:, 3] - dmn_ahat[:, 0]
    interaction = dmn_slope - v1_slope
    t_stat, p_val = stats.ttest_1samp(interaction, 0)
    results.append(f"    V1 slope:    {np.mean(v1_slope):+.4f}")
    results.append(f"    DMN slope:   {np.mean(dmn_slope):+.4f}")
    results.append(f"    Interaction: {np.mean(interaction):+.4f}")
    results.append(f"    t-test: t={t_stat:.3f}, p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
    results.append("")
    
    # Groups
    results.append("[4] GROUP COMPARISON")
    results.append("-"*40)
    results.append(f"    Strong Dissociation (n={len(high_idx)}): {np.mean(dissociation_indices[high_idx]):+.4f}")
    results.append(f"    Weak Dissociation (n={len(low_idx)}):   {np.mean(dissociation_indices[low_idx]):+.4f}")
    results.append("")
    
    # Hierarchy
    results.append("[5] HIERARCHY GRADIENT (L4 Ahat)")
    results.append("-"*40)
    ahat_l4 = ahat_hierarchy[:, 3, :]
    for i, roi in enumerate(roi_names):
        results.append(f"    {roi}: {np.mean(ahat_l4[:, i]):+.4f}")
    
    results_text = "\n".join(results)
    with open(output_dir / 'statistical_results.txt', 'w') as f:
        f.write(results_text)
    
    return results_text


# =============================================================================
# MAIN
# =============================================================================

def run_analysis_for_definition(dmn_def, glasser_labels, ahat_betas, e_betas, ahat_subjects):
    """Run full analysis for one DMN definition."""
    print(f"\n{'='*70}")
    print(f" Processing DMN {dmn_def['name']} ({dmn_def['n_labels']} labels)")
    print(f"{'='*70}")
    
    output_dir = Path(BASE_SAVE_DIR) / dmn_def['folder']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute profiles
    print("  Computing ROI profiles...")
    v1_ahat = compute_roi_profiles(ahat_betas, glasser_labels, EARLY_VISUAL_LABELS)
    dmn_ahat = compute_roi_profiles(ahat_betas, glasser_labels, dmn_def['labels'])
    v1_e = compute_roi_profiles(e_betas, glasser_labels, EARLY_VISUAL_LABELS)
    dmn_e = compute_roi_profiles(e_betas, glasser_labels, dmn_def['labels'])
    
    n_vertices = np.sum(np.isin(glasser_labels, dmn_def['labels']))
    print(f"    DMN vertices: {n_vertices}")
    
    # Hierarchy profiles
    print("  Computing hierarchy profiles...")
    ahat_hierarchy, roi_names = compute_hierarchy_profiles(ahat_betas, glasser_labels, dmn_def['labels'])
    e_hierarchy, _ = compute_hierarchy_profiles(e_betas, glasser_labels, dmn_def['labels'])
    
    # Dissociation Index
    print("  Computing dissociation index...")
    dissociation_indices = compute_dissociation_index(v1_ahat, dmn_ahat)
    high_idx, low_idx = split_groups(dissociation_indices)
    
    print(f"    Mean: {np.mean(dissociation_indices):.4f}")
    print(f"    Strong Dissociation: n={len(high_idx)}, Weak Dissociation: n={len(low_idx)}")
    
    # Generate figures
    print("  Generating figures...")
    plot_layer_profiles(v1_ahat, dmn_ahat, v1_e, dmn_e, dmn_def, output_dir)
    print("    fig1_v1_dmn_layer_profiles.png")
    
    plot_group_comparison(v1_ahat, dmn_ahat, dissociation_indices, high_idx, low_idx, dmn_def, output_dir)
    print("    fig2_group_comparison.png")
    
    plot_crossover(v1_ahat, dmn_ahat, v1_e, dmn_e, dmn_def, output_dir)
    print("    fig3_crossover_interaction.png")
    
    plot_individual_dissociation(dissociation_indices, dmn_def, output_dir)
    print("    fig4_individual_dissociation.png")
    
    plot_hierarchy_gradient(ahat_hierarchy, e_hierarchy, roi_names, dmn_def, output_dir)
    print("    fig5_hierarchy_gradient.png")
    
    plot_hierarchy_l4(ahat_hierarchy, e_hierarchy, roi_names, dmn_def, output_dir)
    print("    fig6_hierarchy_l4.png")
    
    plot_four_panel_all_subjects(v1_ahat, dmn_ahat, v1_e, dmn_e, dmn_def, output_dir)
    print("    fig7_four_panel_all_subjects.png")
    
    # Statistics
    print("  Saving statistics...")
    stats_text = save_statistics(v1_ahat, dmn_ahat, v1_e, dmn_e, dissociation_indices,
                                 high_idx, low_idx, ahat_hierarchy, roi_names, dmn_def, output_dir)
    print("    statistical_results.txt")
    
    # Save data
    np.savez(
        output_dir / 'divergence_analysis_data.npz',
        v1_ahat=v1_ahat, dmn_ahat=dmn_ahat,
        v1_e=v1_e, dmn_e=dmn_e,
        ahat_hierarchy=ahat_hierarchy, e_hierarchy=e_hierarchy,
        roi_names=np.array(roi_names),
        dissociation_indices=dissociation_indices,
        high_idx=high_idx, low_idx=low_idx,
        subjects=np.array(ahat_subjects),
        dmn_definition=dmn_def['name'],
        dmn_labels=np.array(dmn_def['labels'])
    )
    print("    divergence_analysis_data.npz")
    
    print(f"  - Output: {output_dir}")
    
    return {
        'name': dmn_def['name'],
        'dissociation_mean': np.mean(dissociation_indices),
        'dissociation_std': np.std(dissociation_indices),
        'n_vertices': n_vertices
    }


def main():
    print("\n" + "="*70)
    print(" Step 6: V1-DMN Divergence Analysis")
    print(" THREE DMN DEFINITIONS: Narrow, Medium, Wide")
    print("="*70)
    
    # Load parcellation
    print("\n[1] Loading Glasser parcellation...")
    glasser_labels = load_glasser_parcellation(GLASSER_PARCELLATION)
    print(f"    Loaded: {glasser_labels.shape}")
    
    # Load betas
    print("\n[2] Loading beta data...")
    ahat_betas, ahat_subjects = load_all_subject_betas(BASE_SAVE_DIR, SUBJECTS, "Ahat")
    e_betas, e_subjects = load_all_subject_betas(BASE_SAVE_DIR, SUBJECTS, "E")
    print(f"    Subjects: {len(ahat_subjects)}")
    print(f"    Shape: {ahat_betas.shape}")
    
    # Run for each definition
    summary = []
    for dmn_def in DMN_DEFINITIONS:
        result = run_analysis_for_definition(dmn_def, glasser_labels, ahat_betas, e_betas, ahat_subjects)
        summary.append(result)
    
    # Print summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"\n{'Definition':<12} {'Labels':<8} {'Vertices':<10} {'Dissociation Mean':<20}")
    print("-"*60)
    for s in summary:
        print(f"{s['name']:<12} {s['n_vertices']:<10} {s['dissociation_mean']:+.4f} - {s['dissociation_std']:.4f}")
    
    print("\n  Output folders:")
    for dmn_def in DMN_DEFINITIONS:
        marker = "-" if dmn_def['name'] == 'Medium' else " "
        print(f"    {marker} {dmn_def['folder']}/")
    
    print("\n  - RECOMMENDED: Medium (18 labels)")
    print("="*70)


if __name__ == "__main__":
    main()