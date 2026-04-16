"""
Layer Profile Visualization (Data-Driven ROIs, Grouped)
- Uses ROI mask from create_ROI_mask_corr.py
- Groups parcels into broader functional categories (Early Visual, Motion, etc.)
- Layer profile per group (Wen et al. 2018 style)
"""

import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION
# ============================================================

USE_LOG10 = True

# ROI mask from cap1500 encoding
CAP1500_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4'

# NC-normalized data from LORO_PCAcap_log10 (ISC-based, shared)
NC_BASE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap_log10/layer4'

DATADRIVEN_ROI_MASK = os.path.join(CAP1500_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'ROI_mask_datadriven.npy')
DATADRIVEN_PARCEL_CSV = os.path.join(CAP1500_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'parcel_stats.csv')

NC_BETA_DIR = os.path.join(NC_BASE_DIR, 'divergence_analysis', 'step1_normalized_beta', 'no_thresh_range_global_merged')
NC_BETA_AHAT_FILE = 'ahat_beta_norm_global.dscalar.nii'
NC_BETA_E_FILE = 'e_beta_norm_global.dscalar.nii'

# NC-normalized correlation: r / sqrt(ISC)
# sqrt(ISC) is the noise ceiling for r (Spearman-Brown prophecy)
NC_CORR_DIR = os.path.join(NC_BASE_DIR, 'divergence_analysis', 'step1_normalized_corr', 'no_thresh_range_global_merged')
NC_CORR_AHAT_FILE = 'ahat_corr_norm_global.dscalar.nii'
NC_CORR_E_FILE = 'e_corr_norm_global.dscalar.nii'

SAVE_DIR_BETA = os.path.join(NC_BASE_DIR, 'Results', 'Q1', 'layer_profile_datadriven_grouped_beta')
SAVE_DIR_CORR = os.path.join(NC_BASE_DIR, 'Results', 'Q1', 'layer_profile_datadriven_grouped_corr')

N_LAYERS = 4

# ============================================================
# 2. PARCEL GROUPING
# ============================================================
# Functional groups based on Glasser MMP atlas literature.
# Each parcel from create_ROI_mask_corr.py is assigned to one group.
# Order matters: groups appear in plots in this order.

ROI_GROUPS = {
    'Early Visual':   ['V1', 'V2', 'V3', 'V4', 'V8', 'V3CD', 'ProS', 'DVT'],
    'Mid-Dorsal':     ['V3A', 'V3B', 'V6', 'V6A', 'V7', 'POS1'],
    'Motion':         ['MT', 'MST', 'V4t', 'FST'],
    'Object':         ['LO1', 'LO2', 'LO3', 'PIT', 'VVC'],
    'Face':           ['FFC'],
    'Scene/Ventral':  ['VMV1', 'VMV2', 'VMV3', 'PH'],
    'Auditory':       ['A1', 'A4', 'A5', 'PBelt', 'MBelt', 'LBelt', 'RI'],
    'STS/TPOJ':       ['STV', 'STSdp', 'STSvp', 'TPOJ1', 'TPOJ2', 'TPOJ3', 'PSL'],
    'Parietal':       ['IPS1', 'IP0', '7PC', '7Am', '5mv', 'PCV', 'PGp', 'PGi', 'PFcm'],
    'Visuomotor':     ['LIPv', 'MIP', 'VIP'],
    'Frontal':        ['FEF', 'PEF', 'IFJp'],
}

GROUP_ORDER = list(ROI_GROUPS.keys())

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def load_datadriven_rois(roi_mask_path, csv_path):
    """Load ROI mask + parcel metadata."""
    roi_mask = np.load(roi_mask_path).astype(np.int32)
    parcels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parcels.append({
                'roi_idx': int(row['roi_idx']),
                'parcel': row['parcel'],
                'n_total': int(row['n_total']),
                'n_sig': int(row['n_sig']),
            })
    return roi_mask, parcels


def load_beta_cifti(beta_path):
    """Load beta values from CIFTI file. Shape: (n_layers, n_vertices)."""
    img = nib.load(beta_path)
    return img.get_fdata()


def assign_parcels_to_groups(parcels, roi_groups):
    """
    Map each parcel in the data-driven mask to a group.

    Returns:
        group_to_parcels: {group_name: [parcel dict, ...]}
        unassigned: list of parcel names not matched to any group
    """
    # Build name -> group lookup
    parcel_to_group = {}
    for group, names in roi_groups.items():
        for n in names:
            parcel_to_group[n] = group

    group_to_parcels = {g: [] for g in roi_groups.keys()}
    unassigned = []
    for p in parcels:
        g = parcel_to_group.get(p['parcel'])
        if g is None:
            unassigned.append(p['parcel'])
        else:
            group_to_parcels[g].append(p)

    return group_to_parcels, unassigned


def extract_group_layer_profile(beta_data, roi_mask, group_to_parcels, n_layers=4):
    """
    For each group, aggregate vertices across all member parcels and compute
    mean and SEM per layer.

    Returns:
        profiles_mean: {group: [layer1, ..., layerN]}
        profiles_sem:  {group: [layer1_sem, ..., layerN_sem]}
        n_voxels:      {group: total voxel count}
        member_counts: {group: number of parcels in this group}
    """
    n_vertices_beta = beta_data.shape[-1]
    n = min(len(roi_mask), n_vertices_beta)
    mask_trunc = roi_mask[:n]

    profiles_mean = {}
    profiles_sem = {}
    n_voxels = {}
    member_counts = {}

    for group, parcels in group_to_parcels.items():
        # Collect vertex indices from ALL parcels in this group
        all_indices = []
        for p in parcels:
            idx = np.where(mask_trunc == p['roi_idx'])[0]
            all_indices.extend(idx.tolist())
        all_indices = np.array(all_indices, dtype=np.int64)

        n_voxels[group] = len(all_indices)
        member_counts[group] = len(parcels)

        if len(all_indices) > 0:
            mean_vals = []
            sem_vals = []
            for lay in range(n_layers):
                vals = beta_data[lay, all_indices]
                mean_vals.append(np.nanmean(vals))
                n_valid = np.sum(~np.isnan(vals))
                sem_vals.append(np.nanstd(vals) / np.sqrt(max(n_valid, 1)))
            profiles_mean[group] = mean_vals
            profiles_sem[group] = sem_vals
        else:
            profiles_mean[group] = [np.nan] * n_layers
            profiles_sem[group] = [np.nan] * n_layers

    return profiles_mean, profiles_sem, n_voxels, member_counts


def extract_parcel_layer_profiles(beta_data, roi_mask, group_to_parcels, n_layers=4):
    """
    Per-parcel layer profiles within each group.

    Returns:
        parcel_profiles: {group: {parcel_name: {'mean': [...], 'sem': [...], 'n_vox': int}}}
    """
    n_vertices_beta = beta_data.shape[-1]
    n = min(len(roi_mask), n_vertices_beta)
    mask_trunc = roi_mask[:n]

    parcel_profiles = {}
    for group, parcels in group_to_parcels.items():
        parcel_profiles[group] = {}
        for p in parcels:
            idx = np.where(mask_trunc == p['roi_idx'])[0]
            if len(idx) > 0:
                mean_vals = []
                sem_vals = []
                for lay in range(n_layers):
                    vals = beta_data[lay, idx]
                    mean_vals.append(np.nanmean(vals))
                    n_valid = np.sum(~np.isnan(vals))
                    sem_vals.append(np.nanstd(vals) / np.sqrt(max(n_valid, 1)))
                parcel_profiles[group][p['parcel']] = {
                    'mean': mean_vals, 'sem': sem_vals, 'n_vox': len(idx)}
            else:
                parcel_profiles[group][p['parcel']] = {
                    'mean': [np.nan] * n_layers, 'sem': [np.nan] * n_layers, 'n_vox': 0}
    return parcel_profiles


# ============================================================
# 4. PLOTTING
# ============================================================

def plot_group_layer_profiles(pred_mean, pred_sem, error_mean, error_sem,
                               group_order, n_voxels, member_counts, save_path,
                               title_suffix='', y_label='Mean Beta'):
    """One subplot per group, showing both Pred and Error layer profiles."""
    # Drop empty groups
    groups = [g for g in group_order if n_voxels[g] > 0]
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(2.7 * n_groups, 3.5),
                             sharey=True)
    if n_groups == 1:
        axes = [axes]

    layers = np.arange(1, N_LAYERS + 1)
    pred_color = '#3498DB'
    error_color = '#E74C3C'

    # Unified y-axis
    all_vals = []
    for g in groups:
        for m, s in zip(pred_mean[g], pred_sem[g]):
            if not np.isnan(m):
                all_vals.extend([m - s, m + s])
        for m, s in zip(error_mean[g], error_sem[g]):
            if not np.isnan(m):
                all_vals.extend([m - s, m + s])
    if all_vals:
        y_min = min(all_vals) * 1.15 if min(all_vals) < 0 else min(all_vals) * 0.85
        y_max = max(all_vals) * 1.15
    else:
        y_min, y_max = -1, 1

    for ax, g in zip(axes, groups):
        ax.errorbar(layers, pred_mean[g], yerr=pred_sem[g],
                    fmt='o-', color=pred_color, linewidth=2, markersize=8,
                    label='Prediction', markerfacecolor='white',
                    markeredgewidth=2, capsize=4, capthick=1.5, elinewidth=1.5)
        ax.errorbar(layers, error_mean[g], yerr=error_sem[g],
                    fmt='s-', color=error_color, linewidth=2, markersize=8,
                    label='Error', markerfacecolor='white',
                    markeredgewidth=2, capsize=4, capthick=1.5, elinewidth=1.5)

        title = '{}\n({} parcels, {} vox)'.format(g, member_counts[g], n_voxels[g])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_xticks(layers)
        ax.set_xlim(0.5, N_LAYERS + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel(y_label, fontsize=11)
    axes[0].legend(loc='upper left', fontsize=9)

    plt.suptitle('Layer Profile by Functional Group{}'.format(title_suffix),
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: {}'.format(save_path))


def compute_global_profile(profiles_mean, n_voxels, group_order, n_layers=4):
    """Compute voxel-weighted global average across all groups."""
    global_mean = np.zeros(n_layers)
    total_vox = 0
    for g in group_order:
        if n_voxels[g] > 0 and not np.isnan(profiles_mean[g][0]):
            w = n_voxels[g]
            global_mean += np.array(profiles_mean[g]) * w
            total_vox += w
    if total_vox > 0:
        global_mean /= total_vox
    return global_mean.tolist()


def plot_combined_groups(pred_mean, error_mean, group_order, n_voxels, save_path,
                         y_label='Mean Beta'):
    """Two panels (Pred / Error) with all groups overlaid using a colormap."""
    groups = [g for g in group_order if n_voxels[g] > 0]
    n_groups = len(groups)

    cmap = plt.cm.tab20
    colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '<', '>', '*']

    # Compute global (voxel-weighted) averages
    global_pred = compute_global_profile(pred_mean, n_voxels, group_order, N_LAYERS)
    global_error = compute_global_profile(error_mean, n_voxels, group_order, N_LAYERS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    layers = np.arange(1, N_LAYERS + 1)

    all_vals = []
    for g in groups:
        all_vals.extend(pred_mean[g])
        all_vals.extend(error_mean[g])
    all_vals += global_pred + global_error
    all_vals = [v for v in all_vals if not np.isnan(v)]
    y_min = min(all_vals) * 1.15 if min(all_vals) < 0 else min(all_vals) * 0.85
    y_max = max(all_vals) * 1.15

    for ax, profiles, global_profile, title in zip(
            axes, [pred_mean, error_mean], [global_pred, global_error],
            ['Prediction (Ahat)', 'Error (E)']):
        for i, g in enumerate(groups):
            ax.plot(layers, profiles[g],
                    marker=markers[i % len(markers)], color=colors[i],
                    linewidth=2.2, markersize=9, label=g,
                    markerfacecolor='white', markeredgewidth=2)
        # Global average line
        ax.plot(layers, global_profile,
                marker='H', color='black', linewidth=3, markersize=10,
                label='Global', linestyle='--',
                markerfacecolor='black', markeredgewidth=0, zorder=10)
        ax.set_xlabel('PredNet Layer', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(layers)
        ax.set_xlim(0.5, N_LAYERS + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=8, ncol=2)

    plt.suptitle('Layer Profile Across Functional Groups',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: {}'.format(save_path))


def plot_zoomed_groups(pred_mean, pred_sem, error_mean, error_sem,
                       group_order, n_voxels, member_counts, save_path,
                       y_label='Mean Beta'):
    """Zoomed-in grid: one subplot per group, each with its own y-axis range."""
    groups = [g for g in group_order if n_voxels[g] > 0]
    n_groups = len(groups)

    n_cols = 4
    n_rows = int(np.ceil(n_groups / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows))
    axes = np.atleast_2d(axes)
    axes_flat = axes.flatten()

    layers = np.arange(1, N_LAYERS + 1)
    pred_color = '#3498DB'
    error_color = '#E74C3C'

    for idx, g in enumerate(groups):
        ax = axes_flat[idx]

        ax.errorbar(layers, pred_mean[g], yerr=pred_sem[g],
                    fmt='o-', color=pred_color, linewidth=2.2, markersize=9,
                    label='Prediction (Ahat)', markerfacecolor='white',
                    markeredgewidth=2, capsize=5, capthick=1.5, elinewidth=1.5)
        ax.errorbar(layers, error_mean[g], yerr=error_sem[g],
                    fmt='s-', color=error_color, linewidth=2.2, markersize=9,
                    label='Error (E)', markerfacecolor='white',
                    markeredgewidth=2, capsize=5, capthick=1.5, elinewidth=1.5)

        # Per-group y-axis range
        vals = []
        for m, s in zip(pred_mean[g], pred_sem[g]):
            if not np.isnan(m):
                vals.extend([m - s, m + s])
        for m, s in zip(error_mean[g], error_sem[g]):
            if not np.isnan(m):
                vals.extend([m - s, m + s])
        if vals:
            margin = (max(vals) - min(vals)) * 0.2
            y_lo = min(vals) - margin
            y_hi = max(vals) + margin
            ax.set_ylim(y_lo, y_hi)

        title = '{}\n({} parcels, {} vox)'.format(g, member_counts[g], n_voxels[g])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('PredNet Layer', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_xticks(layers)
        ax.set_xlim(0.5, N_LAYERS + 0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    # Legend on first subplot
    axes_flat[0].legend(loc='best', fontsize=8)

    # Hide unused subplots
    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle('Layer Profile (Zoomed) by Functional Group',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved: {}'.format(save_path))


def plot_parcel_within_group(pred_parcel_profiles, error_parcel_profiles,
                              group_order, roi_groups, save_dir, y_label='Mean Beta'):
    """
    One figure per functional group, two panels (Ahat / Error),
    with individual parcel lines within that group.
    """
    layers = np.arange(1, N_LAYERS + 1)
    cmap = plt.cm.tab10

    for group in group_order:
        pred_pp = pred_parcel_profiles[group]
        error_pp = error_parcel_profiles[group]
        # Use the ROI_GROUPS order for consistent parcel ordering
        parcel_names = [n for n in roi_groups[group] if n in pred_pp]
        if not parcel_names:
            continue

        n_parcels = len(parcel_names)
        colors = [cmap(i / max(n_parcels - 1, 1)) for i in range(n_parcels)]
        markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '<', '>', '*']

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Collect all values for shared y-axis within this group
        all_vals = []
        for pn in parcel_names:
            for m, s in zip(pred_pp[pn]['mean'], pred_pp[pn]['sem']):
                if not np.isnan(m):
                    all_vals.extend([m - s, m + s])
            for m, s in zip(error_pp[pn]['mean'], error_pp[pn]['sem']):
                if not np.isnan(m):
                    all_vals.extend([m - s, m + s])
        if all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.15
            y_lo = min(all_vals) - margin
            y_hi = max(all_vals) + margin
        else:
            y_lo, y_hi = -0.05, 0.05

        for ax, pp, title in zip(axes,
                                  [pred_pp, error_pp],
                                  ['Prediction (Ahat)', 'Error (E)']):
            for i, pn in enumerate(parcel_names):
                n_vox = pp[pn]['n_vox']
                label = '{} ({})'.format(pn, n_vox)
                ax.errorbar(layers, pp[pn]['mean'], yerr=pp[pn]['sem'],
                            marker=markers[i % len(markers)], color=colors[i],
                            linewidth=2, markersize=8, label=label,
                            markerfacecolor='white', markeredgewidth=2,
                            capsize=4, capthick=1.5, elinewidth=1.5)
            ax.set_xlabel('PredNet Layer', fontsize=11)
            ax.set_ylabel(y_label, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xticks(layers)
            ax.set_xlim(0.5, N_LAYERS + 0.5)
            ax.set_ylim(y_lo, y_hi)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
            ax.legend(loc='best', fontsize=8)

        plt.suptitle('{} — Per-Parcel Layer Profile'.format(group),
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fname = 'parcel_profile_{}.png'.format(group.replace('/', '_').replace(' ', '_'))
        save_path = os.path.join(save_dir, fname)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print('  Saved: {}'.format(save_path))


# ============================================================
# 5. MAIN
# ============================================================

def run_metric(metric, save_dir, nc_dir, ahat_file, e_file, y_label):
    """Run the full pipeline for one metric type (beta or corr)."""
    print('=' * 70)
    print(' Layer Profile by Functional Group (Data-Driven ROIs)')
    print(' Metric: {}'.format(metric))
    print('=' * 70)

    os.makedirs(save_dir, exist_ok=True)

    # Load mask + parcels
    print('\n[1] Loading data-driven ROI mask...')
    roi_mask, parcels = load_datadriven_rois(DATADRIVEN_ROI_MASK, DATADRIVEN_PARCEL_CSV)
    print('    Loaded {} parcels'.format(len(parcels)))

    # Assign to groups
    print('\n[2] Assigning parcels to functional groups...')
    group_to_parcels, unassigned = assign_parcels_to_groups(parcels, ROI_GROUPS)

    print('\n    Group composition:')
    total_assigned = 0
    for g in GROUP_ORDER:
        names = [p['parcel'] for p in group_to_parcels[g]]
        total_assigned += len(names)
        print('      {:14s} ({:2d}): {}'.format(g, len(names), ', '.join(names) if names else '(empty)'))

    if unassigned:
        print('\n    [WARN] Unassigned parcels ({}): {}'.format(
            len(unassigned), ', '.join(unassigned)))

    print('\n    Total assigned: {}/{}'.format(total_assigned, len(parcels)))

    # Load data
    print('\n[3] Loading NC normalized {} files...'.format(metric))
    ahat_path = os.path.join(nc_dir, ahat_file)
    e_path = os.path.join(nc_dir, e_file)
    ahat_data = load_beta_cifti(ahat_path)
    e_data = load_beta_cifti(e_path)
    print('    Ahat: {}'.format(ahat_data.shape))
    print('    E:    {}'.format(e_data.shape))

    # Extract per-group profiles
    print('\n[4] Computing per-group layer profiles...')
    pred_mean, pred_sem, n_vox_pred, member_counts = extract_group_layer_profile(
        ahat_data, roi_mask, group_to_parcels, N_LAYERS)
    error_mean, error_sem, n_vox_error, _ = extract_group_layer_profile(
        e_data, roi_mask, group_to_parcels, N_LAYERS)

    print('\n    Group profiles (Prediction / Error):')
    for g in GROUP_ORDER:
        if n_vox_pred[g] == 0:
            continue
        pm = ['{:+.4f}'.format(v) for v in pred_mean[g]]
        em = ['{:+.4f}'.format(v) for v in error_mean[g]]
        print('      {:14s} (n_vox={:5d}, n_par={:2d})'.format(
            g, n_vox_pred[g], member_counts[g]))
        print('        Pred:  [{}]'.format(', '.join(pm)))
        print('        Error: [{}]'.format(', '.join(em)))

    # Plots
    print('\n[5] Creating plots...')
    plot_group_layer_profiles(
        pred_mean, pred_sem, error_mean, error_sem,
        GROUP_ORDER, n_vox_pred, member_counts,
        os.path.join(save_dir, 'layer_profile_by_group.png'),
        title_suffix=' - NC Normalized ({})'.format(metric),
        y_label=y_label)

    plot_combined_groups(
        pred_mean, error_mean, GROUP_ORDER, n_vox_pred,
        os.path.join(save_dir, 'layer_profile_combined_groups.png'),
        y_label=y_label)

    plot_zoomed_groups(
        pred_mean, pred_sem, error_mean, error_sem,
        GROUP_ORDER, n_vox_pred, member_counts,
        os.path.join(save_dir, 'layer_profile_zoomed_groups.png'),
        y_label=y_label)

    # Per-parcel within-group plots
    print('\n[5b] Computing per-parcel profiles...')
    pred_parcel = extract_parcel_layer_profiles(ahat_data, roi_mask, group_to_parcels, N_LAYERS)
    error_parcel = extract_parcel_layer_profiles(e_data, roi_mask, group_to_parcels, N_LAYERS)

    parcel_dir = os.path.join(save_dir, 'parcel_profiles')
    os.makedirs(parcel_dir, exist_ok=True)
    plot_parcel_within_group(
        pred_parcel, error_parcel, GROUP_ORDER, ROI_GROUPS,
        parcel_dir, y_label=y_label)

    # CSV summary
    print('\n[6] Saving CSV summary...')
    csv_path = os.path.join(save_dir, 'group_layer_profile_summary.csv')
    with open(csv_path, 'w') as f:
        f.write('group,n_parcels,n_voxels,parcels,'
                'pred_L1,pred_L2,pred_L3,pred_L4,'
                'error_L1,error_L2,error_L3,error_L4\n')
        for g in GROUP_ORDER:
            names = ';'.join(p['parcel'] for p in group_to_parcels[g])
            row = [g, str(member_counts[g]), str(n_vox_pred[g]), names]
            row += ['{:.6f}'.format(v) for v in pred_mean[g]]
            row += ['{:.6f}'.format(v) for v in error_mean[g]]
            f.write(','.join(row) + '\n')
    print('    Saved: {}'.format(csv_path))

    # Numpy
    np.save(os.path.join(save_dir, 'group_pred_mean.npy'), pred_mean)
    np.save(os.path.join(save_dir, 'group_pred_sem.npy'), pred_sem)
    np.save(os.path.join(save_dir, 'group_error_mean.npy'), error_mean)
    np.save(os.path.join(save_dir, 'group_error_sem.npy'), error_sem)

    print('\n  Done! Output: {}'.format(save_dir))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Layer Profile by Functional Group (Data-Driven ROIs)')
    parser.add_argument('--metric', type=str, default='both',
                        choices=['beta', 'corr', 'both'],
                        help='Which metric to use: beta, corr, or both')
    args = parser.parse_args()

    if args.metric in ('beta', 'both'):
        print('\n')
        run_metric(
            metric='beta',
            save_dir=SAVE_DIR_BETA,
            nc_dir=NC_BETA_DIR,
            ahat_file=NC_BETA_AHAT_FILE,
            e_file=NC_BETA_E_FILE,
            y_label='Mean Beta',
        )

    if args.metric in ('corr', 'both'):
        print('\n')
        run_metric(
            metric='corr',
            save_dir=SAVE_DIR_CORR,
            nc_dir=NC_CORR_DIR,
            ahat_file=NC_CORR_AHAT_FILE,
            e_file=NC_CORR_E_FILE,
            y_label='Mean r / \u221AISC',
        )

    print('\n' + '=' * 70)
    print(' ALL DONE!')
    print('=' * 70)
