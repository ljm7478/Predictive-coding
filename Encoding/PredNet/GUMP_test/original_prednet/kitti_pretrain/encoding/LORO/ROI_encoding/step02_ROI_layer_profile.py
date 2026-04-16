"""
Step 2: ROI Layer Profile with Subject-Level Statistics

Pipeline:
  1. Load ROI-masked betas per subject (from step01)
  2. Compute subject-level ROI-group-averaged betas per layer
  3. Across-subject one-sample t-tests (vs 0) per group/layer
  4. Paired t-tests (Ahat vs E) within each group/layer
  5. Generate layer profiles with SEM (across subjects) + significance markers
"""

import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION
# ============================================================

USE_LOG10 = True

if USE_LOG10:
    BASE_ENCODING_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap_log10/layer4'
else:
    BASE_ENCODING_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap/layer4'

ROI_ENCODING_DIR = os.path.join(BASE_ENCODING_DIR, 'Results', 'Q1', 'ROI_encoding')
ROI_MASK_PATH = os.path.join(BASE_ENCODING_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'ROI_mask_datadriven.npy')
PARCEL_CSV_PATH = os.path.join(BASE_ENCODING_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'parcel_stats.csv')

SAVE_DIR = os.path.join(ROI_ENCODING_DIR, 'layer_profiles')

N_LAYERS = 4

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

# ============================================================
# 2. PARCEL GROUPING
# ============================================================

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
# 3. HELPERS
# ============================================================

def load_parcels(csv_path):
    parcels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parcels.append({
                'roi_idx': int(row['roi_idx']),
                'parcel': row['parcel'],
            })
    return parcels


def assign_parcels_to_groups(parcels, roi_groups):
    parcel_to_group = {}
    for group, names in roi_groups.items():
        for n in names:
            parcel_to_group[n] = group
    group_to_parcels = {g: [] for g in roi_groups}
    unassigned = []
    for p in parcels:
        g = parcel_to_group.get(p['parcel'])
        if g is None:
            unassigned.append(p['parcel'])
        else:
            group_to_parcels[g].append(p)
    return group_to_parcels, unassigned


def sig_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


# ============================================================
# 4. COMPUTE SUBJECT-LEVEL GROUP BETAS
# ============================================================

def compute_subject_group_betas(roi_mask, roi_indices, group_to_parcels,
                                 all_betas, n_layers):
    """
    Args:
        roi_mask: (n_cortex,) int array, parcel index per voxel
        roi_indices: indices where roi_mask > 0
        all_betas: (n_subjects, n_layers, n_roi) ROI-masked betas
    Returns:
        group_betas: {group: (n_subjects, n_layers)} subject-level averages
        group_nvox: {group: int}
    """
    # Build mapping: for each voxel in roi_indices, what is its mask value
    roi_mask_vals = roi_mask[roi_indices]  # parcel index per ROI voxel

    group_betas = {}
    group_nvox = {}

    for group, parcels in group_to_parcels.items():
        # Find which ROI voxels belong to this group
        parcel_idxs = [p['roi_idx'] for p in parcels]
        vox_in_group = np.isin(roi_mask_vals, parcel_idxs)
        n_vox = vox_in_group.sum()
        group_nvox[group] = n_vox

        if n_vox > 0:
            # all_betas[:, :, vox_in_group] -> (n_subjects, n_layers, n_vox_group)
            # mean across voxels -> (n_subjects, n_layers)
            group_betas[group] = np.nanmean(all_betas[:, :, vox_in_group], axis=2)
        else:
            n_subj = all_betas.shape[0]
            group_betas[group] = np.full((n_subj, n_layers), np.nan)

    return group_betas, group_nvox


# ============================================================
# 5. STATISTICAL TESTS
# ============================================================

def run_statistics(pred_group_betas, error_group_betas, group_order, n_layers):
    """
    Returns dicts of arrays:
        onesamp_p_pred[group] = (n_layers,)   one-sample t vs 0
        onesamp_p_error[group] = (n_layers,)
        paired_p[group] = (n_layers,)          paired t: pred vs error
    """
    onesamp_t_pred = {}; onesamp_p_pred = {}
    onesamp_t_error = {}; onesamp_p_error = {}
    paired_t = {}; paired_p = {}

    for g in group_order:
        pred = pred_group_betas[g]    # (n_subjects, n_layers)
        err  = error_group_betas[g]

        t_p = np.zeros(n_layers); p_p = np.ones(n_layers)
        t_e = np.zeros(n_layers); p_e = np.ones(n_layers)
        t_pe = np.zeros(n_layers); p_pe = np.ones(n_layers)

        for lay in range(n_layers):
            pred_vals = pred[:, lay]
            err_vals = err[:, lay]

            if np.all(np.isnan(pred_vals)):
                continue

            # One-sample t-test vs 0
            t_val, p_val = stats.ttest_1samp(pred_vals, 0)
            t_p[lay] = t_val; p_p[lay] = p_val

            t_val, p_val = stats.ttest_1samp(err_vals, 0)
            t_e[lay] = t_val; p_e[lay] = p_val

            # Paired t-test: Ahat vs E
            t_val, p_val = stats.ttest_rel(pred_vals, err_vals)
            t_pe[lay] = t_val; p_pe[lay] = p_val

        onesamp_t_pred[g] = t_p; onesamp_p_pred[g] = p_p
        onesamp_t_error[g] = t_e; onesamp_p_error[g] = p_e
        paired_t[g] = t_pe; paired_p[g] = p_pe

    return (onesamp_t_pred, onesamp_p_pred,
            onesamp_t_error, onesamp_p_error,
            paired_t, paired_p)


# ============================================================
# 6. PLOTTING
# ============================================================

def plot_layer_profiles(pred_group_betas, error_group_betas,
                        onesamp_p_pred, onesamp_p_error, paired_p,
                        group_order, group_nvox, member_counts,
                        save_path, n_layers=4):
    """Layer profile per group with SEM across subjects + significance."""
    groups = [g for g in group_order if group_nvox[g] > 0]
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(2.8 * n_groups, 4.0),
                             sharey=True)
    if n_groups == 1:
        axes = [axes]

    layers = np.arange(1, n_layers + 1)
    pred_color = '#3498DB'
    error_color = '#E74C3C'

    # Compute means and SEMs across subjects
    pred_means = {}; pred_sems = {}
    error_means = {}; error_sems = {}
    for g in groups:
        pm = np.nanmean(pred_group_betas[g], axis=0)
        ps = stats.sem(pred_group_betas[g], axis=0, nan_policy='omit')
        em = np.nanmean(error_group_betas[g], axis=0)
        es = stats.sem(error_group_betas[g], axis=0, nan_policy='omit')
        pred_means[g] = pm; pred_sems[g] = ps
        error_means[g] = em; error_sems[g] = es

    # Unified y-axis
    all_vals = []
    for g in groups:
        for m, s in zip(pred_means[g], pred_sems[g]):
            all_vals.extend([m - s, m + s])
        for m, s in zip(error_means[g], error_sems[g]):
            all_vals.extend([m - s, m + s])
    y_min = min(all_vals) * 1.25 if min(all_vals) < 0 else min(all_vals) * 0.75
    y_max = max(all_vals) * 1.35  # extra room for stars

    for ax, g in zip(axes, groups):
        # Plot Prediction (Ahat)
        ax.errorbar(layers - 0.05, pred_means[g], yerr=pred_sems[g],
                    fmt='o-', color=pred_color, linewidth=2, markersize=8,
                    label='Prediction ($\\hat{A}$)', markerfacecolor='white',
                    markeredgewidth=2, capsize=4, capthick=1.5, elinewidth=1.5)
        # Plot Error (E)
        ax.errorbar(layers + 0.05, error_means[g], yerr=error_sems[g],
                    fmt='s-', color=error_color, linewidth=2, markersize=8,
                    label='Error (E)', markerfacecolor='white',
                    markeredgewidth=2, capsize=4, capthick=1.5, elinewidth=1.5)

        # Significance markers
        star_y = y_max * 0.90
        for lay in range(n_layers):
            x = layers[lay]
            # One-sample stars above each point
            sp = sig_stars(onesamp_p_pred[g][lay])
            if sp:
                ax.text(x - 0.12, pred_means[g][lay] + pred_sems[g][lay] + (y_max - y_min) * 0.03,
                        sp, ha='center', va='bottom', fontsize=7,
                        color=pred_color, fontweight='bold')
            se = sig_stars(onesamp_p_error[g][lay])
            if se:
                ax.text(x + 0.12, error_means[g][lay] + error_sems[g][lay] + (y_max - y_min) * 0.03,
                        se, ha='center', va='bottom', fontsize=7,
                        color=error_color, fontweight='bold')

            # Paired t-test marker between the two
            sp_paired = sig_stars(paired_p[g][lay])
            if sp_paired:
                ax.text(x, star_y, sp_paired, ha='center', va='bottom',
                        fontsize=8, color='black', fontweight='bold')

        title = f'{g}\n({member_counts[g]} parcels, {group_nvox[g]} vox)'
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_xticks(layers)
        ax.set_xlim(0.5, n_layers + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel('Mean Beta (ROI PC1)\nSEM across subjects', fontsize=10)
    axes[0].legend(loc='upper left', fontsize=8)

    plt.suptitle('ROI-Selective Layer Profile (Subject-Level Stats)\n'
                 'colored *: one-sample t vs 0 | black *: paired t ($\\hat{A}$ vs E)',
                 fontsize=12, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_combined(pred_group_betas, error_group_betas,
                  group_order, group_nvox, save_path, n_layers=4):
    """Two panels (Pred/Error) with all groups overlaid."""
    groups = [g for g in group_order if group_nvox[g] > 0]
    n_groups = len(groups)

    cmap = plt.cm.tab20
    colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '<', '>', '*']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    layer_x = np.arange(1, n_layers + 1)

    all_vals = []
    for g in groups:
        all_vals.extend(np.nanmean(pred_group_betas[g], axis=0).tolist())
        all_vals.extend(np.nanmean(error_group_betas[g], axis=0).tolist())
    y_min = min(all_vals) * 1.15 if min(all_vals) < 0 else min(all_vals) * 0.85
    y_max = max(all_vals) * 1.15

    for ax, gb, title in zip(axes,
                              [pred_group_betas, error_group_betas],
                              ['Prediction ($\\hat{A}$)', 'Error (E)']):
        for i, g in enumerate(groups):
            means = np.nanmean(gb[g], axis=0)
            sems = stats.sem(gb[g], axis=0, nan_policy='omit')
            ax.errorbar(layer_x, means, yerr=sems,
                        marker=markers[i % len(markers)], color=colors[i],
                        linewidth=2.2, markersize=9, label=g,
                        markerfacecolor='white', markeredgewidth=2,
                        capsize=3, capthick=1.2, elinewidth=1.2)
        ax.set_xlabel('PredNet Layer', fontsize=11)
        ax.set_ylabel('Mean Beta (ROI PC1)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(layer_x)
        ax.set_xlim(0.5, n_layers + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=7, ncol=2)

    plt.suptitle('ROI-Selective Layer Profile (Subject-Level SEM)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print('=' * 70)
    print(' ROI Layer Profile with Subject-Level Statistics')
    print('=' * 70)

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load ROI mask + parcels
    print('\n[1] Loading ROI mask + parcels...')
    roi_mask = np.load(ROI_MASK_PATH).astype(np.int32)
    roi_indices = np.where(roi_mask > 0)[0]
    parcels = load_parcels(PARCEL_CSV_PATH)
    print(f'    {len(parcels)} parcels, {len(roi_indices)} ROI voxels')

    # Assign to groups
    group_to_parcels, unassigned = assign_parcels_to_groups(parcels, ROI_GROUPS)
    member_counts = {g: len(ps) for g, ps in group_to_parcels.items()}

    print('\n    Group composition:')
    for g in GROUP_ORDER:
        names = [p['parcel'] for p in group_to_parcels[g]]
        print(f'      {g:14s} ({len(names):2d}): {", ".join(names) if names else "(empty)"}')
    if unassigned:
        print(f'\n    [WARN] Unassigned: {", ".join(unassigned)}')

    # Load subject-level ROI betas
    print('\n[2] Loading subject-level ROI betas...')
    pred_path = os.path.join(ROI_ENCODING_DIR, 'Ahat', 'all_subjects_beta_ROI.npy')
    error_path = os.path.join(ROI_ENCODING_DIR, 'E', 'all_subjects_beta_ROI.npy')

    pred_all = np.load(pred_path)   # (n_subjects, n_layers, n_roi)
    error_all = np.load(error_path)
    print(f'    Ahat: {pred_all.shape}')
    print(f'    E:    {error_all.shape}')

    # Compute subject-level group-averaged betas
    print('\n[3] Computing subject-level group averages...')
    pred_group_betas, group_nvox = compute_subject_group_betas(
        roi_mask, roi_indices, group_to_parcels, pred_all, N_LAYERS)
    error_group_betas, _ = compute_subject_group_betas(
        roi_mask, roi_indices, group_to_parcels, error_all, N_LAYERS)

    n_subjects = pred_all.shape[0]

    print('\n    Subject-level group betas (mean +/- SEM across subjects):')
    for g in GROUP_ORDER:
        if group_nvox[g] == 0:
            continue
        pm = np.nanmean(pred_group_betas[g], axis=0)
        ps = stats.sem(pred_group_betas[g], axis=0, nan_policy='omit')
        em = np.nanmean(error_group_betas[g], axis=0)
        es = stats.sem(error_group_betas[g], axis=0, nan_policy='omit')
        print(f'      {g:14s} (n_vox={group_nvox[g]:5d})')
        pm_str = ', '.join(f'{m:+.4f}+/-{s:.4f}' for m, s in zip(pm, ps))
        em_str = ', '.join(f'{m:+.4f}+/-{s:.4f}' for m, s in zip(em, es))
        print(f'        Pred:  [{pm_str}]')
        print(f'        Error: [{em_str}]')

    # Statistical tests
    print('\n[4] Running statistical tests...')
    (onesamp_t_pred, onesamp_p_pred,
     onesamp_t_error, onesamp_p_error,
     paired_t_vals, paired_p_vals) = run_statistics(
        pred_group_betas, error_group_betas, GROUP_ORDER, N_LAYERS)

    print('\n    One-sample t-test vs 0:')
    print(f'    {"Group":14s} {"Lay":3s}  {"Pred t":>8s} {"p":>10s}  {"Error t":>8s} {"p":>10s}  {"Paired t":>8s} {"p":>10s}')
    print('    ' + '-' * 80)
    for g in GROUP_ORDER:
        if group_nvox[g] == 0:
            continue
        for lay in range(N_LAYERS):
            g_label = g if lay == 0 else ''
            print(f'    {g_label:14s} L{lay+1:d}  '
                  f'{onesamp_t_pred[g][lay]:8.3f} {onesamp_p_pred[g][lay]:10.4f}{sig_stars(onesamp_p_pred[g][lay]):3s}  '
                  f'{onesamp_t_error[g][lay]:8.3f} {onesamp_p_error[g][lay]:10.4f}{sig_stars(onesamp_p_error[g][lay]):3s}  '
                  f'{paired_t_vals[g][lay]:8.3f} {paired_p_vals[g][lay]:10.4f}{sig_stars(paired_p_vals[g][lay]):3s}')

    # Plots
    print('\n[5] Generating plots...')
    plot_layer_profiles(
        pred_group_betas, error_group_betas,
        onesamp_p_pred, onesamp_p_error, paired_p_vals,
        GROUP_ORDER, group_nvox, member_counts,
        os.path.join(SAVE_DIR, 'ROI_layer_profile_by_group.png'),
        n_layers=N_LAYERS)

    plot_combined(
        pred_group_betas, error_group_betas,
        GROUP_ORDER, group_nvox,
        os.path.join(SAVE_DIR, 'ROI_layer_profile_combined.png'),
        n_layers=N_LAYERS)

    # Save CSV
    print('\n[6] Saving CSV...')
    csv_path = os.path.join(SAVE_DIR, 'ROI_group_stats.csv')
    with open(csv_path, 'w') as f:
        f.write('group,n_parcels,n_voxels,layer,'
                'pred_mean,pred_sem,pred_t,pred_p,'
                'error_mean,error_sem,error_t,error_p,'
                'paired_t,paired_p\n')
        for g in GROUP_ORDER:
            if group_nvox[g] == 0:
                continue
            pm = np.nanmean(pred_group_betas[g], axis=0)
            ps = stats.sem(pred_group_betas[g], axis=0, nan_policy='omit')
            em = np.nanmean(error_group_betas[g], axis=0)
            es = stats.sem(error_group_betas[g], axis=0, nan_policy='omit')
            for lay in range(N_LAYERS):
                f.write(f'{g},{member_counts[g]},{group_nvox[g]},{lay+1},'
                        f'{pm[lay]:.6f},{ps[lay]:.6f},'
                        f'{onesamp_t_pred[g][lay]:.4f},{onesamp_p_pred[g][lay]:.6f},'
                        f'{em[lay]:.6f},{es[lay]:.6f},'
                        f'{onesamp_t_error[g][lay]:.4f},{onesamp_p_error[g][lay]:.6f},'
                        f'{paired_t_vals[g][lay]:.4f},{paired_p_vals[g][lay]:.6f}\n')
    print(f'    Saved: {csv_path}')

    # Save numpy
    np.save(os.path.join(SAVE_DIR, 'pred_group_betas.npy'), pred_group_betas)
    np.save(os.path.join(SAVE_DIR, 'error_group_betas.npy'), error_group_betas)

    print(f'\n{"="*70}')
    print(f' Done! Output: {SAVE_DIR}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
