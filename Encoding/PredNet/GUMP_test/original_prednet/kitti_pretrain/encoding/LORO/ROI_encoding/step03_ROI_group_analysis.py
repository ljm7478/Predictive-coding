"""
Step 3: ROI Group Analysis — Group dscalar maps + Correlation layer profiles

Outputs:
  A) Group-mean ROI beta dscalar   (non-ROI = 0)
  B) Group-mean ROI corr dscalar   (non-ROI = 0)
  C) Correlation layer profiles with subject-level t-tests
"""

import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

USE_LOG10 = True

if USE_LOG10:
    BASE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap_log10/layer4'
else:
    BASE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap/layer4'

CIFTI_TEMPLATE = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii'

ROI_MASK_PATH = os.path.join(BASE_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'ROI_mask_datadriven.npy')
PARCEL_CSV_PATH = os.path.join(BASE_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'parcel_stats.csv')
ROI_ENCODING_DIR = os.path.join(BASE_DIR, 'Results', 'Q1', 'ROI_encoding')

SIGNAL_TYPES = ['Ahat', 'E']
N_LAYERS = 4
CORR_FILENAME = 'average_correlations_LORO.npy'

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

# Parcel grouping
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

# =============================================================================
# HELPERS
# =============================================================================

def save_cifti_dscalar(data, template_file, output_file, map_names=None):
    template = nib.load(template_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_maps, n_data = data.shape
    brain_axis = template.header.get_axis(1)
    ci = []
    for name, slc, _ in brain_axis.iter_structures():
        if 'CORTEX' in name:
            ci.extend(range(slc.start, slc.stop))
    cortex_axis = brain_axis[np.array(ci)]
    if n_data != len(ci):
        print(f'    WARNING: data ({n_data}) != cortex ({len(ci)})')
        return False
    if map_names is None:
        map_names = [f'Map_{i+1}' for i in range(n_maps)]
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_axis))
    img = nib.Cifti2Image(data.astype(np.float32), header=hdr)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(img, output_file)
    return True


def load_parcels(csv_path):
    parcels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parcels.append({'roi_idx': int(row['roi_idx']), 'parcel': row['parcel']})
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


def compute_subject_group_vals(roi_mask, roi_indices, group_to_parcels,
                                all_vals, n_layers):
    """Average per-subject values within each group.
    all_vals: (n_subjects, n_layers, n_roi)
    Returns: {group: (n_subjects, n_layers)}, {group: n_vox}
    """
    roi_mask_vals = roi_mask[roi_indices]
    group_vals = {}
    group_nvox = {}
    for group, parcels in group_to_parcels.items():
        parcel_idxs = [p['roi_idx'] for p in parcels]
        vox_in_group = np.isin(roi_mask_vals, parcel_idxs)
        n_vox = vox_in_group.sum()
        group_nvox[group] = n_vox
        if n_vox > 0:
            group_vals[group] = np.nanmean(all_vals[:, :, vox_in_group], axis=2)
        else:
            group_vals[group] = np.full((all_vals.shape[0], n_layers), np.nan)
    return group_vals, group_nvox


def sig_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''


def run_statistics(pred_gv, error_gv, group_order, n_layers):
    onesamp_p_pred = {}; onesamp_p_error = {}; paired_p = {}
    onesamp_t_pred = {}; onesamp_t_error = {}; paired_t = {}
    for g in group_order:
        tp = np.zeros(n_layers); pp = np.ones(n_layers)
        te = np.zeros(n_layers); pe = np.ones(n_layers)
        tpe = np.zeros(n_layers); ppe = np.ones(n_layers)
        for lay in range(n_layers):
            pv = pred_gv[g][:, lay]; ev = error_gv[g][:, lay]
            if np.all(np.isnan(pv)): continue
            t, p = stats.ttest_1samp(pv, 0); tp[lay] = t; pp[lay] = p
            t, p = stats.ttest_1samp(ev, 0); te[lay] = t; pe[lay] = p
            t, p = stats.ttest_rel(pv, ev); tpe[lay] = t; ppe[lay] = p
        onesamp_t_pred[g] = tp; onesamp_p_pred[g] = pp
        onesamp_t_error[g] = te; onesamp_p_error[g] = pe
        paired_t[g] = tpe; paired_p[g] = ppe
    return (onesamp_t_pred, onesamp_p_pred,
            onesamp_t_error, onesamp_p_error,
            paired_t, paired_p)


# =============================================================================
# PLOTTING
# =============================================================================

def plot_layer_profiles(pred_gv, error_gv,
                        onesamp_p_pred, onesamp_p_error, paired_p_vals,
                        group_order, group_nvox, member_counts,
                        save_path, n_layers=4, y_label='Mean r',
                        title_prefix='Correlation'):
    groups = [g for g in group_order if group_nvox[g] > 0]
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(2.8 * n_groups, 4.0), sharey=True)
    if n_groups == 1: axes = [axes]
    layers = np.arange(1, n_layers + 1)
    pred_color = '#3498DB'; error_color = '#E74C3C'

    pred_means = {}; pred_sems = {}; error_means = {}; error_sems = {}
    for g in groups:
        pred_means[g] = np.nanmean(pred_gv[g], axis=0)
        pred_sems[g] = stats.sem(pred_gv[g], axis=0, nan_policy='omit')
        error_means[g] = np.nanmean(error_gv[g], axis=0)
        error_sems[g] = stats.sem(error_gv[g], axis=0, nan_policy='omit')

    all_vals = []
    for g in groups:
        for m, s in zip(pred_means[g], pred_sems[g]):
            all_vals.extend([m - s, m + s])
        for m, s in zip(error_means[g], error_sems[g]):
            all_vals.extend([m - s, m + s])
    y_min = min(all_vals) * 1.25 if min(all_vals) < 0 else min(all_vals) * 0.75
    y_max = max(all_vals) * 1.35

    for ax, g in zip(axes, groups):
        ax.errorbar(layers - 0.05, pred_means[g], yerr=pred_sems[g],
                    fmt='o-', color=pred_color, lw=2, ms=8,
                    label='Prediction ($\\hat{A}$)', mfc='white', mew=2,
                    capsize=4, capthick=1.5, elinewidth=1.5)
        ax.errorbar(layers + 0.05, error_means[g], yerr=error_sems[g],
                    fmt='s-', color=error_color, lw=2, ms=8,
                    label='Error (E)', mfc='white', mew=2,
                    capsize=4, capthick=1.5, elinewidth=1.5)
        star_y = y_max * 0.90
        for lay in range(n_layers):
            x = layers[lay]; dy = (y_max - y_min) * 0.03
            sp = sig_stars(onesamp_p_pred[g][lay])
            if sp:
                ax.text(x - 0.12, pred_means[g][lay] + pred_sems[g][lay] + dy,
                        sp, ha='center', va='bottom', fontsize=7,
                        color=pred_color, fontweight='bold')
            se = sig_stars(onesamp_p_error[g][lay])
            if se:
                ax.text(x + 0.12, error_means[g][lay] + error_sems[g][lay] + dy,
                        se, ha='center', va='bottom', fontsize=7,
                        color=error_color, fontweight='bold')
            sp2 = sig_stars(paired_p_vals[g][lay])
            if sp2:
                ax.text(x, star_y, sp2, ha='center', va='bottom',
                        fontsize=8, color='black', fontweight='bold')

        ax.set_title(f'{g}\n({member_counts[g]} par, {group_nvox[g]} vox)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_xticks(layers); ax.set_xlim(0.5, n_layers + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, ls='--'); ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)

    axes[0].set_ylabel(f'{y_label}\nSEM across subjects', fontsize=10)
    axes[0].legend(loc='upper left', fontsize=8)
    plt.suptitle(f'ROI {title_prefix} Layer Profile (Subject-Level Stats)\n'
                 'colored *: t vs 0 | black *: paired t ($\\hat{A}$ vs E)',
                 fontsize=12, fontweight='bold', y=1.06)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def plot_combined(pred_gv, error_gv, group_order, group_nvox,
                  save_path, n_layers=4, y_label='Mean r'):
    groups = [g for g in group_order if group_nvox[g] > 0]
    n_groups = len(groups)
    cmap = plt.cm.tab20
    colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', 'X', '<', '>', '*']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    layer_x = np.arange(1, n_layers + 1)

    all_vals = []
    for g in groups:
        all_vals.extend(np.nanmean(pred_gv[g], axis=0).tolist())
        all_vals.extend(np.nanmean(error_gv[g], axis=0).tolist())
    y_min = min(all_vals) * 1.15 if min(all_vals) < 0 else min(all_vals) * 0.85
    y_max = max(all_vals) * 1.15

    for ax, gb, title in zip(axes, [pred_gv, error_gv],
                              ['Prediction ($\\hat{A}$)', 'Error (E)']):
        for i, g in enumerate(groups):
            means = np.nanmean(gb[g], axis=0)
            sems = stats.sem(gb[g], axis=0, nan_policy='omit')
            ax.errorbar(layer_x, means, yerr=sems,
                        marker=markers[i % len(markers)], color=colors[i],
                        lw=2.2, ms=9, label=g, mfc='white', mew=2,
                        capsize=3, capthick=1.2, elinewidth=1.2)
        ax.set_xlabel('PredNet Layer', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(layer_x); ax.set_xlim(0.5, n_layers + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.5)
        ax.grid(True, alpha=0.3, ls='--'); ax.set_axisbelow(True)
        ax.legend(loc='best', fontsize=7, ncol=2)
    plt.suptitle('ROI Layer Profile Across Functional Groups',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {save_path}')


def save_stats_csv(pred_gv, error_gv, stat_results, group_order,
                   group_nvox, member_counts, csv_path, n_layers=4):
    (onesamp_t_pred, onesamp_p_pred,
     onesamp_t_error, onesamp_p_error,
     paired_t_vals, paired_p_vals) = stat_results
    with open(csv_path, 'w') as f:
        f.write('group,n_parcels,n_voxels,layer,'
                'pred_mean,pred_sem,pred_t,pred_p,'
                'error_mean,error_sem,error_t,error_p,'
                'paired_t,paired_p\n')
        for g in group_order:
            if group_nvox[g] == 0: continue
            pm = np.nanmean(pred_gv[g], axis=0)
            ps = stats.sem(pred_gv[g], axis=0, nan_policy='omit')
            em = np.nanmean(error_gv[g], axis=0)
            es = stats.sem(error_gv[g], axis=0, nan_policy='omit')
            for lay in range(n_layers):
                f.write(f'{g},{member_counts[g]},{group_nvox[g]},{lay+1},'
                        f'{pm[lay]:.6f},{ps[lay]:.6f},'
                        f'{onesamp_t_pred[g][lay]:.4f},{onesamp_p_pred[g][lay]:.6f},'
                        f'{em[lay]:.6f},{es[lay]:.6f},'
                        f'{onesamp_t_error[g][lay]:.4f},{onesamp_p_error[g][lay]:.6f},'
                        f'{paired_t_vals[g][lay]:.4f},{paired_p_vals[g][lay]:.6f}\n')
    print(f'  Saved: {csv_path}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print(' ROI Group Analysis: dscalar maps + Correlation layer profiles')
    print('=' * 70)

    roi_mask = np.load(ROI_MASK_PATH).astype(np.int32)
    roi_indices = np.where(roi_mask > 0)[0]
    n_cortex = len(roi_mask)
    n_roi = len(roi_indices)
    parcels = load_parcels(PARCEL_CSV_PATH)
    group_to_parcels, unassigned = assign_parcels_to_groups(parcels, ROI_GROUPS)
    member_counts = {g: len(ps) for g, ps in group_to_parcels.items()}

    print(f'  ROI: {n_roi} voxels, {len(parcels)} parcels')
    if unassigned:
        print(f'  [WARN] Unassigned: {", ".join(unassigned)}')

    group_dir = os.path.join(ROI_ENCODING_DIR, 'group')
    os.makedirs(group_dir, exist_ok=True)

    # =====================================================================
    # PART A: Group-mean ROI Beta dscalar
    # =====================================================================
    print('\n' + '=' * 70)
    print(' PART A: Group-mean ROI Beta dscalar maps')
    print('=' * 70)

    for sig_type in SIGNAL_TYPES:
        print(f'\n  [{sig_type}]')
        all_betas_path = os.path.join(ROI_ENCODING_DIR, sig_type, 'all_subjects_beta_ROI.npy')
        all_betas = np.load(all_betas_path)  # (n_subjects, n_layers, n_roi)
        print(f'    Loaded: {all_betas.shape}')

        group_mean_roi = np.mean(all_betas, axis=0)  # (n_layers, n_roi)

        # Expand to whole-brain (non-ROI = 0)
        group_mean_brain = np.zeros((N_LAYERS, n_cortex), dtype=np.float32)
        group_mean_brain[:, roi_indices] = group_mean_roi

        map_names = [f'{sig_type}{i}_beta_ROI' for i in range(N_LAYERS)]
        out_path = os.path.join(group_dir, f'group_mean_beta_ROI_{sig_type}.dscalar.nii')
        save_cifti_dscalar(group_mean_brain, CIFTI_TEMPLATE, out_path, map_names)
        print(f'    Saved: {out_path}')

        # Also save npy
        np.save(os.path.join(group_dir, f'group_mean_beta_ROI_{sig_type}.npy'), group_mean_brain)

        for i in range(N_LAYERS):
            roi_vals = group_mean_roi[i]
            print(f'    Layer {i+1}: mean={roi_vals.mean():.5f}, '
                  f'std={roi_vals.std():.5f}, range=[{roi_vals.min():.5f}, {roi_vals.max():.5f}]')

    # =====================================================================
    # PART B: Group-mean ROI Correlation dscalar + subject-level corr data
    # =====================================================================
    print('\n' + '=' * 70)
    print(' PART B: Group-mean ROI Correlation dscalar maps')
    print('=' * 70)

    corr_data = {}  # {sig_type: (n_subjects, n_layers, n_roi)}

    for sig_type in SIGNAL_TYPES:
        print(f'\n  [{sig_type}]')
        subj_corrs = []
        for subject in SUBJECTS:
            corr_path = Path(BASE_DIR) / sig_type / subject / CORR_FILENAME
            if not corr_path.exists():
                print(f'    [WARN] {subject}: not found')
                continue
            r = np.load(str(corr_path))
            if r.ndim == 2 and r.shape[0] != N_LAYERS:
                r = r.T
            subj_corrs.append(r)
            print(f'    {subject}: {r.shape}')

        stacked = np.stack(subj_corrs, axis=0)  # (n_subjects, n_layers, n_cortex)
        print(f'    Stacked: {stacked.shape}')

        # ROI-masked version
        stacked_roi = stacked[:, :, roi_indices]  # (n_subjects, n_layers, n_roi)
        corr_data[sig_type] = stacked_roi

        group_mean_corr_roi = np.mean(stacked_roi, axis=0)  # (n_layers, n_roi)

        # Expand to brain
        group_mean_corr_brain = np.zeros((N_LAYERS, n_cortex), dtype=np.float32)
        group_mean_corr_brain[:, roi_indices] = group_mean_corr_roi

        map_names = [f'{sig_type}{i}_corr_ROI' for i in range(N_LAYERS)]
        out_path = os.path.join(group_dir, f'group_mean_corr_ROI_{sig_type}.dscalar.nii')
        save_cifti_dscalar(group_mean_corr_brain, CIFTI_TEMPLATE, out_path, map_names)
        print(f'    Saved: {out_path}')

        np.save(os.path.join(group_dir, f'group_mean_corr_ROI_{sig_type}.npy'), group_mean_corr_brain)

        for i in range(N_LAYERS):
            rv = group_mean_corr_roi[i]
            print(f'    Layer {i+1}: mean={rv.mean():.5f}, max={rv.max():.5f}')

    # =====================================================================
    # PART C: Correlation layer profiles with subject-level stats
    # =====================================================================
    print('\n' + '=' * 70)
    print(' PART C: Correlation layer profiles with subject-level stats')
    print('=' * 70)

    corr_profile_dir = os.path.join(ROI_ENCODING_DIR, 'layer_profiles_corr')
    os.makedirs(corr_profile_dir, exist_ok=True)

    pred_corr_gv, group_nvox = compute_subject_group_vals(
        roi_mask, roi_indices, group_to_parcels, corr_data['Ahat'], N_LAYERS)
    error_corr_gv, _ = compute_subject_group_vals(
        roi_mask, roi_indices, group_to_parcels, corr_data['E'], N_LAYERS)

    print('\n  Subject-level group correlations (mean +/- SEM):')
    for g in GROUP_ORDER:
        if group_nvox[g] == 0: continue
        pm = np.nanmean(pred_corr_gv[g], axis=0)
        ps = stats.sem(pred_corr_gv[g], axis=0, nan_policy='omit')
        em = np.nanmean(error_corr_gv[g], axis=0)
        es = stats.sem(error_corr_gv[g], axis=0, nan_policy='omit')
        print(f'    {g:14s} (n_vox={group_nvox[g]:5d})')
        print(f'      Pred:  [{", ".join(f"{m:+.5f}+/-{s:.5f}" for m,s in zip(pm,ps))}]')
        print(f'      Error: [{", ".join(f"{m:+.5f}+/-{s:.5f}" for m,s in zip(em,es))}]')

    print('\n  Statistical tests:')
    stat_results = run_statistics(pred_corr_gv, error_corr_gv, GROUP_ORDER, N_LAYERS)
    (onesamp_t_pred, onesamp_p_pred,
     onesamp_t_error, onesamp_p_error,
     paired_t_vals, paired_p_vals) = stat_results

    print(f'  {"Group":14s} {"L":3s}  {"Pred t":>8s} {"p":>10s}  {"Error t":>8s} {"p":>10s}  {"Paired t":>8s} {"p":>10s}')
    print('  ' + '-' * 80)
    for g in GROUP_ORDER:
        if group_nvox[g] == 0: continue
        for lay in range(N_LAYERS):
            gl = g if lay == 0 else ''
            print(f'  {gl:14s} L{lay+1:d}  '
                  f'{onesamp_t_pred[g][lay]:8.3f} {onesamp_p_pred[g][lay]:10.4f}{sig_stars(onesamp_p_pred[g][lay]):3s}  '
                  f'{onesamp_t_error[g][lay]:8.3f} {onesamp_p_error[g][lay]:10.4f}{sig_stars(onesamp_p_error[g][lay]):3s}  '
                  f'{paired_t_vals[g][lay]:8.3f} {paired_p_vals[g][lay]:10.4f}{sig_stars(paired_p_vals[g][lay]):3s}')

    # Plots
    print('\n  Generating plots...')
    plot_layer_profiles(
        pred_corr_gv, error_corr_gv,
        onesamp_p_pred, onesamp_p_error, paired_p_vals,
        GROUP_ORDER, group_nvox, member_counts,
        os.path.join(corr_profile_dir, 'ROI_corr_layer_profile_by_group.png'),
        n_layers=N_LAYERS, y_label='Mean r (ROI)',
        title_prefix='Correlation')

    plot_combined(
        pred_corr_gv, error_corr_gv,
        GROUP_ORDER, group_nvox,
        os.path.join(corr_profile_dir, 'ROI_corr_layer_profile_combined.png'),
        n_layers=N_LAYERS, y_label='Mean r (ROI)')

    save_stats_csv(pred_corr_gv, error_corr_gv, stat_results,
                   GROUP_ORDER, group_nvox, member_counts,
                   os.path.join(corr_profile_dir, 'ROI_corr_group_stats.csv'),
                   n_layers=N_LAYERS)

    np.save(os.path.join(corr_profile_dir, 'pred_corr_group_vals.npy'), pred_corr_gv)
    np.save(os.path.join(corr_profile_dir, 'error_corr_group_vals.npy'), error_corr_gv)

    print(f'\n{"="*70}')
    print(' ALL DONE!')
    print(f'  Group dscalar: {group_dir}')
    print(f'  Corr profiles: {corr_profile_dir}')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
