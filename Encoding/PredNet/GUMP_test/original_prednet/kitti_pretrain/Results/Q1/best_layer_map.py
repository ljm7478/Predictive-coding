"""
Best-Layer Map (Wen et al. 2018 style) with SELECTIVITY FILTERING

For each voxel/parcel, determines the 'best layer' = argmax(r_L1, r_L2, r_L3, r_L4).
Also computes a SELECTIVITY INDEX = (max(r) - mean(r_others)) / (max(r) + eps)
which quantifies how strongly one layer stands out.

Voxels/parcels with low selectivity are flagged as 'undetermined' (value 0)
so that only confidently layer-preferring regions are colored.

Outputs per signal type (Ahat, E):
  - best_layer_<ST>_parcelwise.dscalar.nii       (no filter)
  - best_layer_<ST>_parcelwise_thresh.dscalar.nii (selectivity >= threshold)
  - best_layer_<ST>_voxelwise.dscalar.nii
  - best_layer_<ST>_voxelwise_thresh.dscalar.nii
  - selectivity_<ST>_parcelwise.dscalar.nii       (continuous)
  - selectivity_<ST>_voxelwise.dscalar.nii
Plus:
  - best_layer_summary.csv
  - best_layer_bar.png
  - selectivity_histograms.png
"""

import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURATION
# ============================================================

USE_LOG10 = True

if USE_LOG10:
    BASE_DIR = ('/combinelab/03_user/jungmin/01_project/01_Encoding/'
                '01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/'
                'encoding_analysis/original_prednet/kitti_pretrain_result/'
                'LORO_PCAcap_log10/layer4')
else:
    BASE_DIR = ('/combinelab/03_user/jungmin/01_project/01_Encoding/'
                '01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/'
                'encoding_analysis/original_prednet/kitti_pretrain_result/'
                'LORO_PCAcap/layer4')

ROI_MASK_PATH = os.path.join(
    BASE_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'ROI_mask_datadriven.npy'
)
PARCEL_CSV_PATH = os.path.join(
    BASE_DIR, 'Results', 'Q1', 'sig_mask_datadriven', 'parcel_stats.csv'
)

CIFTI_TEMPLATE = ('/combinelab2/03_user/jungmin/02_data/01_Gump/'
                  '01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/'
                  'sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/'
                  'ses-movie_task-movie_run-1/'
                  'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii')

CORR_FILENAME = 'average_correlations_LORO.npy'
SIGNAL_TYPES  = ['Ahat', 'E']
N_LAYERS      = 4

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

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

# ---------------- SELECTIVITY SETTINGS ----------------
# Selectivity = (max_r - mean_of_other_layers) / (|max_r| + EPS)
# Range roughly [0, 1+]: higher = more layer-selective
#
# We auto-set thresholds at the MEDIAN of each signal's selectivity
# distribution (so ~50% of ROI voxels pass). Override by setting these
# to fixed values if you prefer:
#     VOXEL_SELECTIVITY_THRESHOLD = 0.10
#     PARCEL_SELECTIVITY_THRESHOLD = 0.10
VOXEL_SELECTIVITY_THRESHOLD  = None   # None = use median per signal
PARCEL_SELECTIVITY_THRESHOLD = None   # None = use median per signal
EPS = 1e-6

OUTPUT_DIR = os.path.join(BASE_DIR, 'Results', 'Q1', 'best_layer_map')


# ============================================================
# 2. HELPERS
# ============================================================

def load_r_all_subjects(base_dir, signal_type, subjects, fn):
    r_list = []
    for subj in subjects:
        p = Path(base_dir) / signal_type / subj / fn
        if not p.exists():
            continue
        r = np.load(str(p))
        if r.ndim == 2 and r.shape[0] != N_LAYERS:
            r = r.T
        r_list.append(r)
    if not r_list:
        return None
    return np.stack(r_list, axis=0)


def load_parcels(csv_path):
    parcels = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parcels.append({
                'roi_idx': int(row['roi_idx']),
                'parcel':  row['parcel'],
                'r_max':   float(row['r_max']),
            })
    return parcels


def save_dscalar(data, template_path, output_path, map_name='best_layer'):
    template   = nib.load(template_path)
    brain_axis = template.header.get_axis(1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_data = data.shape[1]
    n_tmpl = len(brain_axis)
    if n_tmpl != n_data:
        ci = []
        for name, slc, _ in brain_axis.iter_structures():
            if 'CORTEX' in name:
                s = slc.start if slc.start is not None else 0
                e = slc.stop  if slc.stop  is not None else n_tmpl
                ci.extend(range(s, e))
        brain_axis = brain_axis[np.array(ci)]
    scalar_axis = nib.cifti2.ScalarAxis([map_name])
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, brain_axis))
    img = nib.Cifti2Image(data.astype(np.float32), header=hdr)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(img, output_path)


def assign_parcels_to_groups(parcels, roi_groups):
    parcel_to_group = {}
    for group, names in roi_groups.items():
        for n in names:
            parcel_to_group[n] = group
    return parcel_to_group


def compute_selectivity(r_per_layer_axis0):
    """
    r_per_layer_axis0: shape (n_layers, ...) OR (n_layers,)

    Returns selectivity with shape (...) OR scalar.
    selectivity = (max - mean_of_others) / (|max| + EPS)

    'mean of others' = mean of the non-max layers.
    """
    r = r_per_layer_axis0
    n_layers = r.shape[0]

    max_vals  = np.max(r, axis=0)
    sum_vals  = np.sum(r, axis=0)
    mean_others = (sum_vals - max_vals) / max(n_layers - 1, 1)

    sel = (max_vals - mean_others) / (np.abs(max_vals) + EPS)
    return sel


# ============================================================
# 3. MAIN
# ============================================================

def main():
    print('=' * 70)
    print(' Best-Layer Map with Selectivity Filtering')
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load ROI mask + parcels
    # ------------------------------------------------------------------
    print('\n[1] Loading ROI mask + parcels...')
    roi_mask = np.load(ROI_MASK_PATH).astype(np.int32)
    parcels  = load_parcels(PARCEL_CSV_PATH)
    parcel_to_group = assign_parcels_to_groups(parcels, ROI_GROUPS)
    n_vox = len(roi_mask)
    roi_vox_mask = roi_mask > 0
    print('    %d parcels, %d voxels (%d in ROI)' %
          (len(parcels), n_vox, roi_vox_mask.sum()))

    # ------------------------------------------------------------------
    # Load correlations
    # ------------------------------------------------------------------
    print('\n[2] Loading correlations...')
    r_stacks = {}
    for st in SIGNAL_TYPES:
        rs = load_r_all_subjects(BASE_DIR, st, SUBJECTS, CORR_FILENAME)
        if rs is None:
            raise RuntimeError('No correlations found for %s' % st)
        r_stacks[st] = rs
        print('    %s: %s' % (st, rs.shape))

    # Group-mean r per layer (voxel-wise)
    r_mean_voxel = {st: np.nanmean(r_stacks[st], axis=0) for st in SIGNAL_TYPES}

    # ------------------------------------------------------------------
    # Parcel-level best layer + selectivity
    # ------------------------------------------------------------------
    print('\n[3] Computing parcel-level best layer + selectivity...')

    parcel_best_layer = {st: {} for st in SIGNAL_TYPES}
    parcel_layer_r    = {st: {} for st in SIGNAL_TYPES}
    parcel_selectivity = {st: {} for st in SIGNAL_TYPES}

    for p in parcels:
        vtx = np.where(roi_mask == p['roi_idx'])[0]
        if len(vtx) == 0:
            continue
        for st in SIGNAL_TYPES:
            r_per_layer = np.array([
                np.nanmean(r_mean_voxel[st][lay, vtx]) for lay in range(N_LAYERS)
            ])
            parcel_layer_r[st][p['parcel']]    = r_per_layer
            parcel_best_layer[st][p['parcel']] = int(np.argmax(r_per_layer)) + 1
            parcel_selectivity[st][p['parcel']] = float(compute_selectivity(r_per_layer))

    # ------------------------------------------------------------------
    # Voxel-level best layer + selectivity
    # ------------------------------------------------------------------
    print('\n[4] Computing voxel-level best layer + selectivity...')

    voxel_best_layer = {}
    voxel_selectivity = {}
    for st in SIGNAL_TYPES:
        rm = r_mean_voxel[st]               # (n_layers, n_vox)
        bl = np.argmax(rm, axis=0) + 1      # 1..4
        bl_masked = np.where(roi_vox_mask, bl, 0).astype(np.int32)
        voxel_best_layer[st] = bl_masked

        sel = compute_selectivity(rm)       # (n_vox,)
        sel_masked = np.where(roi_vox_mask, sel, 0.0).astype(np.float32)
        voxel_selectivity[st] = sel_masked

    # ------------------------------------------------------------------
    # Thresholds (auto-compute if not set)
    # ------------------------------------------------------------------
    vox_thresh = {}
    par_thresh = {}
    for st in SIGNAL_TYPES:
        vox_vals = voxel_selectivity[st][roi_vox_mask]
        par_vals = np.array(list(parcel_selectivity[st].values()))

        if VOXEL_SELECTIVITY_THRESHOLD is None:
            vox_thresh[st] = float(np.median(vox_vals))
        else:
            vox_thresh[st] = float(VOXEL_SELECTIVITY_THRESHOLD)

        if PARCEL_SELECTIVITY_THRESHOLD is None:
            par_thresh[st] = float(np.median(par_vals))
        else:
            par_thresh[st] = float(PARCEL_SELECTIVITY_THRESHOLD)

        print('    %s: voxel thresh=%.4f (median), parcel thresh=%.4f (median)'
              % (st, vox_thresh[st], par_thresh[st]))

    # ------------------------------------------------------------------
    # Build parcel-wise dscalar + thresholded version
    # ------------------------------------------------------------------
    print('\n[5] Building parcel-wise dscalar maps...')
    parcel_map        = {st: np.zeros(n_vox, dtype=np.int32) for st in SIGNAL_TYPES}
    parcel_map_thresh = {st: np.zeros(n_vox, dtype=np.int32) for st in SIGNAL_TYPES}
    parcel_sel_map    = {st: np.zeros(n_vox, dtype=np.float32) for st in SIGNAL_TYPES}

    for p in parcels:
        vtx = np.where(roi_mask == p['roi_idx'])[0]
        for st in SIGNAL_TYPES:
            bl  = parcel_best_layer[st].get(p['parcel'], 0)
            sel = parcel_selectivity[st].get(p['parcel'], 0.0)
            parcel_map[st][vtx]     = bl
            parcel_sel_map[st][vtx] = sel
            if sel >= par_thresh[st]:
                parcel_map_thresh[st][vtx] = bl

    # ------------------------------------------------------------------
    # Voxel-wise thresholded version
    # ------------------------------------------------------------------
    voxel_best_layer_thresh = {}
    for st in SIGNAL_TYPES:
        bl = voxel_best_layer[st].copy()
        pass_mask = voxel_selectivity[st] >= vox_thresh[st]
        bl[~pass_mask] = 0
        voxel_best_layer_thresh[st] = bl

    # ------------------------------------------------------------------
    # Save all CIFTI dscalars
    # ------------------------------------------------------------------
    print('\n[6] Saving CIFTI dscalars...')
    for st in SIGNAL_TYPES:
        # Parcel-wise
        save_dscalar(parcel_map[st].astype(np.float32), CIFTI_TEMPLATE,
                     os.path.join(OUTPUT_DIR, 'best_layer_%s_parcelwise.dscalar.nii' % st),
                     map_name='best_layer_%s_parcel' % st)
        save_dscalar(parcel_map_thresh[st].astype(np.float32), CIFTI_TEMPLATE,
                     os.path.join(OUTPUT_DIR, 'best_layer_%s_parcelwise_thresh.dscalar.nii' % st),
                     map_name='best_layer_%s_parcel_thresh' % st)
        save_dscalar(parcel_sel_map[st], CIFTI_TEMPLATE,
                     os.path.join(OUTPUT_DIR, 'selectivity_%s_parcelwise.dscalar.nii' % st),
                     map_name='selectivity_%s_parcel' % st)

        # Voxel-wise
        save_dscalar(voxel_best_layer[st].astype(np.float32), CIFTI_TEMPLATE,
                     os.path.join(OUTPUT_DIR, 'best_layer_%s_voxelwise.dscalar.nii' % st),
                     map_name='best_layer_%s_voxel' % st)
        save_dscalar(voxel_best_layer_thresh[st].astype(np.float32), CIFTI_TEMPLATE,
                     os.path.join(OUTPUT_DIR, 'best_layer_%s_voxelwise_thresh.dscalar.nii' % st),
                     map_name='best_layer_%s_voxel_thresh' % st)
        save_dscalar(voxel_selectivity[st], CIFTI_TEMPLATE,
                     os.path.join(OUTPUT_DIR, 'selectivity_%s_voxelwise.dscalar.nii' % st),
                     map_name='selectivity_%s_voxel' % st)

        print('    Saved: all maps for %s' % st)

    # ------------------------------------------------------------------
    # CSV summary (parcel-level)
    # ------------------------------------------------------------------
    print('\n[7] Saving CSV summary...')
    csv_out = os.path.join(OUTPUT_DIR, 'best_layer_summary.csv')
    with open(csv_out, 'w') as f:
        f.write('roi_idx,parcel,group,'
                'ahat_r_L1,ahat_r_L2,ahat_r_L3,ahat_r_L4,ahat_best_layer,ahat_selectivity,ahat_passed_thresh,'
                'e_r_L1,e_r_L2,e_r_L3,e_r_L4,e_best_layer,e_selectivity,e_passed_thresh\n')
        for p in sorted(parcels, key=lambda x: x['r_max'], reverse=True):
            grp = parcel_to_group.get(p['parcel'], 'Unassigned')
            if p['parcel'] not in parcel_layer_r['Ahat']:
                continue
            ar  = parcel_layer_r['Ahat'][p['parcel']]
            er  = parcel_layer_r['E'][p['parcel']]
            abl = parcel_best_layer['Ahat'][p['parcel']]
            ebl = parcel_best_layer['E'][p['parcel']]
            asel = parcel_selectivity['Ahat'][p['parcel']]
            esel = parcel_selectivity['E'][p['parcel']]
            apass = int(asel >= par_thresh['Ahat'])
            epass = int(esel >= par_thresh['E'])
            f.write('%d,%s,%s,'
                    '%.5f,%.5f,%.5f,%.5f,%d,%.4f,%d,'
                    '%.5f,%.5f,%.5f,%.5f,%d,%.4f,%d\n' %
                    (p['roi_idx'], p['parcel'], grp,
                     ar[0], ar[1], ar[2], ar[3], abl, asel, apass,
                     er[0], er[1], er[2], er[3], ebl, esel, epass))
    print('    Saved: %s' % os.path.basename(csv_out))

    # ------------------------------------------------------------------
    # Selectivity histograms
    # ------------------------------------------------------------------
    print('\n[8] Plotting selectivity histograms...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, st in enumerate(SIGNAL_TYPES):
        # Voxel-level histogram
        ax = axes[row, 0]
        vox_vals = voxel_selectivity[st][roi_vox_mask]
        ax.hist(vox_vals, bins=60, color='#5499C7', edgecolor='white')
        ax.axvline(vox_thresh[st], color='#C0392B', lw=2, ls='--',
                   label='threshold = %.3f' % vox_thresh[st])
        ax.set_xlabel('Voxel selectivity')
        ax.set_ylabel('Count')
        ax.set_title('%s - voxel-level selectivity' % st, fontweight='bold')
        ax.legend()

        # Parcel-level histogram
        ax = axes[row, 1]
        par_vals = np.array(list(parcel_selectivity[st].values()))
        ax.hist(par_vals, bins=20, color='#E67E22', edgecolor='white')
        ax.axvline(par_thresh[st], color='#C0392B', lw=2, ls='--',
                   label='threshold = %.3f' % par_thresh[st])
        ax.set_xlabel('Parcel selectivity')
        ax.set_ylabel('Count')
        ax.set_title('%s - parcel-level selectivity' % st, fontweight='bold')
        ax.legend()

    plt.tight_layout()
    fig_out = os.path.join(OUTPUT_DIR, 'selectivity_histograms.png')
    plt.savefig(fig_out, dpi=200, bbox_inches='tight')
    plt.close()
    print('    Saved: %s' % os.path.basename(fig_out))

    # ------------------------------------------------------------------
    # Bar plot (parcel-level best layer, hierarchy-ordered)
    # ------------------------------------------------------------------
    print('\n[9] Plotting best-layer bar chart...')

    ordered_parcels = []
    for g in GROUP_ORDER:
        for name in ROI_GROUPS[g]:
            if name in parcel_best_layer['Ahat']:
                ordered_parcels.append((g, name))

    n_parcels = len(ordered_parcels)
    layer_colors = {1: '#2E7DB8', 2: '#6FA87B', 3: '#E8A33D', 4: '#C23B3B'}
    faded_color = '#DDDDDD'

    fig, axes = plt.subplots(2, 1, figsize=(max(12, n_parcels * 0.22), 7),
                             sharex=True)

    for ax, st, title in zip(axes, ['Ahat', 'E'],
                              ['Prediction ($\\hat{A}$)', 'Error (E)']):
        xs = np.arange(n_parcels)
        best_layers = [parcel_best_layer[st][name] for _, name in ordered_parcels]
        sels        = [parcel_selectivity[st][name] for _, name in ordered_parcels]
        passed      = [s >= par_thresh[st] for s in sels]

        # Bars: colored if passed, grey if not
        cols = [layer_colors[bl] if p else faded_color
                for bl, p in zip(best_layers, passed)]
        edges = ['black' if p else '#AAAAAA' for p in passed]

        ax.bar(xs, best_layers, color=cols, edgecolor=edges, linewidth=0.4)
        ax.set_ylim(0, 4.5)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylabel('Best Layer')
        ax.set_title('%s  (grey = below selectivity threshold %.3f)' %
                     (title, par_thresh[st]),
                     fontsize=11, fontweight='bold', loc='left')

        # Group boundaries
        cur_g = None
        for i, (g, _) in enumerate(ordered_parcels):
            if g != cur_g:
                if cur_g is not None:
                    ax.axvline(i - 0.5, color='gray', lw=0.4, alpha=0.6)
                cur_g = g

        # Group labels (top subplot only)
        if st == 'Ahat':
            cur_g = None
            start = 0
            for i, (g, _) in enumerate(ordered_parcels):
                if g != cur_g:
                    if cur_g is not None:
                        mid = (start + i - 1) / 2
                        ax.text(mid, 4.7, cur_g, ha='center', va='bottom',
                                fontsize=8, fontweight='bold')
                    cur_g = g
                    start = i
            mid = (start + n_parcels - 1) / 2
            ax.text(mid, 4.7, cur_g, ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    axes[1].set_xticks(np.arange(n_parcels))
    axes[1].set_xticklabels([name for _, name in ordered_parcels],
                             rotation=90, fontsize=7)
    axes[1].set_xlabel('Parcels (grouped by hierarchy)')

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=layer_colors[l], label='L%d best' % l)
               for l in [1, 2, 3, 4]]
    handles.append(mpatches.Patch(color=faded_color, label='non-selective'))
    axes[0].legend(handles=handles, loc='upper right', fontsize=9, ncol=5)

    plt.tight_layout()
    fig_out = os.path.join(OUTPUT_DIR, 'best_layer_bar.png')
    plt.savefig(fig_out, dpi=200, bbox_inches='tight')
    plt.close()
    print('    Saved: %s' % os.path.basename(fig_out))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print(' SUMMARY')
    print('=' * 70)
    for st in SIGNAL_TYPES:
        counts_all    = {1: 0, 2: 0, 3: 0, 4: 0}
        counts_thresh = {1: 0, 2: 0, 3: 0, 4: 0}
        total_pass = 0
        for _, name in ordered_parcels:
            bl  = parcel_best_layer[st][name]
            sel = parcel_selectivity[st][name]
            counts_all[bl] += 1
            if sel >= par_thresh[st]:
                counts_thresh[bl] += 1
                total_pass += 1
        total = sum(counts_all.values())
        print('\n  %s parcel-level best-layer distribution:' % st)
        print('    (all parcels / thresholded)')
        for l in [1, 2, 3, 4]:
            pct_all = 100. * counts_all[l] / total if total else 0
            pct_th  = 100. * counts_thresh[l] / total_pass if total_pass else 0
            print('    L%d: %2d (%.1f%%) / %2d (%.1f%%)' %
                  (l, counts_all[l], pct_all, counts_thresh[l], pct_th))
        print('    threshold-passing parcels: %d/%d (%.1f%%)' %
              (total_pass, total, 100. * total_pass / max(total, 1)))

    print('\n  Output: %s' % OUTPUT_DIR)
    print('=' * 70)


if __name__ == '__main__':
    main()