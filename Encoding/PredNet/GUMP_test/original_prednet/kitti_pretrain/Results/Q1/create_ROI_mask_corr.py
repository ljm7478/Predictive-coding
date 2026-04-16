"""
Data-Driven ROI Mask (Method A)

Criteria:
  1. FDR-corrected significance (p < 0.05) on encoding accuracy (r)
  2. Effect size threshold: r >= 0.030 (top quartile)
  3. Reliable stimulus-driven response: ISC >= 0.05

Pipeline:
  - Load per-subject correlations for Ahat and E
  - Union FDR mask (sig in Ahat OR E)
  - Map sig voxels -> Glasser parcels -> merge bilateral
  - Filter parcels by r_max >= 0.030 AND mean_ISC >= 0.05
  - Save ROI mask + parcel list
"""

import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set to True to use log10-transformed encoding output directory
USE_LOG10 = True

if USE_LOG10:
    BASE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4'
else:
    BASE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500/layer4'

# ISC is fMRI-only — reuse from LORO_PCAcap_log10 (computed once)
ISC_PATH = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap_log10/layer4/divergence_analysis/step0_noise_ceiling/isc_map.npy'

GLASSER_LABEL_PATH = '/combinelab2/03_user/jungmin/02_data/17_MMP_label/mmp/Q1-Q6_RelatedParcellation210.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii'

CIFTI_TEMPLATE = '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii'

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20'
]

CORR_FILENAME = 'average_correlations_LORO.npy'
SIGNAL_TYPES  = ['Ahat', 'E']
FDR_ALPHA     = 0.05

# Thresholds
MIN_R_RAW = 0.030     # effect size
MIN_ISC   = 0.05      # reliability

OUTPUT_DIR = os.path.join(BASE_DIR, 'Results', 'Q1', 'sig_mask_datadriven')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_glasser_labels(dlabel_path):
    img = nib.load(dlabel_path)
    labels = img.get_fdata().squeeze()
    label_table = img.header.get_axis(0).label[0]
    label_dict = {idx: name for idx, (name, rgba) in label_table.items()}
    return labels, label_dict


def save_dscalar(data, template_path, output_path, map_name='mask'):
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


def load_r_all_subjects(base_dir, signal_type, subjects, fn):
    r_list = []
    for subj in subjects:
        p = Path(base_dir) / signal_type / subj / fn
        if not p.exists():
            continue
        r = np.load(str(p))
        if r.ndim == 2 and r.shape[0] != 4:
            r = r.T
        r_list.append(r)
    if not r_list:
        return None
    return np.stack(r_list, axis=0)


def compute_sig_mask(r_stack, alpha=0.05):
    rc = np.nan_to_num(r_stack, nan=0.0)
    rm = np.max(rc, axis=1)
    t, p = stats.ttest_1samp(rm, popmean=0, axis=0)
    p1 = p / 2.0
    p1[t < 0] = 1.0
    p1 = np.nan_to_num(p1, nan=1.0)
    sig, _ = fdrcorrection(p1, alpha=alpha, method='indep')
    return sig, np.mean(rm, axis=0)


def clean_name(parcel_name):
    s = parcel_name
    for pf in ['L_', 'R_']:
        if s.startswith(pf):
            s = s[2:]
    if s.endswith('_ROI'):
        s = s[:-4]
    return s


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    print('=' * 70)
    print(' Data-Driven ROI Mask (Method A)')
    print(' FDR < %.2f | r >= %.3f | ISC >= %.2f' %
          (FDR_ALPHA, MIN_R_RAW, MIN_ISC))
    print('=' * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load ISC
    # ------------------------------------------------------------------
    print('\n[1] Loading ISC...')
    isc = np.load(ISC_PATH).squeeze()
    n_voxels_isc = len(isc)

    # ------------------------------------------------------------------
    # 2. Load correlations + FDR
    # ------------------------------------------------------------------
    print('\n[2] Loading correlations + FDR...')
    sig_masks = {}
    r_means   = {}
    n_voxels  = None

    for st in SIGNAL_TYPES:
        rs = load_r_all_subjects(BASE_DIR, st, SUBJECTS, CORR_FILENAME)
        if rs is None:
            continue
        n_subjects, n_layers, n_voxels = rs.shape
        sig, rm = compute_sig_mask(rs, FDR_ALPHA)
        sig_masks[st] = sig
        r_means[st]   = rm
        print('  %s: %d sig voxels (%.1f%%)' %
              (st, sig.sum(), 100. * sig.sum() / n_voxels))

    # align voxel counts
    nv = min(n_voxels, n_voxels_isc)
    isc = isc[:nv]
    for k in sig_masks:
        sig_masks[k] = sig_masks[k][:nv]
    for k in r_means:
        r_means[k] = r_means[k][:nv]
    n_voxels = nv

    # union FDR mask
    union_fdr = np.zeros(n_voxels, dtype=bool)
    for m in sig_masks.values():
        union_fdr |= m
    print('  Union FDR: %d voxels (%.1f%%)' %
          (union_fdr.sum(), 100. * union_fdr.sum() / n_voxels))

    # r_max per voxel
    r_max_voxel = np.zeros(n_voxels)
    for rm in r_means.values():
        r_max_voxel = np.maximum(r_max_voxel, rm)

    # ------------------------------------------------------------------
    # 3. Map sig voxels -> Glasser parcels
    # ------------------------------------------------------------------
    print('\n[3] Loading Glasser + computing parcel stats...')
    labels, label_dict = load_glasser_labels(GLASSER_LABEL_PATH)

    raw_parcels = []
    for pidx, pname in label_dict.items():
        if pidx == 0:
            continue
        vtx = np.where(labels == pidx)[0]
        vtx = vtx[vtx < n_voxels]
        nt = len(vtx)
        if nt == 0:
            continue

        vtx_fdr = vtx[union_fdr[vtx]]
        ns = len(vtx_fdr)

        mean_isc    = np.nanmean(isc[vtx]) if nt > 0 else 0
        mean_r_max  = r_max_voxel[vtx_fdr].mean() if ns > 0 else 0
        mean_r_ahat = r_means['Ahat'][vtx_fdr].mean() if ns > 0 and 'Ahat' in r_means else 0
        mean_r_e    = r_means['E'][vtx_fdr].mean() if ns > 0 and 'E' in r_means else 0

        raw_parcels.append({
            'pidx': pidx, 'short': clean_name(pname),
            'nt': nt, 'ns': ns,
            'mean_isc': mean_isc, 'mean_r_max': mean_r_max,
            'mean_r_ahat': mean_r_ahat, 'mean_r_e': mean_r_e,
        })

    # ------------------------------------------------------------------
    # 4. Merge bilateral
    # ------------------------------------------------------------------
    print('\n[4] Merging bilateral...')
    md = {}
    for p in raw_parcels:
        k = p['short']
        if k not in md:
            md[k] = {
                'short': k, 'nt': p['nt'], 'ns': p['ns'],
                'isc_w': p['mean_isc'] * p['nt'],
                'r_w':   p['mean_r_max'] * p['ns'],
                'ra_w':  p['mean_r_ahat'] * p['ns'],
                're_w':  p['mean_r_e'] * p['ns'],
                'pidxs': [p['pidx']],
            }
        else:
            m = md[k]
            m['nt'] += p['nt']; m['ns'] += p['ns']
            m['isc_w'] += p['mean_isc'] * p['nt']
            m['r_w']   += p['mean_r_max'] * p['ns']
            m['ra_w']  += p['mean_r_ahat'] * p['ns']
            m['re_w']  += p['mean_r_e'] * p['ns']
            m['pidxs'].append(p['pidx'])

    merged = []
    for m in md.values():
        ns = m['ns']
        m['mean_isc']    = m['isc_w'] / m['nt'] if m['nt'] > 0 else 0
        m['mean_r_max']  = m['r_w'] / ns if ns > 0 else 0
        m['mean_r_ahat'] = m['ra_w'] / ns if ns > 0 else 0
        m['mean_r_e']    = m['re_w'] / ns if ns > 0 else 0
        merged.append(m)

    # ------------------------------------------------------------------
    # 5. Apply Method A threshold
    # ------------------------------------------------------------------
    print('\n[5] Applying threshold (r >= %.3f, ISC >= %.2f)...' %
          (MIN_R_RAW, MIN_ISC))

    filtered = [p for p in merged
                if p['ns'] > 0
                and p['mean_r_max'] >= MIN_R_RAW
                and p['mean_isc'] >= MIN_ISC]
    filtered.sort(key=lambda x: x['mean_r_max'], reverse=True)

    print('  %d / %d parcels pass threshold' % (len(filtered), len(merged)))

    print('\n  %-5s %-15s %8s %8s %10s %10s %10s' %
          ('Rank', 'Parcel', 'n_total', 'n_sig', 'ISC', 'r_Ahat', 'r_E'))
    print('  ' + '-' * 75)
    for i, p in enumerate(filtered, 1):
        print('  %-5d %-15s %8d %8d %10.4f %10.5f %10.5f' %
              (i, p['short'], p['nt'], p['ns'],
               p['mean_isc'], p['mean_r_ahat'], p['mean_r_e']))

    # ------------------------------------------------------------------
    # 6. Create ROI mask
    # ------------------------------------------------------------------
    print('\n[6] Creating ROI mask...')
    roi_mask = np.zeros(n_voxels, dtype=np.int32)

    for ri, p in enumerate(filtered, 1):
        for pidx in p['pidxs']:
            vtx = np.where(labels == pidx)[0]
            vtx = vtx[vtx < n_voxels]
            vtx_sig = vtx[union_fdr[vtx]]
            roi_mask[vtx_sig] = ri

    n_roi_vox = (roi_mask > 0).sum()
    print('  ROI voxels: %d / %d (%.1f%%)' %
          (n_roi_vox, n_voxels, 100. * n_roi_vox / n_voxels))

    np.save(os.path.join(OUTPUT_DIR, 'ROI_mask_datadriven.npy'), roi_mask)
    save_dscalar(roi_mask.astype(np.float32), CIFTI_TEMPLATE,
                 os.path.join(OUTPUT_DIR, 'ROI_mask_datadriven.dscalar.nii'),
                 map_name='ROI_mask')
    print('  Saved: ROI_mask_datadriven.npy / .dscalar.nii')

    # also save union FDR mask
    np.save(os.path.join(OUTPUT_DIR, 'sig_mask_union.npy'), union_fdr)
    save_dscalar(union_fdr.astype(np.float32), CIFTI_TEMPLATE,
                 os.path.join(OUTPUT_DIR, 'sig_mask_union.dscalar.nii'),
                 map_name='sig_mask_union')

    # ------------------------------------------------------------------
    # 7. Save parcel list CSV
    # ------------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, 'parcel_stats.csv')
    with open(csv_path, 'w') as f:
        f.write('roi_idx,parcel,n_total,n_sig,mean_isc,r_ahat,r_e,r_max\n')
        for ri, p in enumerate(filtered, 1):
            f.write('%d,%s,%d,%d,%.5f,%.6f,%.6f,%.6f\n' %
                    (ri, p['short'], p['nt'], p['ns'], p['mean_isc'],
                     p['mean_r_ahat'], p['mean_r_e'], p['mean_r_max']))
    print('  Saved: %s' % csv_path)

    # ------------------------------------------------------------------
    # 8. Bar chart
    # ------------------------------------------------------------------
    print('\n[7] Generating figure...')
    n = len(filtered)
    if n > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, n * 0.3)))
        names  = [p['short'] for p in filtered]
        r_ahat = [p['mean_r_ahat'] for p in filtered]
        r_e    = [p['mean_r_e'] for p in filtered]
        y_pos  = np.arange(n)

        bar_h = 0.35
        ax.barh(y_pos - bar_h/2, r_ahat, bar_h,
                color='#e74c3c', label='Prediction (Ahat)', alpha=0.85)
        ax.barh(y_pos + bar_h/2, r_e, bar_h,
                color='#3498db', label='Error (E)', alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Mean encoding accuracy (r)')
        ax.set_title('Data-Driven ROI Parcels (FDR<%.2f, r>=%.3f, ISC>=%.2f)\n'
                     'n=%d parcels' % (FDR_ALPHA, MIN_R_RAW, MIN_ISC, n),
                     fontweight='bold')
        ax.invert_yaxis()
        ax.axvline(MIN_R_RAW, color='gray', linestyle='--', linewidth=0.8,
                   label='r threshold (%.3f)' % MIN_R_RAW)
        ax.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, 'parcel_r_bar.png')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print('  Saved: %s' % out_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 70)
    print(' SUMMARY')
    print('=' * 70)
    print(' FDR alpha:    %.3f' % FDR_ALPHA)
    print(' r threshold:  %.3f' % MIN_R_RAW)
    print(' ISC threshold:%.3f' % MIN_ISC)
    for st, mask in sig_masks.items():
        print(' %s sig:      %d voxels' % (st, mask.sum()))
    print(' Union FDR:    %d voxels' % union_fdr.sum())
    print(' ROI parcels:  %d' % len(filtered))
    print(' ROI voxels:   %d (%.1f%%)' % (n_roi_vox, 100. * n_roi_vox / n_voxels))
    print(' Output:       %s' % OUTPUT_DIR)
    print('=' * 70)