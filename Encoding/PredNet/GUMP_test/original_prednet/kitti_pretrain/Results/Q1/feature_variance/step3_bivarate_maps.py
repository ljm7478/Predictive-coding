"""
Step 05: Bivariate Surface Maps for Prediction vs Error
========================================================
Two complementary visualizations (Popham et al. 2021, Nat Neurosci):

  Figure A -- r-based bivariate map (Popham Extended Data Fig. 1 analog)
      Axes: (r_Ahat, r_E) per voxel per layer
      Both axes [0, vmax], unsigned
      Shows WHERE each model explains brain activity
      No dimensionality reduction -- r uses full weight vector implicitly

  Figure B -- NC-beta signed bivariate map (layer profile spatial expansion)
      Axes: (beta_Ahat, beta_E) per voxel per layer
      Both axes [-vmax, +vmax], signed (4-quadrant)
      Shows HOW each model drives each voxel (positive or negative)
      Reveals explaining away (negative Ahat beta in V1 at L4)

Usage:
    python step05_bivariate_maps.py --mode r          # r-based map
    python step05_bivariate_maps.py --mode beta        # NC-beta map
    python step05_bivariate_maps.py --mode both        # both
    python step05_bivariate_maps.py --mode both --group # group-mean
"""

import os
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import nibabel as nib
from nilearn import plotting
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

USE_LOG10 = True

if USE_LOG10:
    BASE_ENCODING_DIR = (
        '/combinelab/03_user/jungmin/01_project/01_Encoding/'
        '01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/'
        'encoding_analysis/original_prednet/kitti_pretrain_result/'
        'LORO_PCAcap_log10/layer4'
    )
else:
    BASE_ENCODING_DIR = (
        '/combinelab/03_user/jungmin/01_project/01_Encoding/'
        '01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/'
        'encoding_analysis/original_prednet/kitti_pretrain_result/'
        'LORO_PCAcap/layer4'
    )

# --- Per-subject raw r paths ---
AHAT_DIR = os.path.join(BASE_ENCODING_DIR, 'Ahat')
E_DIR = os.path.join(BASE_ENCODING_DIR, 'E')
# Expected: <AHAT_DIR>/<subject>/average_correlations_LORO.dscalar.nii (4 layers)

# --- Per-subject ISC ---
ISC_PER_SUBJECT_PATH = os.path.join(
    BASE_ENCODING_DIR, 'divergence_analysis',
    'step0_noise_ceiling', 'isc_per_subject.npy'
)

# --- NC-normalized correlation (r/sqrt(ISC)) dscalar paths ---
NC_CORR_DIR = os.path.join(
    BASE_ENCODING_DIR, 'divergence_analysis',
    'step1_normalized_corr', 'no_thresh_range_global_merged'
)
NC_CORR_AHAT_FILE = 'ahat_corr_norm_global.dscalar.nii'
NC_CORR_E_FILE = 'e_corr_norm_global.dscalar.nii'

# --- NC-normalized beta dscalar paths ---
NC_BETA_DIR = os.path.join(
    BASE_ENCODING_DIR, 'divergence_analysis',
    'step1_normalized_beta', 'no_thresh_range_global_merged'
)
NC_BETA_AHAT_FILE = 'ahat_beta_norm_global.dscalar.nii'
NC_BETA_E_FILE = 'e_beta_norm_global.dscalar.nii'

# --- ROI mask for thresholding ---
ROI_MASK_PATH = os.path.join(
    BASE_ENCODING_DIR, 'Results', 'Q1',
    'sig_mask_datadriven', 'ROI_mask_datadriven.npy'
)

# --- Surface rendering ---
CIFTI_TEMPLATE = (
    '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/'
    'sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/'
    'Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/'
    'ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii'
)
CIFTIFY_DIR_TEMPLATE = (
    '/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/'
    '{subject}/{subject}_ciftify/ciftify/{subject}/MNINonLinear/fsaverage_LR32k'
)

NUM_LAYERS = 4
N_VERTS_FSLR32K = 32492

SUBJECTS = [
    'sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
    'sub-06', 'sub-09', 'sub-10', 'sub-14', 'sub-15',
    'sub-16', 'sub-17', 'sub-18', 'sub-19', 'sub-20',
]

SAVE_DIR = os.path.join(BASE_ENCODING_DIR, 'Results', 'Q1', 'bivariate_maps')


# =============================================================================
# 2. COLORMAPS
# =============================================================================

def make_unsigned_bivariate_cmap(n_bins=8):
    """
    Unsigned (r_Ahat, r_E) colormap for Figure A.
    Popham et al. 2021 Extended Data Fig 1 style.

    Channel mapping (swapped for visibility on dark background):
        Red channel  = E axis (error) -- warm/visible against dark bg
        Blue channel = Ahat axis (prediction)
        Green channel = min(Red, Blue) -> white when both high

        (0, 0) = black    -- neither model explains
        (high Ahat, 0) = blue   -- prediction only
        (0, high E) = red       -- error only
        (high, high) = white    -- both explain equally well
    """
    colors = np.zeros((n_bins * n_bins, 4), dtype=np.float32)
    for a in range(n_bins):
        for e in range(n_bins):
            ra = (a + 0.5) / n_bins   # prediction intensity
            rb = (e + 0.5) / n_bins   # error intensity

            R = rb          # RED from error (visible)
            B = ra          # BLUE from prediction
            G = min(ra, rb) # white when both high

            idx = a * n_bins + e
            colors[idx] = [
                float(np.clip(R, 0, 1)),
                float(np.clip(G, 0, 1)),
                float(np.clip(B, 0, 1)),
                1.0,
            ]
    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    return cmap, colors


def make_signed_bivariate_cmap(n_bins_half=8):
    """
    Signed (beta_Ahat, beta_E) colormap for Figure B.
    Both axes [-1, +1]. 4-quadrant.

    Channel mapping (swapped for visibility):
        Red channel  = max(0, E_normalized)      (positive error -- warm)
        Blue channel = max(0, Ahat_normalized)    (positive prediction)
        Green        = min(Red, Blue)             (white when both positive)

        (+Ahat, +E) = magenta/white  -- both positively tuned
        (+Ahat, -E) = blue           -- prediction only
        (-Ahat, +E) = red            -- EXPLAINING AWAY (key finding, now visible!)
        (-Ahat, -E) = dark           -- both negative / suppressed
    """
    full = 2 * n_bins_half
    colors = np.zeros((full * full, 4), dtype=np.float32)

    for ai in range(full):
        for ei in range(full):
            a = (ai - n_bins_half + 0.5) / n_bins_half
            e = (ei - n_bins_half + 0.5) / n_bins_half

            a_pos = max(0.0, a)
            e_pos = max(0.0, e)

            R = e_pos          # RED from positive error (warm, visible)
            B = a_pos          # BLUE from positive prediction
            G = min(a_pos, e_pos)  # white when both positive

            # Negative quadrants: subtle gray to distinguish from no-data
            a_neg = max(0.0, -a)
            e_neg = max(0.0, -e)
            if a < 0 or e < 0:
                gray_boost = 0.08 * (a_neg + e_neg)
                R = max(R, gray_boost)
                G = max(G, gray_boost * 0.7)
                B = max(B, gray_boost)

            idx = ai * full + ei
            colors[idx] = [
                float(np.clip(R, 0, 1)),
                float(np.clip(G, 0, 1)),
                float(np.clip(B, 0, 1)),
                1.0,
            ]

    cmap = ListedColormap(colors)
    cmap.set_bad((0.2, 0.2, 0.2, 0))
    return cmap, colors, full


# =============================================================================
# 3. CIFTI / SURFACE UTILITIES
# =============================================================================

def grayords_to_full_surface(cifti_img):
    """Expand CIFTI cortex-only grayordinates to full fsLR_32k."""
    data = cifti_img.get_fdata().astype(np.float32)
    if data.ndim == 1:
        data = data[None, :]
    n_maps = data.shape[0]

    brain_axis = cifti_img.header.get_axis(1)
    data_l = np.full((n_maps, N_VERTS_FSLR32K), np.nan, dtype=np.float32)
    data_r = np.full((n_maps, N_VERTS_FSLR32K), np.nan, dtype=np.float32)

    for name, slc, bm in brain_axis.iter_structures():
        verts = bm.vertex
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            data_l[:, verts] = data[:, slc]
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            data_r[:, verts] = data[:, slc]

    return data_l, data_r


def numpy_to_surface(data_2d, template_path=CIFTI_TEMPLATE):
    """Expand (n_layers, n_grayords) numpy array to full fsLR_32k surface.
    Uses CIFTI template for brain model axis mapping."""
    template = nib.load(template_path)
    brain_axis = template.header.get_axis(1)

    if data_2d.ndim == 1:
        data_2d = data_2d[None, :]
    n_maps = data_2d.shape[0]

    data_l = np.full((n_maps, N_VERTS_FSLR32K), np.nan, dtype=np.float32)
    data_r = np.full((n_maps, N_VERTS_FSLR32K), np.nan, dtype=np.float32)

    for name, slc, bm in brain_axis.iter_structures():
        verts = bm.vertex
        n_verts_struct = slc.stop - slc.start if isinstance(slc, slice) else len(slc)
        if n_verts_struct > data_2d.shape[1]:
            continue
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            data_l[:, verts] = data_2d[:, slc]
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            data_r[:, verts] = data_2d[:, slc]

    return data_l, data_r


def load_sulc(subject):
    """Load sulcal depth for background shading."""
    sulc_path = (
        Path(CIFTIFY_DIR_TEMPLATE.format(subject=subject))
        / ('%s.sulc.32k_fs_LR.dscalar.nii' % subject)
    )
    if not sulc_path.exists():
        return None, None
    img = nib.load(str(sulc_path))
    sl, sr = grayords_to_full_surface(img)
    return np.nan_to_num(sl[0], nan=0.0), np.nan_to_num(sr[0], nan=0.0)


def get_surface_meshes(subject):
    """Return paths to LH/RH inflated surfaces."""
    surf_dir = Path(CIFTIFY_DIR_TEMPLATE.format(subject=subject))
    lh = str(surf_dir / ('%s.L.inflated.32k_fs_LR.surf.gii' % subject))
    rh = str(surf_dir / ('%s.R.inflated.32k_fs_LR.surf.gii' % subject))
    if Path(lh).exists() and Path(rh).exists():
        return lh, rh
    return None, None


# =============================================================================
# 4. DATA LOADERS
# =============================================================================

def load_nc_corr_maps():
    """
    Load group-level NC-normalized correlation (r/sqrt(ISC)) dscalars.
    Returns (ahat_corr, e_corr), each (NUM_LAYERS, n_grayords).

    sqrt(ISC) is the noise ceiling for r (Spearman-Brown prophecy).
    """
    ahat_path = os.path.join(NC_CORR_DIR, NC_CORR_AHAT_FILE)
    e_path = os.path.join(NC_CORR_DIR, NC_CORR_E_FILE)

    if not (os.path.exists(ahat_path) and os.path.exists(e_path)):
        print('  [ERROR] NC correlation files not found:')
        print('    %s' % ahat_path)
        print('    %s' % e_path)
        return None, None

    ahat = nib.load(ahat_path).get_fdata().astype(np.float32)
    e = nib.load(e_path).get_fdata().astype(np.float32)
    print('  Loaded NC-normalized r/sqrt(ISC): Ahat %s, E %s' % (str(ahat.shape), str(e.shape)))
    return ahat, e


def load_nc_beta_maps():
    """
    Load group-level NC-normalized beta dscalars.
    Returns (ahat_beta, e_beta), each (NUM_LAYERS, n_grayords).
    """
    ahat_path = os.path.join(NC_BETA_DIR, NC_BETA_AHAT_FILE)
    e_path = os.path.join(NC_BETA_DIR, NC_BETA_E_FILE)

    if not (os.path.exists(ahat_path) and os.path.exists(e_path)):
        print('  [ERROR] NC beta files not found:')
        print('    %s' % ahat_path)
        print('    %s' % e_path)
        return None, None

    ahat = nib.load(ahat_path).get_fdata().astype(np.float32)
    e = nib.load(e_path).get_fdata().astype(np.float32)
    print('  Loaded NC-normalized beta: Ahat %s, E %s' % (str(ahat.shape), str(e.shape)))
    return ahat, e


def load_roi_mask():
    """Load data-driven ROI mask. Returns boolean array or None."""
    if not os.path.exists(ROI_MASK_PATH):
        print('  [WARN] ROI mask not found: %s' % ROI_MASK_PATH)
        return None
    mask = np.load(ROI_MASK_PATH).astype(np.int32)
    return mask > 0  # boolean: True = inside any ROI


# =============================================================================
# 5. BINNING FUNCTIONS
# =============================================================================

def unsigned_to_index(ahat_vals, e_vals, vmax, n_bins):
    """Map (r_Ahat, r_E) to bin index. Both axes [0, vmax]."""
    a_norm = np.clip(ahat_vals / vmax, 0.0, 1.0 - 1e-9)
    e_norm = np.clip(e_vals / vmax, 0.0, 1.0 - 1e-9)
    a_bin = (a_norm * n_bins).astype(int)
    e_bin = (e_norm * n_bins).astype(int)
    idx = (a_bin * n_bins + e_bin).astype(np.float32)
    nan_mask = np.isnan(ahat_vals) | np.isnan(e_vals)
    idx[nan_mask] = 0.0
    return idx


def signed_to_index(ahat_vals, e_vals, vmax, n_bins_half):
    """Map signed (beta_Ahat, beta_E) to bin index. Axes [-vmax, +vmax]."""
    full = 2 * n_bins_half
    a_norm = np.clip(ahat_vals / vmax, -1.0 + 1e-9, 1.0 - 1e-9)
    e_norm = np.clip(e_vals / vmax, -1.0 + 1e-9, 1.0 - 1e-9)
    a_bin = ((a_norm + 1.0) / 2.0 * full).astype(int)
    e_bin = ((e_norm + 1.0) / 2.0 * full).astype(int)
    a_bin = np.clip(a_bin, 0, full - 1)
    e_bin = np.clip(e_bin, 0, full - 1)
    idx = (a_bin * full + e_bin).astype(np.float32)
    nan_mask = np.isnan(ahat_vals) | np.isnan(e_vals)
    idx[nan_mask] = float(n_bins_half * full + n_bins_half)
    return idx


# =============================================================================
# 6. FIGURE RENDERING
# =============================================================================

def render_bivariate_figure(
    ahat_l, ahat_r, e_l, e_r,
    sulc_l, sulc_r, lh_mesh, rh_mesh,
    out_path, title,
    mode='r',          # 'r' or 'beta'
    vmax=0.10,
    n_bins=8,
    roi_mask_l=None,   # optional: mask non-ROI vertices
    roi_mask_r=None,
):
    """
    Render NUM_LAYERS rows x 4 brain views + 2D legend.
    """
    if mode == 'r':
        cmap, color_grid = make_unsigned_bivariate_cmap(n_bins=n_bins)
        total_bins = n_bins
        bin_func = lambda a, e: unsigned_to_index(a, e, vmax, n_bins)
        extent = [0, vmax, 0, vmax]
        x_label = r'$r_{Ahat}$ (prediction accuracy)'
        y_label = r'$r_{E}$ (error accuracy)'
        legend_title = '2D legend (unsigned)'
    elif mode == 'beta':
        cmap, color_grid, total_bins = make_signed_bivariate_cmap(n_bins_half=n_bins)
        bin_func = lambda a, e: signed_to_index(a, e, vmax, n_bins)
        extent = [-vmax, vmax, -vmax, vmax]
        x_label = r'$\beta_{Ahat}$ (prediction weight)'
        y_label = r'$\beta_{E}$ (error weight)'
        legend_title = '4-quadrant legend (signed)'
    else:
        raise ValueError('Unknown mode: %s' % mode)

    fig = plt.figure(figsize=(18, 4 * NUM_LAYERS + 1.5), facecolor='white')
    gs = GridSpec(
        NUM_LAYERS, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.8],
        wspace=0.02, hspace=0.08,
    )

    views = [
        ('left',  'lateral', lh_mesh, sulc_l, 0),
        ('left',  'medial',  lh_mesh, sulc_l, 1),
        ('right', 'lateral', rh_mesh, sulc_r, 2),
        ('right', 'medial',  rh_mesh, sulc_r, 3),
    ]

    for lay in range(NUM_LAYERS):
        # Apply ROI mask if provided
        al = ahat_l[lay].copy()
        ar = ahat_r[lay].copy()
        el = e_l[lay].copy()
        er = e_r[lay].copy()

        if roi_mask_l is not None:
            al[~roi_mask_l] = np.nan
            el[~roi_mask_l] = np.nan
        if roi_mask_r is not None:
            ar[~roi_mask_r] = np.nan
            er[~roi_mask_r] = np.nan

        idx_l = bin_func(al, el)
        idx_r = bin_func(ar, er)

        for hemi, view, mesh, sulc, col in views:
            ax = fig.add_subplot(gs[lay, col], projection='3d')
            idx = idx_l if hemi == 'left' else idx_r

            plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=idx,
                hemi=hemi,
                view=view,
                cmap=cmap,
                vmin=0,
                vmax=total_bins * total_bins - 1,
                symmetric_cbar=False,
                bg_map=sulc,
                bg_on_data=True,
                darkness=0.4,
                axes=ax,
                figure=fig,
                colorbar=False,
            )

            if lay == 0:
                ax.set_title('%sH %s' % (hemi.upper(), view), fontsize=11)
            if col == 0:
                ax.text2D(
                    -0.10, 0.5, 'Layer %d' % (lay + 1),
                    transform=ax.transAxes, fontsize=13, fontweight='bold',
                    rotation=90, va='center',
                )

    # --- 2D legend ---
    ax_leg = fig.add_subplot(gs[:, 4])
    grid_img = color_grid.reshape(total_bins, total_bins, 4)
    grid_img = np.transpose(grid_img, (1, 0, 2))  # rows=E, cols=Ahat; origin='lower' puts low E at bottom
    ax_leg.imshow(grid_img, extent=extent, origin='lower', aspect='equal')
    ax_leg.set_xlabel(x_label, fontsize=10)
    ax_leg.set_ylabel(y_label, fontsize=10)
    ax_leg.set_title(legend_title, fontsize=10)

    if mode == 'beta':
        ax_leg.axhline(0, color='white', linewidth=0.5, alpha=0.6)
        ax_leg.axvline(0, color='white', linewidth=0.5, alpha=0.6)
        # Quadrant annotations
        off = vmax * 0.55
        ax_leg.text(+off, +off, 'Both +\n(converge)',
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold')
        ax_leg.text(+off, -off, 'Pred +\nErr -',
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold')
        ax_leg.text(-off, +off, 'Pred -\nErr +\n(explaining\naway)',
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold')
        ax_leg.text(-off, -off, 'Both -',
                    ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold')

    if mode == 'r':
        ax_leg.set_xticks([0, vmax / 2, vmax])
        ax_leg.set_yticks([0, vmax / 2, vmax])
    else:
        ax_leg.set_xticks([-vmax, 0, vmax])
        ax_leg.set_yticks([-vmax, 0, vmax])

    for spine in ax_leg.spines.values():
        spine.set_linewidth(0.5)

    fig.suptitle(title, fontsize=13, y=0.995)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('    Saved: %s' % out_path)


# =============================================================================
# 7. EXPAND GRAYORDINATES MASK TO FULL SURFACE
# =============================================================================

def expand_mask_to_surface(mask_1d, cifti_template_path):
    """
    Expand a grayordinates-space boolean mask (n_cortex_grayords,)
    into two full fsLR_32k arrays (one per hemisphere).
    """
    template = nib.load(cifti_template_path)
    brain_axis = template.header.get_axis(1)

    mask_l = np.zeros(N_VERTS_FSLR32K, dtype=bool)
    mask_r = np.zeros(N_VERTS_FSLR32K, dtype=bool)

    cortex_idx = 0
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX' not in name:
            continue
        verts = bm.vertex
        n_verts_struct = slc.stop - slc.start
        struct_mask = mask_1d[cortex_idx:cortex_idx + n_verts_struct]
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            mask_l[verts] = struct_mask
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            mask_r[verts] = struct_mask
        cortex_idx += n_verts_struct

    return mask_l, mask_r


# =============================================================================
# 8. PIPELINE FUNCTIONS
# =============================================================================

def run_r_based(vmax, n_bins, use_roi_mask):
    """
    Figure A: NC-normalized r bivariate map (Popham Ext Data Fig 1 analog).
    Uses group-level NC-normalized correlation files (r/ISC).
    NC normalization removes ISC confound so the map shows genuine
    encoding differences, not just signal reliability.
    """
    print('\n' + '=' * 70)
    print(' Figure A: NC-normalized r bivariate map')
    print(' (Popham Ext Data Fig 1 analog, ISC-corrected)')
    print('=' * 70)

    ahat_corr, e_corr = load_nc_corr_maps()
    if ahat_corr is None:
        return

    # Load ROI mask if requested
    roi_mask_l, roi_mask_r = None, None
    if use_roi_mask:
        raw_mask = load_roi_mask()
        if raw_mask is not None:
            roi_mask_l, roi_mask_r = expand_mask_to_surface(
                raw_mask, CIFTI_TEMPLATE)
            print('  ROI mask loaded: L=%d, R=%d vertices' %
                  (roi_mask_l.sum(), roi_mask_r.sum()))

    # Expand to surface
    ahat_cifti = nib.load(os.path.join(NC_CORR_DIR, NC_CORR_AHAT_FILE))
    e_cifti = nib.load(os.path.join(NC_CORR_DIR, NC_CORR_E_FILE))

    al, ar = grayords_to_full_surface(ahat_cifti)
    el, er = grayords_to_full_surface(e_cifti)

    print('  Surface data: L=%s, R=%s' % (str(al.shape), str(ar.shape)))

    # Print per-layer stats
    for lay in range(NUM_LAYERS):
        print('    Layer %d: Ahat_r/ISC [%.4f, %.4f] mean=%.4f  '
              'E_r/ISC [%.4f, %.4f] mean=%.4f' % (
            lay + 1,
            np.nanmin(al[lay]), np.nanmax(al[lay]), np.nanmean(al[lay]),
            np.nanmin(el[lay]), np.nanmax(el[lay]), np.nanmean(el[lay])))

    # Render
    ref_subj = SUBJECTS[0]
    lh_mesh, rh_mesh = get_surface_meshes(ref_subj)
    sulc_l, sulc_r = load_sulc(ref_subj)

    if lh_mesh is None:
        print('  [ERROR] Reference surfaces not found')
        return

    tag = 'ROI_masked' if use_roi_mask else 'unmasked'
    out_path = os.path.join(
        SAVE_DIR,
        'GROUP_NC_corr_bivariate_vmax%.2f_%s.png' % (vmax, tag))
    title = ('GROUP: NC-normalized r (r/ISC) bivariate | vmax=%.2f | %s'
             % (vmax, tag))

    render_bivariate_figure(
        al, ar, el, er, sulc_l, sulc_r, lh_mesh, rh_mesh,
        out_path, title, mode='r', vmax=vmax, n_bins=n_bins,
        roi_mask_l=roi_mask_l, roi_mask_r=roi_mask_r)


def run_beta_based(vmax, n_bins, use_roi_mask):
    """
    Figure B: NC-beta signed bivariate map.
    This uses group-level NC-normalized beta files (already exist).
    """
    print('\n' + '=' * 70)
    print(' Figure B: NC-beta signed bivariate map (layer profile spatial)')
    print('=' * 70)

    ahat_beta, e_beta = load_nc_beta_maps()
    if ahat_beta is None:
        return

    # Load ROI mask
    roi_mask_l, roi_mask_r = None, None
    if use_roi_mask:
        raw_mask = load_roi_mask()
        if raw_mask is not None:
            roi_mask_l, roi_mask_r = expand_mask_to_surface(
                raw_mask, CIFTI_TEMPLATE)
            print('  ROI mask: L=%d, R=%d vertices' %
                  (roi_mask_l.sum(), roi_mask_r.sum()))

    # Expand to surface
    ahat_cifti = nib.load(os.path.join(NC_BETA_DIR, NC_BETA_AHAT_FILE))
    e_cifti = nib.load(os.path.join(NC_BETA_DIR, NC_BETA_E_FILE))

    al, ar = grayords_to_full_surface(ahat_cifti)
    el, er = grayords_to_full_surface(e_cifti)

    print('  Surface data: L=%s, R=%s' % (str(al.shape), str(ar.shape)))

    # Print per-layer stats
    for lay in range(NUM_LAYERS):
        print('    Layer %d: Ahat_beta [%.4f, %.4f] mean=%.4f  '
              'E_beta [%.4f, %.4f] mean=%.4f' % (
            lay + 1,
            np.nanmin(al[lay]), np.nanmax(al[lay]), np.nanmean(al[lay]),
            np.nanmin(el[lay]), np.nanmax(el[lay]), np.nanmean(el[lay])))

    # Render
    ref_subj = SUBJECTS[0]
    lh_mesh, rh_mesh = get_surface_meshes(ref_subj)
    sulc_l, sulc_r = load_sulc(ref_subj)

    if lh_mesh is None:
        print('  [ERROR] Reference surfaces not found')
        return

    tag = 'ROI_masked' if use_roi_mask else 'unmasked'
    out_path = os.path.join(
        SAVE_DIR,
        'GROUP_NC_beta_bivariate_signed_vmax%.2f_%s.png' % (vmax, tag))
    title = ('GROUP: NC-beta signed bivariate | vmax=%.2f | %s'
             % (vmax, tag))

    render_bivariate_figure(
        al, ar, el, er, sulc_l, sulc_r, lh_mesh, rh_mesh,
        out_path, title, mode='beta', vmax=vmax, n_bins=n_bins,
        roi_mask_l=roi_mask_l, roi_mask_r=roi_mask_r)


# =============================================================================
# 9. INDIVIDUAL (UNIVARIATE) MAPS -- sanity check
# =============================================================================

def render_individual_figure(
    ahat_l, ahat_r, e_l, e_r,
    sulc_l, sulc_r, lh_mesh, rh_mesh,
    out_path, title,
    metric_type='r',   # 'r' (unsigned) or 'beta' (signed)
    vmax=0.15,
    roi_mask_l=None,
    roi_mask_r=None,
):
    """
    Render individual (univariate) maps: Ahat and E side by side per layer.
    Each layer = 1 row with 8 brain views (4 Ahat + 4 E).
    """
    if metric_type == 'r':
        cmap_shared = 'hot'
        vmin = 0
        ahat_label = r'$r_{Ahat}/ISC$'
        e_label = r'$r_E/ISC$'
    else:
        cmap_shared = 'RdBu_r'
        vmin = -vmax
        ahat_label = r'$\beta_{Ahat}$'
        e_label = r'$\beta_E$'

    fig = plt.figure(figsize=(28, 4 * NUM_LAYERS + 1), facecolor='white')
    gs = GridSpec(
        NUM_LAYERS, 9, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.15, 1, 1, 1, 1],
        wspace=0.02, hspace=0.10,
    )

    hemi_views = [
        ('left',  'lateral', lh_mesh, sulc_l),
        ('left',  'medial',  lh_mesh, sulc_l),
        ('right', 'lateral', rh_mesh, sulc_r),
        ('right', 'medial',  rh_mesh, sulc_r),
    ]

    for lay in range(NUM_LAYERS):
        # Ahat maps (columns 0-3)
        al = ahat_l[lay].copy()
        ar = ahat_r[lay].copy()
        if roi_mask_l is not None:
            al[~roi_mask_l] = np.nan
        if roi_mask_r is not None:
            ar[~roi_mask_r] = np.nan

        for col, (hemi, view, mesh, sulc) in enumerate(hemi_views):
            ax = fig.add_subplot(gs[lay, col], projection='3d')
            stat = al if hemi == 'left' else ar

            plotting.plot_surf_stat_map(
                surf_mesh=mesh, stat_map=stat,
                hemi=hemi, view=view,
                cmap=cmap_shared, vmin=vmin, vmax=vmax,
                symmetric_cbar=False,
                bg_map=sulc, bg_on_data=True, darkness=0.4,
                axes=ax, figure=fig, colorbar=False,
            )
            if lay == 0:
                ax.set_title('%sH %s' % (hemi.upper(), view), fontsize=9)
            if col == 0:
                ax.text2D(-0.12, 0.5, 'L%d' % (lay + 1),
                          transform=ax.transAxes, fontsize=12,
                          fontweight='bold', rotation=90, va='center')

        # Separator column (4) - add label
        if lay == 0:
            ax_sep = fig.add_subplot(gs[lay, 4])
            ax_sep.axis('off')

        # E maps (columns 5-8)
        el = e_l[lay].copy()
        er = e_r[lay].copy()
        if roi_mask_l is not None:
            el[~roi_mask_l] = np.nan
        if roi_mask_r is not None:
            er[~roi_mask_r] = np.nan

        for col_offset, (hemi, view, mesh, sulc) in enumerate(hemi_views):
            col = col_offset + 5
            ax = fig.add_subplot(gs[lay, col], projection='3d')
            stat = el if hemi == 'left' else er

            plotting.plot_surf_stat_map(
                surf_mesh=mesh, stat_map=stat,
                hemi=hemi, view=view,
                cmap=cmap_shared, vmin=vmin, vmax=vmax,
                symmetric_cbar=False,
                bg_map=sulc, bg_on_data=True, darkness=0.4,
                axes=ax, figure=fig, colorbar=False,
            )
            if lay == 0:
                ax.set_title('%sH %s' % (hemi.upper(), view), fontsize=9)

    # Add Ahat / E column headers
    fig.text(0.25, 0.98, 'Prediction (Ahat) -- %s' % ahat_label,
             ha='center', fontsize=14, fontweight='bold', color='#c0392b')
    fig.text(0.75, 0.98, 'Error (E) -- %s' % e_label,
             ha='center', fontsize=14, fontweight='bold', color='#2980b9')

    # Colorbars
    cbar_width = 0.12
    cbar_height = 0.015

    # Ahat colorbar
    cbar_ax1 = fig.add_axes([0.13, 0.02, cbar_width, cbar_height])
    sm1 = plt.cm.ScalarMappable(
        cmap=cmap_shared,
        norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation='horizontal')
    cb1.set_label(ahat_label, fontsize=10)
    cb1.ax.tick_params(labelsize=8)

    # E colorbar
    cbar_ax2 = fig.add_axes([0.63, 0.02, cbar_width, cbar_height])
    sm2 = plt.cm.ScalarMappable(
        cmap=cmap_shared,
        norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation='horizontal')
    cb2.set_label(e_label, fontsize=10)
    cb2.ax.tick_params(labelsize=8)

    fig.suptitle(title, fontsize=13, y=1.01)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print('    Saved: %s' % out_path)


def run_individual(vmax_r, vmax_beta, use_roi_mask):
    """Render individual (univariate) maps for sanity checking bivariate maps."""
    print('\n' + '=' * 70)
    print(' INDIVIDUAL MAPS (univariate sanity check)')
    print('=' * 70)

    # ROI mask
    roi_mask_l, roi_mask_r = None, None
    if use_roi_mask:
        raw_mask = load_roi_mask()
        if raw_mask is not None:
            roi_mask_l, roi_mask_r = expand_mask_to_surface(
                raw_mask, CIFTI_TEMPLATE)
            print('  ROI mask: L=%d, R=%d vertices' %
                  (roi_mask_l.sum(), roi_mask_r.sum()))

    ref_subj = SUBJECTS[0]
    lh_mesh, rh_mesh = get_surface_meshes(ref_subj)
    sulc_l, sulc_r = load_sulc(ref_subj)
    if lh_mesh is None:
        print('  [ERROR] Surfaces not found')
        return

    tag = 'ROI_masked' if use_roi_mask else 'unmasked'

    # --- r/sqrt(ISC) individual maps ---
    print('\n  Loading NC-normalized r/sqrt(ISC)...')
    ahat_corr, e_corr = load_nc_corr_maps()
    if ahat_corr is not None:
        ahat_cifti = nib.load(os.path.join(NC_CORR_DIR, NC_CORR_AHAT_FILE))
        e_cifti = nib.load(os.path.join(NC_CORR_DIR, NC_CORR_E_FILE))
        al, ar = grayords_to_full_surface(ahat_cifti)
        el, er = grayords_to_full_surface(e_cifti)

        out_path = os.path.join(
            SAVE_DIR,
            'INDIVIDUAL_NC_corr_vmax%.2f_%s.png' % (vmax_r, tag))
        render_individual_figure(
            al, ar, el, er, sulc_l, sulc_r, lh_mesh, rh_mesh,
            out_path,
            title='Individual maps: NC-normalized r/ISC | vmax=%.2f | %s' % (vmax_r, tag),
            metric_type='r', vmax=vmax_r,
            roi_mask_l=roi_mask_l, roi_mask_r=roi_mask_r)

    # --- Beta individual maps ---
    print('\n  Loading NC-normalized beta...')
    ahat_beta, e_beta = load_nc_beta_maps()
    if ahat_beta is not None:
        ahat_cifti = nib.load(os.path.join(NC_BETA_DIR, NC_BETA_AHAT_FILE))
        e_cifti = nib.load(os.path.join(NC_BETA_DIR, NC_BETA_E_FILE))
        al, ar = grayords_to_full_surface(ahat_cifti)
        el, er = grayords_to_full_surface(e_cifti)

        out_path = os.path.join(
            SAVE_DIR,
            'INDIVIDUAL_NC_beta_vmax%.2f_%s.png' % (vmax_beta, tag))
        render_individual_figure(
            al, ar, el, er, sulc_l, sulc_r, lh_mesh, rh_mesh,
            out_path,
            title='Individual maps: NC-normalized beta | vmax=%.2f | %s' % (vmax_beta, tag),
            metric_type='beta', vmax=vmax_beta,
            roi_mask_l=roi_mask_l, roi_mask_r=roi_mask_r)


# =============================================================================
# 10. PER-SUBJECT GRID (Popham Extended Data Fig. 1 style)
# =============================================================================

def load_per_subject_nc_r(subject, subj_idx, isc_all):
    """
    Load raw r for one subject, divide by that subject's ISC.

    Args:
        subject: subject ID string
        subj_idx: index into isc_per_subject array
        isc_all: (n_subjects, n_voxels) ISC array

    Returns:
        r_ahat_nc, r_e_nc: each (NUM_LAYERS, n_grayords) or None
    """
    ahat_path = os.path.join(AHAT_DIR, subject, 'average_correlations_LORO.dscalar.nii')
    e_path = os.path.join(E_DIR, subject, 'average_correlations_LORO.dscalar.nii')

    if not (os.path.exists(ahat_path) and os.path.exists(e_path)):
        return None, None

    r_ahat = nib.load(ahat_path).get_fdata().astype(np.float32)
    r_e = nib.load(e_path).get_fdata().astype(np.float32)

    if r_ahat.ndim == 1:
        r_ahat = r_ahat[None, :]
    if r_e.ndim == 1:
        r_e = r_e[None, :]

    # Get this subject's ISC (1, n_voxels)
    isc_subj = isc_all[subj_idx]
    isc_subj = np.where(isc_subj > 0.01, isc_subj, np.nan)  # avoid division by near-zero

    # NC normalize: r / sqrt(ISC) per layer (noise ceiling for r = sqrt(ISC))
    sqrt_isc = np.sqrt(isc_subj)
    r_ahat_nc = r_ahat / sqrt_isc[None, :]
    r_e_nc = r_e / sqrt_isc[None, :]

    # Clip extreme values (near-zero ISC can create huge ratios)
    r_ahat_nc = np.clip(r_ahat_nc, 0, 2.0)
    r_e_nc = np.clip(r_e_nc, 0, 2.0)

    return r_ahat_nc, r_e_nc


def load_per_subject_nc_beta(subject, subj_idx, isc_all):
    """
    Load raw beta (PC1) for one subject, divide by that subject's ISC.

    Beta is signed, ISC is positive, so sign is preserved.

    Returns:
        beta_ahat_nc, beta_e_nc: each (NUM_LAYERS, n_grayords) or None
    """
    ahat_path = os.path.join(AHAT_DIR, subject, 'beta_pc1_all_layers_LORO.dscalar.nii')
    e_path = os.path.join(E_DIR, subject, 'beta_pc1_all_layers_LORO.dscalar.nii')

    if not (os.path.exists(ahat_path) and os.path.exists(e_path)):
        return None, None

    b_ahat = nib.load(ahat_path).get_fdata().astype(np.float32)
    b_e = nib.load(e_path).get_fdata().astype(np.float32)

    if b_ahat.ndim == 1:
        b_ahat = b_ahat[None, :]
    if b_e.ndim == 1:
        b_e = b_e[None, :]

    # Get this subject's ISC
    isc_subj = isc_all[subj_idx]
    isc_subj = np.where(isc_subj > 0.01, isc_subj, np.nan)

    # NC normalize: beta / ISC (sign preserved)
    b_ahat_nc = b_ahat / isc_subj[None, :]
    b_e_nc = b_e / isc_subj[None, :]

    # Clip extremes
    b_ahat_nc = np.clip(b_ahat_nc, -2.0, 2.0)
    b_e_nc = np.clip(b_e_nc, -2.0, 2.0)

    return b_ahat_nc, b_e_nc


def render_subject_grid(
    layer_idx,
    subjects, isc_all,
    lh_mesh, rh_mesh, sulc_l, sulc_r,
    out_path, vmax=0.15,
    n_bins=8,
    metric_type='r',
    roi_mask_l=None, roi_mask_r=None,
):
    """
    Render per-subject bivariate grid for ONE layer.
    Popham Extended Data Fig. 1 style: each subject = LH lat + RH lat.
    Grid layout: 3 columns x 5 rows = 15 subjects.

    metric_type: 'r' (unsigned) or 'beta' (signed 4-quadrant)
    """
    if metric_type == 'r':
        cmap, color_grid = make_unsigned_bivariate_cmap(n_bins=n_bins)
        total_bins = n_bins
        bin_func = lambda a, e: unsigned_to_index(a, e, vmax, n_bins)
        extent = [0, vmax, 0, vmax]
        x_label = r'$r_{Ahat}/ISC$'
        y_label = r'$r_E/ISC$'
        title_metric = 'NC-normalized r'
    else:
        cmap, color_grid, total_bins = make_signed_bivariate_cmap(n_bins_half=n_bins)
        bin_func = lambda a, e: signed_to_index(a, e, vmax, n_bins)
        extent = [-vmax, vmax, -vmax, vmax]
        x_label = r'$\beta_{Ahat}/ISC$'
        y_label = r'$\beta_E/ISC$'
        title_metric = 'NC-normalized beta'

    n_subj = len(subjects)
    ncols = 3
    nrows = int(np.ceil(n_subj / ncols))

    fig = plt.figure(figsize=(5.5 * ncols + 2, 3.5 * nrows), facecolor='white')
    gs = GridSpec(nrows, ncols * 2 + 1, figure=fig,
                  width_ratios=[1, 1] * ncols + [0.6],
                  wspace=0.01, hspace=0.12)

    template = nib.load(CIFTI_TEMPLATE)
    brain_axis = template.header.get_axis(1)

    for si, subject in enumerate(subjects):
        row = si // ncols
        col_base = (si % ncols) * 2

        # Load and NC-normalize
        if metric_type == 'r':
            ahat_nc, e_nc = load_per_subject_nc_r(subject, si, isc_all)
        else:
            ahat_nc, e_nc = load_per_subject_nc_beta(subject, si, isc_all)

        if ahat_nc is None:
            print('    [SKIP] %s' % subject)
            continue

        val_a = ahat_nc[layer_idx]
        val_e = e_nc[layer_idx]

        # Expand to surface
        al = np.full(N_VERTS_FSLR32K, np.nan, dtype=np.float32)
        ar = np.full(N_VERTS_FSLR32K, np.nan, dtype=np.float32)
        el = np.full(N_VERTS_FSLR32K, np.nan, dtype=np.float32)
        er = np.full(N_VERTS_FSLR32K, np.nan, dtype=np.float32)

        for name, slc, bm in brain_axis.iter_structures():
            verts = bm.vertex
            if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
                al[verts] = val_a[slc]
                el[verts] = val_e[slc]
            elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
                ar[verts] = val_a[slc]
                er[verts] = val_e[slc]

        if roi_mask_l is not None:
            al[~roi_mask_l] = np.nan
            el[~roi_mask_l] = np.nan
        if roi_mask_r is not None:
            ar[~roi_mask_r] = np.nan
            er[~roi_mask_r] = np.nan

        idx_l = bin_func(al, el)
        idx_r = bin_func(ar, er)

        # LH lateral
        ax_l = fig.add_subplot(gs[row, col_base], projection='3d')
        plotting.plot_surf_stat_map(
            surf_mesh=lh_mesh, stat_map=idx_l,
            hemi='left', view='lateral',
            cmap=cmap, vmin=0, vmax=total_bins * total_bins - 1,
            symmetric_cbar=False,
            bg_map=sulc_l, bg_on_data=True, darkness=0.4,
            axes=ax_l, figure=fig, colorbar=False)

        ax_l.set_title(subject, fontsize=10, fontweight='bold', pad=0)

        # RH lateral
        ax_r = fig.add_subplot(gs[row, col_base + 1], projection='3d')
        plotting.plot_surf_stat_map(
            surf_mesh=rh_mesh, stat_map=idx_r,
            hemi='right', view='lateral',
            cmap=cmap, vmin=0, vmax=total_bins * total_bins - 1,
            symmetric_cbar=False,
            bg_map=sulc_r, bg_on_data=True, darkness=0.4,
            axes=ax_r, figure=fig, colorbar=False)

        print('    + %s' % subject)

    # Legend
    ax_leg = fig.add_subplot(gs[:, ncols * 2])
    grid_img = color_grid.reshape(total_bins, total_bins, 4)
    grid_img = np.transpose(grid_img, (1, 0, 2))
    ax_leg.imshow(grid_img, extent=extent, origin='lower', aspect='equal')
    ax_leg.set_xlabel(x_label, fontsize=10)
    ax_leg.set_ylabel(y_label, fontsize=10)
    ax_leg.set_title('2D legend', fontsize=10)

    if metric_type == 'beta':
        ax_leg.axhline(0, color='white', linewidth=0.5, alpha=0.6)
        ax_leg.axvline(0, color='white', linewidth=0.5, alpha=0.6)

    fig.suptitle(
        'Per-subject %s bivariate | Layer %d | vmax=%.2f'
        % (title_metric, layer_idx + 1, vmax),
        fontsize=14, y=1.01)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('    Saved: %s' % out_path)


def run_subject_grid(vmax_r, vmax_beta, n_bins, use_roi_mask, layers=None):
    """Render per-subject bivariate grid for both r and beta."""
    print('\n' + '=' * 70)
    print(' PER-SUBJECT GRID (Popham Extended Data Fig 1 style)')
    print('=' * 70)

    # Load ISC per subject
    if not os.path.exists(ISC_PER_SUBJECT_PATH):
        print('  [ERROR] ISC per subject not found: %s' % ISC_PER_SUBJECT_PATH)
        return

    isc_all = np.load(ISC_PER_SUBJECT_PATH)
    print('  ISC per subject shape: %s' % str(isc_all.shape))

    if isc_all.shape[0] != len(SUBJECTS):
        print('  [ERROR] ISC subjects (%d) != SUBJECTS list (%d)' %
              (isc_all.shape[0], len(SUBJECTS)))
        return

    # ROI mask
    roi_mask_l, roi_mask_r = None, None
    if use_roi_mask:
        raw_mask = load_roi_mask()
        if raw_mask is not None:
            roi_mask_l, roi_mask_r = expand_mask_to_surface(
                raw_mask, CIFTI_TEMPLATE)
            print('  ROI mask: L=%d, R=%d' % (roi_mask_l.sum(), roi_mask_r.sum()))

    # Surfaces
    ref_subj = SUBJECTS[0]
    lh_mesh, rh_mesh = get_surface_meshes(ref_subj)
    sulc_l, sulc_r = load_sulc(ref_subj)
    if lh_mesh is None:
        print('  [ERROR] Surfaces not found')
        return

    tag = 'ROI_masked' if use_roi_mask else 'unmasked'

    if layers is None:
        layers = list(range(NUM_LAYERS))

    # --- r-based grids ---
    print('\n  === r-based subject grids ===')
    for lay in layers:
        print('\n  Layer %d (r)...' % (lay + 1))
        out_path = os.path.join(
            SAVE_DIR,
            'SUBJECT_GRID_NC_r_layer%d_vmax%.2f_%s.png' % (lay + 1, vmax_r, tag))
        render_subject_grid(
            lay, SUBJECTS, isc_all,
            lh_mesh, rh_mesh, sulc_l, sulc_r,
            out_path, vmax=vmax_r, n_bins=n_bins,
            metric_type='r',
            roi_mask_l=roi_mask_l, roi_mask_r=roi_mask_r)

    # --- beta-based grids ---
    print('\n  === beta-based subject grids ===')
    for lay in layers:
        print('\n  Layer %d (beta)...' % (lay + 1))
        out_path = os.path.join(
            SAVE_DIR,
            'SUBJECT_GRID_NC_beta_layer%d_vmax%.2f_%s.png' % (lay + 1, vmax_beta, tag))
        render_subject_grid(
            lay, SUBJECTS, isc_all,
            lh_mesh, rh_mesh, sulc_l, sulc_r,
            out_path, vmax=vmax_beta, n_bins=n_bins,
            metric_type='beta',
            roi_mask_l=roi_mask_l, roi_mask_r=roi_mask_r)


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Step 05: Bivariate + individual maps (NC-corr + NC-beta)')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['r', 'beta', 'both', 'individual',
                                 'subject_grid', 'all'],
                        help='Which figure(s) to generate: '
                             'r = NC-normalized r/ISC bivariate, '
                             'beta = NC-normalized beta bivariate (signed), '
                             'both = r + beta bivariate, '
                             'individual = univariate Ahat/E side-by-side, '
                             'subject_grid = per-subject bivariate grid (Popham style), '
                             'all = everything')
    parser.add_argument('--vmax_r', type=float, default=0.10,
                        help='vmax for r-based map axes (r/ISC range, '
                             'typical values ~0.20, default 0.10)')
    parser.add_argument('--vmax_beta', type=float, default=0.10,
                        help='vmax for beta-based map axes (symmetric)')
    parser.add_argument('--n_bins', type=int, default=8,
                        help='Bins per axis (r) or per half-axis (beta)')
    parser.add_argument('--roi_mask', action='store_true',
                        help='Mask non-ROI vertices (show only data-driven ROI)')
    parser.add_argument('--no_roi_mask', action='store_true',
                        help='Show entire cortex (no masking)')
    parser.add_argument('--layers', nargs='+', type=int, default=None,
                        help='Which layers to plot for subject_grid (1-indexed, e.g. --layers 3 4)')

    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)

    use_mask = args.roi_mask and not args.no_roi_mask

    if args.mode in ('r', 'both', 'all'):
        run_r_based(vmax=args.vmax_r, n_bins=args.n_bins,
                    use_roi_mask=use_mask)

    if args.mode in ('beta', 'both', 'all'):
        run_beta_based(vmax=args.vmax_beta, n_bins=args.n_bins,
                       use_roi_mask=use_mask)

    if args.mode in ('individual', 'all'):
        run_individual(vmax_r=args.vmax_r, vmax_beta=args.vmax_beta,
                       use_roi_mask=use_mask)

    if args.mode in ('subject_grid', 'all'):
        layers_0idx = None
        if args.layers is not None:
            layers_0idx = [l - 1 for l in args.layers]
        run_subject_grid(vmax_r=args.vmax_r, vmax_beta=args.vmax_beta,
                         n_bins=args.n_bins,
                         use_roi_mask=use_mask, layers=layers_0idx)

    print('\n' + '=' * 70)
    print(' Done! Output: %s' % SAVE_DIR)
    print('=' * 70)


if __name__ == '__main__':
    main()