"""
Wen-style Best Layer Map + ROI Profile  (STANDALONE)
=====================================================
Run independently:  python wen_standalone.py

Loads pre-computed .npy/.dscalar.nii files from SAVE_DIR
and writes PNGs to WEN_SAVE_DIR.
No dependency on the main layer_profile script.
"""

import os, sys, io, traceback
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# PATHS  -- edit if needed
# ============================================================
SAVE_DIR = (
    '/combinelab/03_user/jungmin/01_project/01_Encoding/'
    '01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/'
    'encoding_analysis/modified_prednet/kitti_pretrain_result/'
    'four_train_four_test_PCAcap/layer4/Results/Q1/layer_profile'
)
WEN_SAVE_DIR = (
    '/combinelab/03_user/jungmin/01_project/01_Encoding/'
    '01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/'
    'encoding_analysis/modified_prednet/kitti_pretrain_result/'
    'four_train_four_test_PCAcap/layer4/Results/Q1/wen_style_layer_profile'
)
HCP_SURF_LH = (
    '/combinelab/02_data/21_standard_file/template/surface/'
    'cortical_surface/S900.L.inflated_MSMAll.32k_fs_LR.surf.gii'
)
HCP_SURF_RH = (
    '/combinelab/02_data/21_standard_file/template/surface/'
    'cortical_surface/S900.R.inflated_MSMAll.32k_fs_LR.surf.gii'
)

N_LAYERS = 4
HIERARCHY_ORDER = [
    'Early Visual', 'Mid-Dorsal', 'Motion', 'Object',
    'Face', 'Scene', 'Attention', 'Visuomotor'
]
ROI_SUBSET = ['Early Visual', 'Object', 'Face', 'Attention', 'Visuomotor']

ROI_COLORS = {
    'Early Visual': '#2C3E50', 'Mid-Dorsal': '#E74C3C',
    'Motion':       '#F39C12', 'Object':     '#27AE60',
    'Face':         '#9B59B6', 'Scene':      '#3498DB',
    'Attention':    '#1ABC9C', 'Visuomotor': '#E91E63',
}
ROI_MARKERS = {
    'Early Visual': 'o', 'Mid-Dorsal': 's', 'Motion': '^',
    'Object': 'D',        'Face': 'v',       'Scene': 'p',
    'Attention': 'h',     'Visuomotor': 'X',
}

LAYER_COLORS = ['#2166AC', '#74ADD1', '#F46D43', '#A50026']
LAYER_CMAP   = ListedColormap(LAYER_COLORS)
LAYER_NORM   = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], LAYER_CMAP.N)

# ============================================================
# 1. LOAD SAVED DATA
# ============================================================

def load_profiles(save_dir):
    """Load pred/error mean+sem dicts from .npy files."""
    def npy_to_dict(path):
        obj = np.load(path, allow_pickle=True)
        # np.save on a dict produces array(dict, dtype=object)
        if obj.ndim == 0:
            return obj.item()   # unwrap to actual dict
        return obj

    pred_mean = npy_to_dict(os.path.join(save_dir, 'pred_profiles_mean.npy'))
    pred_sem  = npy_to_dict(os.path.join(save_dir, 'pred_profiles_sem.npy'))
    err_mean  = npy_to_dict(os.path.join(save_dir, 'error_profiles_mean.npy'))
    err_sem   = npy_to_dict(os.path.join(save_dir, 'error_profiles_sem.npy'))
    return pred_mean, pred_sem, err_mean, err_sem


def split_cifti(cifti_path):
    """Split CIFTI dscalar -> (lh_32k, rh_32k) numpy arrays."""
    img  = nib.load(cifti_path)
    data = img.get_fdata().squeeze()
    axis = img.header.get_axis(1)
    lh   = np.full(32492, np.nan)
    rh   = np.full(32492, np.nan)
    for name, data_idx, model in axis.iter_structures():
        if 'CORTEX_LEFT'  in name.upper():
            lh[model.vertex] = data[data_idx]
        elif 'CORTEX_RIGHT' in name.upper():
            rh[model.vertex] = data[data_idx]
    return lh, rh

# ============================================================
# 2. RENDER SURFACE PANELS
# ============================================================

def render_one_panel(surf_path, data, hemi, view, title, dpi=120):
    """Render a single surface view to a numpy RGBA array."""
    from nilearn import plotting as nl_plotting
    fig = plt.figure(figsize=(4, 3.5))
    ax  = fig.add_subplot(111, projection='3d')
    nl_plotting.plot_surf_stat_map(
        surf_mesh=surf_path,
        stat_map=data,
        hemi=hemi, view=view,
        cmap=LAYER_CMAP, vmin=0.5, vmax=4.5,
        colorbar=False, axes=ax, figure=fig,
        bg_on_data=True, darkness=0.4,
        title=title,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi)   # no bbox_inches='tight'
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)   # RGBA float32


def build_surface_strip(lh, rh, dpi=120):
    """
    Render 4 views and stitch horizontally.
    Returns RGBA numpy array, or None on failure.
    """
    views_info = [
        (HCP_SURF_LH, lh, 'left',  'lateral', 'LH lateral'),
        (HCP_SURF_LH, lh, 'left',  'medial',  'LH medial'),
        (HCP_SURF_RH, rh, 'right', 'lateral', 'RH lateral'),
        (HCP_SURF_RH, rh, 'right', 'medial',  'RH medial'),
    ]
    panels = []
    for surf, data, hemi, view, title in views_info:
        try:
            arr = render_one_panel(surf, data, hemi, view, title, dpi=dpi)
            panels.append(arr)
            print("  panel '{}': {}x{}".format(title, arr.shape[1], arr.shape[0]))
        except Exception:
            print("  [ERROR] panel '{}' failed:".format(title))
            traceback.print_exc()
            panels.append(None)

    valid = [p for p in panels if p is not None]
    if not valid:
        return None

    h = min(p.shape[0] for p in valid)
    w = min(p.shape[1] for p in valid)

    rgba = []
    for p in panels:
        if p is None:
            p = np.ones((h, w, 4), dtype=np.float32)
        else:
            p = p[:h, :w, :]
        if p.shape[2] == 3:
            p = np.concatenate([p, np.ones((*p.shape[:2], 1), dtype=p.dtype)], axis=2)
        rgba.append(p)

    return np.concatenate(rgba, axis=1)

# ============================================================
# 3. COMBINED WEN-STYLE FIGURE
# ============================================================

def make_wen_figure(pred_cifti, signal_label, profiles_mean, profiles_sem,
                    roi_subset, out_path):
    print("\n--- {} ---".format(signal_label))

    # Surface strip
    lh, rh = split_cifti(pred_cifti)
    print("  LH non-zero verts: {}  RH: {}".format(
        np.sum(lh > 0), np.sum(rh > 0)))

    strip = build_surface_strip(lh, rh)

    # Figure
    n_cols  = len(roi_subset)
    layers  = np.arange(1, N_LAYERS + 1)
    fig     = plt.figure(figsize=(max(4 * n_cols, 16), 10))
    gs      = gridspec.GridSpec(2, n_cols,
                                height_ratios=[2.5, 1.2],
                                hspace=0.45, wspace=0.3)

    # --- Top: surface ---
    ax_top = fig.add_subplot(gs[0, :])
    if strip is not None:
        ax_top.imshow(strip)
        ax_top.axis('off')
        sm   = plt.cm.ScalarMappable(cmap=LAYER_CMAP, norm=LAYER_NORM)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_top, orientation='vertical',
                            fraction=0.012, pad=0.01, shrink=0.8)
        cbar.set_ticks([1, 2, 3, 4])
        cbar.set_ticklabels(['L1', 'L2', 'L3', 'L4'], fontsize=11)
        cbar.set_label('layer index', fontsize=11)
    else:
        ax_top.set_facecolor('#EEEEEE')
        ax_top.text(0.5, 0.5,
                    'Surface render failed.\nOpen {} in wb_view'.format(
                        os.path.basename(pred_cifti)),
                    ha='center', va='center',
                    transform=ax_top.transAxes, fontsize=11, color='#555')
        ax_top.axis('off')

    ax_top.set_title('{} - Best Layer Map'.format(signal_label),
                     fontsize=13, fontweight='bold', pad=8)

    # --- Bottom: ROI line plots ---
    all_vals = []
    for r in roi_subset:
        if r in profiles_mean:
            all_vals.extend(profiles_mean[r])
            if profiles_sem and r in profiles_sem:
                ms = list(profiles_mean[r])
                ss = list(profiles_sem[r])
                all_vals += [m - s for m, s in zip(ms, ss)]
                all_vals += [m + s for m, s in zip(ms, ss)]

    finite = [v for v in all_vals if np.isfinite(v)]
    if finite:
        pad   = (max(finite) - min(finite)) * 0.18
        y_min = min(finite) - pad
        y_max = max(finite) + pad
    else:
        y_min, y_max = -0.1, 0.5

    for ci, roi in enumerate(roi_subset):
        ax = fig.add_subplot(gs[1, ci])

        if roi not in profiles_mean:
            ax.set_title(roi, fontsize=10)
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            continue

        betas  = list(profiles_mean[roi])
        color  = ROI_COLORS.get(roi, '#333')
        marker = ROI_MARKERS.get(roi, 'o')

        if profiles_sem and roi in profiles_sem:
            sems = list(profiles_sem[roi])
            ax.errorbar(layers, betas, yerr=sems,
                        fmt=marker + '-', color=color,
                        linewidth=2.2, markersize=8,
                        markerfacecolor='white', markeredgewidth=2,
                        capsize=4, capthick=1.5, elinewidth=1.5, zorder=3)
        else:
            ax.plot(layers, betas, marker + '-', color=color,
                    linewidth=2.2, markersize=8,
                    markerfacecolor='white', markeredgewidth=2, zorder=3)

        ax.axhline(0, color='#BBBBBB', linewidth=0.8, linestyle='--', zorder=1)
        ax.set_xlim(0.5, N_LAYERS + 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(layers)
        ax.set_xticklabels(['L{}'.format(l) for l in layers], fontsize=9)
        ax.set_title(roi, fontsize=10, fontweight='bold', color=color)
        ax.grid(True, alpha=0.2, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ci == 0:
            ax.set_ylabel('Beta (PC1)', fontsize=10)
        ax.set_xlabel('Layer', fontsize=9)

    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: {}".format(out_path))

# ============================================================
# 4. MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Wen-style visualization  (standalone)")
    print("=" * 60)

    os.makedirs(WEN_SAVE_DIR, exist_ok=True)

    # -- load profiles --
    print("\n[1] Loading profiles from {}".format(SAVE_DIR))
    try:
        pred_mean, pred_sem, err_mean, err_sem = load_profiles(SAVE_DIR)
        print("  pred_mean keys  :", list(pred_mean.keys()))
        print("  pred_mean sample:", {k: [round(v,4) for v in vals]
                                       for k, vals in list(pred_mean.items())[:2]})
    except Exception:
        print("[FATAL] Could not load profile .npy files")
        traceback.print_exc()
        sys.exit(1)

    roi_subset = [r for r in ROI_SUBSET if r in pred_mean]
    print("  ROI subset for bottom panels:", roi_subset)

    # -- Prediction --
    pred_cifti = os.path.join(SAVE_DIR, 'Prediction_best_layer.dscalar.nii')
    make_wen_figure(
        pred_cifti=pred_cifti,
        signal_label='Prediction (Ahat)',
        profiles_mean=pred_mean,
        profiles_sem=pred_sem,
        roi_subset=roi_subset,
        out_path=os.path.join(WEN_SAVE_DIR, 'wen_style_Prediction.png'),
    )

    # -- Error --
    err_cifti = os.path.join(SAVE_DIR, 'Error_best_layer.dscalar.nii')
    make_wen_figure(
        pred_cifti=err_cifti,
        signal_label='Error (E)',
        profiles_mean=err_mean,
        profiles_sem=err_sem,
        roi_subset=roi_subset,
        out_path=os.path.join(WEN_SAVE_DIR, 'wen_style_Error.png'),
    )

    print("\nDone. Files in:", WEN_SAVE_DIR)