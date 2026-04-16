"""
Step 5: 2D Bivariate Preference Maps (Ahat vs E)
        Popham et al. 2021 (Nat Neurosci) Figure 2 style.

For each cortical vertex, color by (R^2_Ahat, R^2_E) jointly using a bivariate
(2D) colormap. Each axis of the colormap represents one model's prediction
performance:

    high R^2_Ahat, low  R^2_E    -> red       (prediction-driven)
    low  R^2_Ahat, high R^2_E    -> blue      (error-driven)
    high both                    -> magenta   (both models contribute)
    low  both                    -> dark      (neither model explains)

This is the analog of `cortex.Volume2D(story_perf, movie_perf, ..., cmap='PU_RdBu_covar')`
used in the Popham paper (see scripts/plotting/plot_perf_2D.py in gallantlab/vl_interface),
adapted for HCP/ciftify CIFTI grayordinates rendered with nilearn on the subject's
fsLR_32k inflated surface.

Inputs (produced by step04_variance_partitioning.py):
    <BASE_SAVE_DIR>/variance_partitioning_LORO/<subject>/R2_Ahat_LORO.dscalar.nii
    <BASE_SAVE_DIR>/variance_partitioning_LORO/<subject>/R2_E_LORO.dscalar.nii

Output:
    <BASE_SAVE_DIR>/variance_partitioning_LORO/figures_2d_preference/<subject>_2d_preference_*.png

Usage:
    python step05_plot_2d_preference.py                              # all subjects
    python step05_plot_2d_preference.py --subjects sub-01 sub-02
    python step05_plot_2d_preference.py --vmax 0.15 --n_bins 8
    python step05_plot_2d_preference.py --group                      # group-mean figure
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from nilearn import plotting


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

USE_LOG10 = True

if USE_LOG10:
    BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4"
else:
    BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap/layer4"

VARPART_DIR = Path(BASE_SAVE_DIR) / "variance_partitioning_LORO"
OUT_DIR     = VARPART_DIR / "figures_2d_preference"

# Subject-specific surface dir template (ciftify fsLR_32k)
CIFTIFY_DIR_TEMPLATE = (
    "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/"
    "{subject}/{subject}_ciftify/ciftify/{subject}/MNINonLinear/fsaverage_LR32k"
)

# Bivariate colormap params (defaults; overrideable on CLI)
N_BINS_DEFAULT = 8        # 8 x 8 = 64 colors
VMAX_DEFAULT   = 0.10     # max R^2 for both axes (typical encoding R^2 range)

NUM_LAYERS = 4

SUBJECTS_DEFAULT = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15",
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20",
]

# fsLR_32k mesh has 32492 vertices per hemisphere (incl. medial wall)
N_VERTS_FSLR32K = 32492


# =============================================================================
# 2. BIVARIATE COLORMAP
# =============================================================================

def make_bivariate_colormap(n_bins=N_BINS_DEFAULT, scheme='red_blue'):
    """
    Build a discrete bivariate colormap as N*N colors.
    Index ordering: idx = a_bin * n_bins + e_bin
        a_bin = R^2_Ahat bin index (0..n_bins-1)
        e_bin = R^2_E    bin index (0..n_bins-1)

    Schemes:
        'red_blue'      : Ahat -> red channel, E -> blue channel.
                          (0,0)=black, (max,0)=red, (0,max)=blue, (max,max)=magenta.
                          Closest in spirit to Popham's PU_RdBu_covar with a black floor.
        'PU_RdBu_covar' : approximation of pycortex's PU_RdBu_covar (purple-red x blue,
                          lighter mid-range). Slightly less saturated than 'red_blue'.
    """
    colors = np.zeros((n_bins * n_bins, 4), dtype=np.float32)

    if scheme == 'red_blue':
        for a in range(n_bins):
            for e in range(n_bins):
                ra = (a + 0.5) / n_bins   # red intensity from Ahat bin center
                rb = (e + 0.5) / n_bins   # blue intensity from E bin center
                rg = 0.15 * min(ra, rb)   # tiny green lift in mid-range
                idx = a * n_bins + e
                colors[idx] = [ra, rg, rb, 1.0]

    elif scheme == 'PU_RdBu_covar':
        for a in range(n_bins):
            for e in range(n_bins):
                ra = (a + 0.5) / n_bins
                rb = (e + 0.5) / n_bins
                R = ra * (1 - 0.5 * rb) + 0.30 * rb
                G = 0.20 * (ra + rb) - 0.40 * (ra * rb)
                B = rb * (1 - 0.5 * ra) + 0.30 * ra
                idx = a * n_bins + e
                colors[idx] = [
                    float(np.clip(R, 0, 1)),
                    float(np.clip(G, 0, 1)),
                    float(np.clip(B, 0, 1)),
                    1.0,
                ]
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))   # NaN -> transparent (background sulc shows through)
    return cmap, colors


def bivariate_to_index(ahat_vals, e_vals, vmax=VMAX_DEFAULT, n_bins=N_BINS_DEFAULT):
    """
    Map (R^2_Ahat, R^2_E) per vertex to a discrete bin index in [0, n_bins^2 - 1].

    NaN inputs (medial-wall vertices) are mapped to bin 0, which corresponds to
    the (R^2_Ahat=0, R^2_E=0) bin and is the darkest color in the bivariate
    colormap. This makes medial-wall vertices visually indistinguishable from
    un-activated cortical vertices (both are dark) -- the correct scientific
    behavior, and avoids the white-blob artifact that nilearn's `threshold`
    parameter introduces in plot_surf_stat_map.
    """
    a_norm = np.clip(ahat_vals / vmax, 0.0, 1.0 - 1e-9)
    e_norm = np.clip(e_vals    / vmax, 0.0, 1.0 - 1e-9)

    a_bin = (a_norm * n_bins).astype(int)
    e_bin = (e_norm * n_bins).astype(int)

    idx = (a_bin * n_bins + e_bin).astype(np.float32)

    nan_mask = np.isnan(ahat_vals) | np.isnan(e_vals)
    idx[nan_mask] = 0.0
    return idx


# =============================================================================
# 3. CIFTI GRAYORDINATES -> FULL fsLR_32k SURFACE
# =============================================================================

def grayordinates_to_full_surface(cifti_img):
    """
    Read a CIFTI dscalar with cortical-only grayordinates and expand it onto the
    full fsLR_32k surface (32492 vertices/hemi). Medial-wall vertices stay NaN.

    Returns:
        data_l : (n_maps, 32492) float32, NaN at medial wall
        data_r : (n_maps, 32492) float32, NaN at medial wall
    """
    data = cifti_img.get_fdata().astype(np.float32)
    if data.ndim == 1:
        data = data[None, :]
    n_maps = data.shape[0]

    brain_axis = cifti_img.header.get_axis(1)

    data_l = np.full((n_maps, N_VERTS_FSLR32K), np.nan, dtype=np.float32)
    data_r = np.full((n_maps, N_VERTS_FSLR32K), np.nan, dtype=np.float32)

    for name, slc, bm in brain_axis.iter_structures():
        verts = bm.vertex  # mesh vertex indices for these grayordinates
        if name == 'CIFTI_STRUCTURE_CORTEX_LEFT':
            data_l[:, verts] = data[:, slc]
        elif name == 'CIFTI_STRUCTURE_CORTEX_RIGHT':
            data_r[:, verts] = data[:, slc]

    return data_l, data_r


def load_sulc_for_subject(subject):
    """
    Load sulc shading map for the subject's fsLR_32k mesh (combined dscalar).
    Returns (sulc_l, sulc_r), each (32492,), or (None, None) if missing.

    The CIFTI dscalar contains only cortical grayordinates, so medial-wall
    vertices come back as NaN. We fill NaN with 0 so that bg_map is valid
    across the full 32k mesh and nilearn doesn't render those vertices as
    plain white.
    """
    sulc_path = Path(CIFTIFY_DIR_TEMPLATE.format(subject=subject)) / \
                f"{subject}.sulc.32k_fs_LR.dscalar.nii"
    if not sulc_path.exists():
        return None, None
    sulc_img = nib.load(str(sulc_path))
    sulc_l, sulc_r = grayordinates_to_full_surface(sulc_img)
    sl = np.nan_to_num(sulc_l[0], nan=0.0)
    sr = np.nan_to_num(sulc_r[0], nan=0.0)
    return sl, sr


# Mapping from --mode to (Ahat-axis CIFTI stem, E-axis CIFTI stem) and labels.
# Each mode loads two cortical maps and renders them as the (red, blue) axes
# of the bivariate preference figure.
MODE_FILES = {
    "r2": (
        "R2_Ahat",
        "R2_E",
        r"$R^2_{\hat{A}}$ (prediction)",
        r"$R^2_{E}$ (error)",
        "R^2 variance partitioning (Lescroart & Gallant 2019)",
    ),
    "beta_marginal": (
        "var_pred_Ahat_marginal",
        "var_pred_E_marginal",
        r"$\mathrm{Var}(X_{\hat{A}}\beta_{\hat{A}})$  (Ahat-only model)",
        r"$\mathrm{Var}(X_{E}\beta_{E})$  (E-only model)",
        r"$\beta$-prediction-variance, marginal models (Popham 2021 Fig 2 analog)",
    ),
    "beta_joint": (
        "var_pred_Ahat_joint",
        "var_pred_E_joint",
        r"$\mathrm{Var}(X_{\hat{A}}\beta_{\hat{A}}^{\mathrm{joint}})$",
        r"$\mathrm{Var}(X_{E}\beta_{E}^{\mathrm{joint}})$",
        r"$\beta$-prediction-variance, joint model (variance-partitioning analog)",
    ),
}


def load_subject_maps(subject, mode="r2"):
    """
    Load the two per-voxel maps used as the (red, blue) axes of the bivariate
    figure for one subject. Mode selects which pair of CIFTI dscalars to load:

        mode='r2'             : R2_Ahat            vs R2_E             (default)
        mode='beta_marginal'  : var_pred_Ahat_marginal vs var_pred_E_marginal
                                (closest analog to Popham 2021 Fig 2)
        mode='beta_joint'     : var_pred_Ahat_joint    vs var_pred_E_joint
                                (joint-model beta partition)

    Returns:
        a_l, a_r, e_l, e_r : each (NUM_LAYERS, 32492)
    or (None, ..., None) if files missing.
    """
    if mode not in MODE_FILES:
        raise ValueError(f"Unknown mode '{mode}'. Choices: {list(MODE_FILES)}")

    a_stem, e_stem, *_ = MODE_FILES[mode]
    subj_dir = VARPART_DIR / subject
    a_path = subj_dir / f"{a_stem}_LORO.dscalar.nii"
    e_path = subj_dir / f"{e_stem}_LORO.dscalar.nii"

    if not (a_path.exists() and e_path.exists()):
        return None, None, None, None

    a_l, a_r = grayordinates_to_full_surface(nib.load(str(a_path)))
    e_l, e_r = grayordinates_to_full_surface(nib.load(str(e_path)))
    return a_l, a_r, e_l, e_r


# Backwards-compatible alias for older callers
def load_subject_r2(subject):
    return load_subject_maps(subject, mode="r2")


# =============================================================================
# 4. FIGURE BUILDER
# =============================================================================

def _add_2d_legend(ax_leg, color_grid, n_bins, vmax, x_label, y_label):
    """Render the bivariate colormap as a 2D legend in `ax_leg`."""
    # color_grid is (n_bins*n_bins, 4) in row-major (a_bin, e_bin) order.
    # Display so that x-axis = Ahat-axis quantity, y-axis = E-axis quantity.
    grid_img = color_grid.reshape(n_bins, n_bins, 4)
    grid_img = np.transpose(grid_img, (1, 0, 2))[::-1]  # (e_bin, a_bin, RGBA), origin lower
    ax_leg.imshow(grid_img, extent=[0, vmax, 0, vmax], origin='lower', aspect='equal')
    ax_leg.set_xlabel(x_label, fontsize=11)
    ax_leg.set_ylabel(y_label, fontsize=11)
    ax_leg.set_title(f'2D legend (n_bins={n_bins})', fontsize=10)
    ax_leg.set_xticks([0, vmax / 2, vmax])
    ax_leg.set_yticks([0, vmax / 2, vmax])
    for spine in ax_leg.spines.values():
        spine.set_linewidth(0.5)


def make_figure(
    r2a_l, r2a_r, r2e_l, r2e_r,
    sulc_l, sulc_r,
    lh_inflated, rh_inflated,
    title,
    out_path,
    vmax=VMAX_DEFAULT,
    n_bins=N_BINS_DEFAULT,
    scheme='red_blue',
    x_label=r'$R^2_{\hat{A}}$ (prediction)',
    y_label=r'$R^2_{E}$ (error)',
):
    """
    Render a single multi-layer figure: NUM_LAYERS rows x 4 brain views + 2D legend.
    """
    cmap, color_grid = make_bivariate_colormap(n_bins=n_bins, scheme=scheme)

    fig = plt.figure(figsize=(16, 4 * NUM_LAYERS), facecolor='white')
    gs = GridSpec(
        NUM_LAYERS, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.7],
        wspace=0.02, hspace=0.05,
    )

    views = [
        ('left',  'lateral', lh_inflated, sulc_l, 0),
        ('left',  'medial',  lh_inflated, sulc_l, 1),
        ('right', 'lateral', rh_inflated, sulc_r, 2),
        ('right', 'medial',  rh_inflated, sulc_r, 3),
    ]

    for lay in range(NUM_LAYERS):
        idx_l = bivariate_to_index(r2a_l[lay], r2e_l[lay], vmax=vmax, n_bins=n_bins)
        idx_r = bivariate_to_index(r2a_r[lay], r2e_r[lay], vmax=vmax, n_bins=n_bins)

        for hemi, view, mesh, sulc, col in views:
            ax = fig.add_subplot(gs[lay, col], projection='3d')
            idx = idx_l if hemi == 'left' else idx_r

            # plot_surf_stat_map with bg_on_data blends sulc shading underneath
            # the bivariate colormap. We deliberately do NOT use `threshold` --
            # nilearn 0.10.4 renders below-threshold vertices as solid white,
            # not as bg_map. Instead, medial-wall vertices were already mapped
            # to bin 0 (darkest color) by `bivariate_to_index`.
            plotting.plot_surf_stat_map(
                surf_mesh=mesh,
                stat_map=idx,
                hemi=hemi,
                view=view,
                cmap=cmap,
                vmin=0,
                vmax=n_bins * n_bins - 1,
                symmetric_cbar=False,
                bg_map=sulc,
                bg_on_data=True,
                darkness=0.4,
                axes=ax,
                figure=fig,
                colorbar=False,
            )

            if lay == 0:
                ax.set_title(f"{hemi.upper()}H {view}", fontsize=11)
            if col == 0:
                ax.text2D(
                    -0.10, 0.5, f"Layer {lay + 1}",
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    rotation=90, va='center',
                )

    # 2D legend
    ax_leg = fig.add_subplot(gs[:, 4])
    _add_2d_legend(ax_leg, color_grid, n_bins=n_bins, vmax=vmax,
                   x_label=x_label, y_label=y_label)

    fig.suptitle(title, fontsize=14, y=0.995)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {out_path}")


def make_figure_for_subject(subject, vmax, n_bins, scheme, mode):
    print(f"\n  Subject: {subject}  (mode={mode})")
    a_l, a_r, e_l, e_r = load_subject_maps(subject, mode=mode)
    if a_l is None:
        print(f"    [SKIP] {mode} files not found in {VARPART_DIR / subject}")
        return

    surf_dir = Path(CIFTIFY_DIR_TEMPLATE.format(subject=subject))
    lh_inflated = str(surf_dir / f"{subject}.L.inflated.32k_fs_LR.surf.gii")
    rh_inflated = str(surf_dir / f"{subject}.R.inflated.32k_fs_LR.surf.gii")
    if not (Path(lh_inflated).exists() and Path(rh_inflated).exists()):
        print(f"    [ERROR] Inflated surfaces not found at {surf_dir}")
        return

    sulc_l, sulc_r = load_sulc_for_subject(subject)

    _, _, x_label, y_label, mode_caption = MODE_FILES[mode]

    out_path = OUT_DIR / f"{subject}_2d_preference_{mode}_{scheme}_vmax{vmax:g}.png"
    title = (
        f"{subject}: {mode_caption}  |  LORO  |  scheme={scheme}  vmax={vmax}"
    )
    make_figure(
        a_l, a_r, e_l, e_r,
        sulc_l, sulc_r,
        lh_inflated, rh_inflated,
        title=title,
        out_path=out_path,
        vmax=vmax,
        n_bins=n_bins,
        scheme=scheme,
        x_label=x_label,
        y_label=y_label,
    )


def make_group_figure(subjects, vmax, n_bins, scheme, mode):
    """
    Average the chosen mode's two maps across subjects (vertex-wise per layer),
    then render the bivariate figure on the first subject's fsLR_32k inflated
    surface. Subjects are aligned in fsLR_32k space, so vertex-wise averaging
    is valid.
    """
    print(f"\n  Group average across {len(subjects)} subjects  (mode={mode})")

    accum_a_l = np.zeros((NUM_LAYERS, N_VERTS_FSLR32K), dtype=np.float64)
    accum_a_r = np.zeros((NUM_LAYERS, N_VERTS_FSLR32K), dtype=np.float64)
    accum_e_l = np.zeros((NUM_LAYERS, N_VERTS_FSLR32K), dtype=np.float64)
    accum_e_r = np.zeros((NUM_LAYERS, N_VERTS_FSLR32K), dtype=np.float64)
    counts_l  = np.zeros((NUM_LAYERS, N_VERTS_FSLR32K), dtype=np.int32)
    counts_r  = np.zeros((NUM_LAYERS, N_VERTS_FSLR32K), dtype=np.int32)

    for subj in subjects:
        a_l, a_r, e_l, e_r = load_subject_maps(subj, mode=mode)
        if a_l is None:
            print(f"    [SKIP] {subj}: missing files for mode '{mode}'")
            continue

        valid_l = ~np.isnan(a_l)
        valid_r = ~np.isnan(a_r)
        accum_a_l += np.where(valid_l, a_l, 0.0)
        accum_a_r += np.where(valid_r, a_r, 0.0)
        accum_e_l += np.where(valid_l, e_l, 0.0)
        accum_e_r += np.where(valid_r, e_r, 0.0)
        counts_l  += valid_l.astype(np.int32)
        counts_r  += valid_r.astype(np.int32)
        print(f"    + {subj}")

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_a_l = np.where(counts_l > 0, accum_a_l / counts_l, np.nan).astype(np.float32)
        mean_a_r = np.where(counts_r > 0, accum_a_r / counts_r, np.nan).astype(np.float32)
        mean_e_l = np.where(counts_l > 0, accum_e_l / counts_l, np.nan).astype(np.float32)
        mean_e_r = np.where(counts_r > 0, accum_e_r / counts_r, np.nan).astype(np.float32)

    # Use the first subject's surfaces as the rendering target
    ref_subj = subjects[0]
    surf_dir = Path(CIFTIFY_DIR_TEMPLATE.format(subject=ref_subj))
    lh_inflated = str(surf_dir / f"{ref_subj}.L.inflated.32k_fs_LR.surf.gii")
    rh_inflated = str(surf_dir / f"{ref_subj}.R.inflated.32k_fs_LR.surf.gii")
    sulc_l, sulc_r = load_sulc_for_subject(ref_subj)

    _, _, x_label, y_label, mode_caption = MODE_FILES[mode]

    out_path = OUT_DIR / f"GROUP_n{len(subjects)}_2d_preference_{mode}_{scheme}_vmax{vmax:g}.png"
    title = (
        f"GROUP (n={len(subjects)}): {mode_caption}  |  LORO  |  scheme={scheme}  vmax={vmax}"
    )
    make_figure(
        mean_a_l, mean_a_r, mean_e_l, mean_e_r,
        sulc_l, sulc_r,
        lh_inflated, rh_inflated,
        title=title,
        out_path=out_path,
        vmax=vmax,
        n_bins=n_bins,
        scheme=scheme,
        x_label=x_label,
        y_label=y_label,
    )


# =============================================================================
# 5. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: 2D bivariate preference figure (Ahat vs E), Popham 2021 Fig 2 style."
    )
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS_DEFAULT,
                        help="Subject IDs to plot (ignored if --group)")
    parser.add_argument("--group", action="store_true",
                        help="Render a single group-mean figure across the given subjects")
    parser.add_argument("--vmax", type=float, default=VMAX_DEFAULT,
                        help="Max value for both colormap axes (default 0.10). "
                             "Choose vmax to match the actual range of the chosen mode.")
    parser.add_argument("--n_bins", type=int, default=N_BINS_DEFAULT,
                        help="Bins per axis; total colors = n_bins^2 (default 8)")
    parser.add_argument("--scheme", type=str, default="red_blue",
                        choices=["red_blue", "PU_RdBu_covar"],
                        help="Bivariate colormap scheme")
    parser.add_argument("--mode", type=str, default="r2",
                        choices=list(MODE_FILES.keys()),
                        help="Which pair of per-voxel maps to use as the (red, blue) axes:\n"
                             "  r2            : R^2 from variance partitioning (Lescroart & Gallant)\n"
                             "  beta_marginal : var(X@beta) from Ahat-only / E-only models (Popham analog)\n"
                             "  beta_joint    : var(X@beta) from joint model with Ahat/E split")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.group:
        make_group_figure(args.subjects, vmax=args.vmax, n_bins=args.n_bins,
                          scheme=args.scheme, mode=args.mode)
    else:
        for subj in args.subjects:
            make_figure_for_subject(subj, vmax=args.vmax, n_bins=args.n_bins,
                                    scheme=args.scheme, mode=args.mode)


if __name__ == "__main__":
    main()
