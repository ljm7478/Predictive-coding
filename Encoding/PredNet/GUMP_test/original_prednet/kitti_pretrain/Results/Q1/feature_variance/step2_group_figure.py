"""
Step 4b: Group-level figures for Direct Comparison (Ahat vs E).
Loads per-subject results and creates summary visualizations.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================

BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCAcap_log10/layer4"
OUTPUT_DIR_NAME = "direct_comparison_LORO"

subjects = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15",
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]

layer_labels = ["Layer 1\n(Ahat0/E0)", "Layer 2\n(Ahat1/E1)",
                "Layer 3\n(Ahat2/E2)", "Layer 4\n(Ahat3/E3)"]
num_layers = 4

# Significance threshold for "well-predicted" voxels
R2_THRESHOLD = 0.005  # voxels with R2_Ahat + R2_E > this are considered meaningful

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading per-subject results...")
all_R2_Ahat = []
all_R2_E = []
all_PI = []
all_r_Ahat = []
all_r_E = []

for subj in subjects:
    mat_path = Path(BASE_SAVE_DIR) / OUTPUT_DIR_NAME / subj / "direct_comparison_LORO.npz"
    d = np.load(str(mat_path))
    all_R2_Ahat.append(d["R2_Ahat"])   # (4, n_voxels)
    all_R2_E.append(d["R2_E"])
    all_PI.append(d["PI_direct"])
    all_r_Ahat.append(d["mean_r_Ahat"])
    all_r_E.append(d["mean_r_E"])
    print(f"  {subj}: loaded")

# Stack: (n_subjects, num_layers, n_voxels)
R2_Ahat = np.stack(all_R2_Ahat)
R2_E = np.stack(all_R2_E)
PI = np.stack(all_PI)
r_Ahat = np.stack(all_r_Ahat)
r_E = np.stack(all_r_E)

n_subjects, _, n_voxels = R2_Ahat.shape
print(f"\nGroup shape: {n_subjects} subjects x {num_layers} layers x {n_voxels} voxels")

# Group-average across subjects
mean_R2_Ahat = R2_Ahat.mean(axis=0)  # (4, n_voxels)
mean_R2_E = R2_E.mean(axis=0)
mean_PI = PI.mean(axis=0)
mean_r_Ahat = r_Ahat.mean(axis=0)
mean_r_E = r_E.mean(axis=0)

# =============================================================================
# FIGURE OUTPUT DIR
# =============================================================================
fig_dir = Path(BASE_SAVE_DIR) / OUTPUT_DIR_NAME / "group_figures"
fig_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# FIGURE 1: Group-mean R2 comparison (Ahat vs E) per layer
# =============================================================================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel A: Bar plot of mean R2 across all voxels ---
ax = axes1[0]
x = np.arange(num_layers)
width = 0.35

# Per-subject means for error bars
subj_mean_R2_Ahat = R2_Ahat.mean(axis=2)  # (n_subj, 4)
subj_mean_R2_E = R2_E.mean(axis=2)

bar_ahat = subj_mean_R2_Ahat.mean(axis=0)
bar_e = subj_mean_R2_E.mean(axis=0)
sem_ahat = subj_mean_R2_Ahat.std(axis=0) / np.sqrt(n_subjects)
sem_e = subj_mean_R2_E.std(axis=0) / np.sqrt(n_subjects)

bars1 = ax.bar(x - width/2, bar_ahat * 1000, width, yerr=sem_ahat * 1000,
               label='Ahat (prediction)', color='#2196F3', alpha=0.85, capsize=3)
bars2 = ax.bar(x + width/2, bar_e * 1000, width, yerr=sem_e * 1000,
               label='E (error)', color='#FF5722', alpha=0.85, capsize=3)

ax.set_xlabel('PredNet Layer')
ax.set_ylabel(r'Mean $R^2$ ($\times 10^{-3}$)')
ax.set_title('A. Group-mean encoding performance')
ax.set_xticks(x)
ax.set_xticklabels([f'L{i+1}' for i in range(num_layers)])
ax.legend(frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# --- Panel B: Mean PI per layer (subject-level dots) ---
ax = axes1[1]
subj_mean_PI = PI.mean(axis=2)  # (n_subj, 4)

for i in range(num_layers):
    jitter = np.random.uniform(-0.08, 0.08, n_subjects)
    ax.scatter(np.full(n_subjects, i) + jitter, subj_mean_PI[:, i],
               color='gray', alpha=0.5, s=25, zorder=3)

group_mean_PI = subj_mean_PI.mean(axis=0)
group_sem_PI = subj_mean_PI.std(axis=0) / np.sqrt(n_subjects)

ax.bar(x, group_mean_PI, width=0.5, color='#9C27B0', alpha=0.3, zorder=1)
ax.errorbar(x, group_mean_PI, yerr=group_sem_PI, fmt='o', color='#9C27B0',
            markersize=8, capsize=5, zorder=4)
ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
ax.set_xlabel('PredNet Layer')
ax.set_ylabel('Mean PI (direct)')
ax.set_title('B. Preference Index per layer')
ax.set_xticks(x)
ax.set_xticklabels([f'L{i+1}' for i in range(num_layers)])
ax.set_ylim(-0.3, 0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(0.02, 0.98, 'Ahat\n(prediction)', transform=ax.transAxes,
        va='top', ha='left', fontsize=8, color='#2196F3')
ax.text(0.02, 0.02, 'E\n(error)', transform=ax.transAxes,
        va='bottom', ha='left', fontsize=8, color='#FF5722')

fig1.suptitle('Direct Comparison: Ahat vs E encoding (LORO CV, N=15)',
              fontsize=13, fontweight='bold', y=1.02)
fig1.tight_layout()
fig1.savefig(str(fig_dir / "fig1_R2_and_PI_bars.png"), dpi=200, bbox_inches='tight')
print(f"Saved: fig1_R2_and_PI_bars.png")


# =============================================================================
# FIGURE 2: PI distribution histograms per layer (group-average voxelwise)
# =============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
axes2 = axes2.flatten()

for lay in range(num_layers):
    ax = axes2[lay]
    pi_vals = mean_PI[lay, :]

    # Only plot voxels with some signal
    r2_sum = mean_R2_Ahat[lay, :] + mean_R2_E[lay, :]
    mask = r2_sum > R2_THRESHOLD
    n_sig = mask.sum()

    if n_sig > 0:
        pi_sig = pi_vals[mask]
        ax.hist(pi_sig, bins=80, range=(-1, 1), color='#9C27B0', alpha=0.7,
                edgecolor='white', linewidth=0.3)
        ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        ax.axvline(x=np.median(pi_sig), color='#E91E63', linewidth=1.5,
                   linestyle='-', label=f'median={np.median(pi_sig):.3f}')

        # Count Ahat-dominant vs E-dominant
        n_ahat = (pi_sig > 0).sum()
        n_e = (pi_sig < 0).sum()
        pct_ahat = n_ahat / len(pi_sig) * 100
        ax.text(0.97, 0.95, f'Ahat>E: {pct_ahat:.1f}%\nE>Ahat: {100-pct_ahat:.1f}%\nn={n_sig}',
                transform=ax.transAxes, va='top', ha='right', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.legend(loc='upper left', fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, f'No voxels above\nR2 threshold ({R2_THRESHOLD})',
                transform=ax.transAxes, ha='center', va='center')

    ax.set_title(layer_labels[lay], fontsize=11)
    ax.set_xlabel('PI (direct)')
    ax.set_ylabel('Voxel count')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig2.suptitle(f'PI distribution (group-mean, voxels with R2_sum > {R2_THRESHOLD})',
              fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(str(fig_dir / "fig2_PI_histograms.png"), dpi=200, bbox_inches='tight')
print(f"Saved: fig2_PI_histograms.png")


# =============================================================================
# FIGURE 3: Subject x Layer heatmap of mean PI
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(8, 8))

subj_mean_PI_for_heatmap = PI.mean(axis=2)  # (n_subj, 4)

sns.heatmap(subj_mean_PI_for_heatmap, annot=True, fmt='.3f',
            cmap='RdBu_r', center=0, vmin=-0.3, vmax=0.3,
            xticklabels=[f'L{i+1}' for i in range(num_layers)],
            yticklabels=subjects,
            cbar_kws={'label': 'PI (direct)', 'shrink': 0.8},
            linewidths=0.5, linecolor='white',
            ax=ax3)
ax3.set_xlabel('PredNet Layer')
ax3.set_ylabel('Subject')
ax3.set_title('Mean PI per subject and layer\n(+: Ahat dominant, -: E dominant)',
              fontsize=12, fontweight='bold')

fig3.tight_layout()
fig3.savefig(str(fig_dir / "fig3_subject_layer_heatmap.png"), dpi=200, bbox_inches='tight')
print(f"Saved: fig3_subject_layer_heatmap.png")


# =============================================================================
# FIGURE 4: Scatter R2_Ahat vs R2_E (group-averaged voxels, per layer)
# =============================================================================
fig4, axes4 = plt.subplots(2, 2, figsize=(12, 10))
axes4 = axes4.flatten()

for lay in range(num_layers):
    ax = axes4[lay]
    ra = mean_R2_Ahat[lay, :]
    re = mean_R2_E[lay, :]

    # Subsample for plotting (too many voxels)
    r2_sum = ra + re
    mask = r2_sum > 0  # at least one model has R2 > 0
    ra_m, re_m = ra[mask], re[mask]

    # Random subsample if too many points
    if len(ra_m) > 5000:
        idx = np.random.choice(len(ra_m), 5000, replace=False)
        ra_plot, re_plot = ra_m[idx], re_m[idx]
    else:
        ra_plot, re_plot = ra_m, re_m

    ax.scatter(re_plot, ra_plot, s=1, alpha=0.15, color='#555555', rasterized=True)

    # Identity line
    lim = max(ra_plot.max(), re_plot.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', linewidth=0.8, alpha=0.5, label='identity')

    # Highlight well-predicted voxels
    sig_mask = r2_sum[mask] > R2_THRESHOLD
    if len(ra_m) > 5000:
        sig_mask_plot = sig_mask[idx]
    else:
        sig_mask_plot = sig_mask
    if sig_mask_plot.sum() > 0:
        ax.scatter(re_plot[sig_mask_plot], ra_plot[sig_mask_plot],
                   s=3, alpha=0.4, color='#E91E63', rasterized=True,
                   label=f'R2_sum>{R2_THRESHOLD} (n={sig_mask.sum()})')

    ax.set_xlabel(r'$R^2_E$ (error)')
    ax.set_ylabel(r'$R^2_{Ahat}$ (prediction)')
    ax.set_title(layer_labels[lay], fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, frameon=False, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig4.suptitle('Voxelwise R2: Ahat vs E (group-mean, 5k subsample)',
              fontsize=13, fontweight='bold')
fig4.tight_layout()
fig4.savefig(str(fig_dir / "fig4_R2_scatter.png"), dpi=200, bbox_inches='tight')
print(f"Saved: fig4_R2_scatter.png")


# =============================================================================
# FIGURE 5: Top voxels - PI distribution for best-predicted voxels
# =============================================================================
fig5, axes5 = plt.subplots(2, 2, figsize=(12, 10))
axes5 = axes5.flatten()

top_percentiles = [99, 95, 90]
colors_top = ['#E91E63', '#FF9800', '#4CAF50']

for lay in range(num_layers):
    ax = axes5[lay]
    ra = mean_R2_Ahat[lay, :]
    re = mean_R2_E[lay, :]
    pi_vals = mean_PI[lay, :]
    r2_max = np.maximum(ra, re)  # best model R2 per voxel

    for pct, color in zip(top_percentiles, colors_top):
        thresh = np.percentile(r2_max, pct)
        mask = r2_max >= thresh
        n_vox = mask.sum()
        pi_top = pi_vals[mask]
        ax.hist(pi_top, bins=50, range=(-1, 1), alpha=0.4, color=color,
                edgecolor='white', linewidth=0.3,
                label=f'top {100-pct}% (n={n_vox}, med={np.median(pi_top):.3f})')

    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(layer_labels[lay], fontsize=11)
    ax.set_xlabel('PI (direct)')
    ax.set_ylabel('Voxel count')
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig5.suptitle('PI for best-predicted voxels (group-mean)',
              fontsize=13, fontweight='bold')
fig5.tight_layout()
fig5.savefig(str(fig_dir / "fig5_top_voxel_PI.png"), dpi=200, bbox_inches='tight')
print(f"Saved: fig5_top_voxel_PI.png")


# =============================================================================
# SUMMARY STATS
# =============================================================================
print("\n" + "=" * 60)
print("GROUP SUMMARY STATISTICS")
print("=" * 60)

for lay in range(num_layers):
    print(f"\n  Layer {lay+1}:")
    print(f"    Mean R2_Ahat = {subj_mean_R2_Ahat.mean(axis=0)[lay]:.6f} "
          f"(+/- {subj_mean_R2_Ahat.std(axis=0)[lay]:.6f})")
    print(f"    Mean R2_E    = {subj_mean_R2_E.mean(axis=0)[lay]:.6f} "
          f"(+/- {subj_mean_R2_E.std(axis=0)[lay]:.6f})")
    print(f"    Mean PI      = {group_mean_PI[lay]:+.4f} "
          f"(+/- {group_sem_PI[lay]:.4f})")

    # Voxel-level: fraction Ahat > E
    pi_group = mean_PI[lay, :]
    r2_sum = mean_R2_Ahat[lay, :] + mean_R2_E[lay, :]
    sig = r2_sum > R2_THRESHOLD
    if sig.sum() > 0:
        pct_ahat = (pi_group[sig] > 0).sum() / sig.sum() * 100
        print(f"    Voxels R2_sum > {R2_THRESHOLD}: {sig.sum()} "
              f"({sig.sum()/n_voxels*100:.1f}%)")
        print(f"    Of those: {pct_ahat:.1f}% Ahat-dominant, "
              f"{100-pct_ahat:.1f}% E-dominant")

print(f"\nAll figures saved to: {fig_dir}")
