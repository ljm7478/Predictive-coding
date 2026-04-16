"""
Lambda diagnostic: compare λ selection distributions between Ahat and E,
and relate selected λ to per-voxel prediction r.

Inputs (per subject, per layer):
  Lambda_layer{L}.npy             (n_folds=8, n_voxels=59412)
  average_correlations_LORO.npy   (n_layers=4, n_voxels=59412)

Outputs (PNG + CSV) under this script's dir.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(
    "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/"
    "PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/"
    "kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4"
)
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15",
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20",
]
FEATS = ["Ahat", "E"]
LAYERS = [1, 2, 3, 4]
LAMBDAS = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)


def load_feat(feat):
    """Return dict: layer -> (lambdas stacked [S*F, V], r stacked [S, V]),
    plus per-subject pile-up rates."""
    per_layer = {}
    pileups = []

    for L in LAYERS:
        lam_all = []
        r_all = []
        for subj in SUBJECTS:
            lam_p = BASE / feat / subj / f"Lambda_layer{L}.npy"
            r_p = BASE / feat / subj / "average_correlations_LORO.npy"
            if not (lam_p.exists() and r_p.exists()):
                print(f"  [skip] {feat}/{subj}/layer{L}")
                continue
            lam = np.load(lam_p)            # (8, V)
            r = np.load(r_p)[L - 1]         # (V,)
            lam_all.append(lam)
            r_all.append(r)

            # pile-up rate for this subject/layer
            flat = lam.ravel()
            pileups.append({
                "feat": feat, "subject": subj, "layer": L,
                "pct_low": (flat == LAMBDAS.min()).mean() * 100,
                "pct_high": (flat == LAMBDAS.max()).mean() * 100,
                "mean_r": float(np.nanmean(r)),
            })
        per_layer[L] = (np.stack(lam_all), np.stack(r_all))  # lam:(S,8,V) r:(S,V)
    return per_layer, pd.DataFrame(pileups)


def histogram_by_layer(data):
    """data: {feat: {L: (lam[S,8,V], r[S,V])}} — returns pct per λ."""
    rows = []
    for feat, per_layer in data.items():
        for L, (lam, _) in per_layer.items():
            flat = lam.ravel()
            total = flat.size
            for lv in LAMBDAS:
                rows.append({
                    "feat": feat, "layer": L, "lambda": float(lv),
                    "pct": (flat == lv).sum() / total * 100,
                })
    return pd.DataFrame(rows)


def r_by_lambda(data):
    """Mean r for voxels whose fold-mode λ equals each grid value."""
    rows = []
    for feat, per_layer in data.items():
        for L, (lam, r) in per_layer.items():
            # mode λ across 8 folds per voxel, per subject
            # lam: (S, 8, V), r: (S, V)
            S, F, V = lam.shape
            for s in range(S):
                # per-voxel modal λ
                lam_s = lam[s]  # (8, V)
                # vectorized mode over the small grid
                counts = np.stack([(lam_s == lv).sum(axis=0) for lv in LAMBDAS])  # (5, V)
                mode_idx = counts.argmax(axis=0)  # (V,)
                for i, lv in enumerate(LAMBDAS):
                    mask = mode_idx == i
                    if mask.sum() == 0:
                        continue
                    rows.append({
                        "feat": feat, "layer": L, "subject": SUBJECTS[s],
                        "lambda": float(lv),
                        "n_voxels": int(mask.sum()),
                        "mean_r": float(np.nanmean(r[s, mask])),
                    })
    return pd.DataFrame(rows)


def plot_histograms(hist_df, out_png):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, L in enumerate(LAYERS):
        ax = axes[i]
        sub = hist_df[hist_df.layer == L]
        x = np.arange(len(LAMBDAS))
        w = 0.38
        for j, feat in enumerate(FEATS):
            vals = sub[sub.feat == feat].sort_values("lambda")["pct"].values
            ax.bar(x + (j - 0.5) * w, vals, w, label=feat,
                   color=["#1f77b4", "#d62728"][j], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{lv:g}" for lv in LAMBDAS])
        ax.set_title(f"Layer {L}")
        ax.set_xlabel("λ")
        if i == 0:
            ax.set_ylabel("% of fold×voxel selections")
        ax.grid(axis="y", alpha=0.3)
    axes[-1].legend()
    fig.suptitle("λ selection distribution across subjects/folds/voxels")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_r_by_lambda(rbl_df, out_png):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    for i, L in enumerate(LAYERS):
        ax = axes[i]
        sub = rbl_df[rbl_df.layer == L]
        for j, feat in enumerate(FEATS):
            f = sub[sub.feat == feat].groupby("lambda").agg(
                mean_r=("mean_r", "mean"), sem=("mean_r", "sem"),
            ).reset_index()
            ax.errorbar(f["lambda"], f["mean_r"], yerr=f["sem"],
                        marker="o", label=feat,
                        color=["#1f77b4", "#d62728"][j], capsize=3)
        ax.set_title(f"Layer {L}")
        ax.set_xlabel("modal λ (across folds)")
        if i == 0:
            ax.set_ylabel("mean r (voxels selecting this λ)")
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.grid(alpha=0.3)
    axes[-1].legend()
    fig.suptitle("Prediction r as a function of selected λ")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_pileup(pileup_df, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, boundary, col in zip(axes, ["pct_low", "pct_high"],
                                 ["λ=0.1 (low)", "λ=0.9 (high)"]):
        piv = pileup_df.pivot_table(index="subject", columns=["feat", "layer"],
                                    values=boundary)
        piv.plot(kind="bar", ax=ax, width=0.9)
        ax.set_title(f"Pile-up at {col}")
        ax.set_ylabel("% of fold×voxel selections")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    print("Loading...")
    data = {}
    pileups = []
    for feat in FEATS:
        per_layer, pu = load_feat(feat)
        data[feat] = per_layer
        pileups.append(pu)
    pileup_df = pd.concat(pileups, ignore_index=True)

    print("Computing histograms...")
    hist_df = histogram_by_layer(data)
    hist_df.to_csv(OUT / "lambda_histogram.csv", index=False)

    print("Computing r-by-λ...")
    rbl_df = r_by_lambda(data)
    rbl_df.to_csv(OUT / "r_by_lambda.csv", index=False)

    pileup_df.to_csv(OUT / "pileup_per_subject.csv", index=False)

    print("Plotting...")
    plot_histograms(hist_df, OUT / "fig1_lambda_histogram.png")
    plot_r_by_lambda(rbl_df, OUT / "fig2_r_by_lambda.png")
    plot_pileup(pileup_df, OUT / "fig3_pileup_per_subject.png")

    # Console summary
    print("\n=== λ histogram (% of selections) ===")
    print(hist_df.pivot_table(index=["feat", "layer"],
                              columns="lambda", values="pct").round(1))
    print("\n=== mean r by modal λ (averaged over subjects) ===")
    print(rbl_df.groupby(["feat", "layer", "lambda"])["mean_r"]
          .mean().unstack("lambda").round(4))

    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
