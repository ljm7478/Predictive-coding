"""
Per-feature variance audit for the 1500-PC regressors used in ridge encoding.

Goal: check whether feature variances span orders of magnitude. If they do,
a single scalar λ cannot balance shrinkage across features, which produces
the bimodal λ-selection pattern we observed.

For each feature type (Ahat, E) and each layer, loads all 8 run regressor
files, computes per-feature variance, and reports:
  - variance range (min / median / max, on log10 scale)
  - effective condition number (max / min)
  - sorted variance plot on log-y axis

Outputs PNG + CSV next to this script. No GPU needed.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REG_BASE = Path(
    "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/"
    "PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/"
    "kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4"
)
OUT = Path(__file__).resolve().parent
ALL_RUNS = [1, 2, 3, 4, 5, 6, 7, 8]

LAYERS = {
    "Ahat": ["Ahat0", "Ahat1", "Ahat2", "Ahat3"],
    "E":    ["E0",    "E1",    "E2",    "E3"],
}


def load_per_feature_var(feat, layer_key):
    """Concatenate all 8 runs (TR axis), demean per run, return per-feature var."""
    chunks = []
    for r in ALL_RUNS:
        with h5py.File(REG_BASE / feat / "regressors" / f"run{r}.h5", "r") as f:
            x = f[layer_key][:].T.astype(np.float32)  # (TR, feat)
        x -= x.mean(0, keepdims=True)
        chunks.append(x)
    X = np.concatenate(chunks, 0)
    return X.var(0)  # (n_features,)


def main():
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharey=True)

    for i, feat in enumerate(["Ahat", "E"]):
        for j, layer in enumerate(LAYERS[feat]):
            var = load_per_feature_var(feat, layer)
            var_sorted = np.sort(var)[::-1]

            vmin, vmed, vmax = float(var.min()), float(np.median(var)), float(var.max())
            cond = vmax / max(vmin, 1e-20)
            span_decades = np.log10(cond)

            rows.append({
                "feat": feat, "layer": layer,
                "n_features": len(var),
                "var_min": vmin, "var_median": vmed, "var_max": vmax,
                "cond_number": cond, "log10_span": span_decades,
            })

            ax = axes[i, j]
            ax.semilogy(np.arange(1, len(var) + 1), var_sorted, lw=1)
            ax.set_title(f"{feat} / {layer}\nspan={span_decades:.2f} dec")
            ax.set_xlabel("feature rank")
            if j == 0:
                ax.set_ylabel("variance (log)")
            ax.grid(alpha=0.3)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "feature_variance_audit.csv", index=False)

    fig.suptitle("Per-feature variance (sorted, log y) — wider span → single λ can't fit all features")
    fig.tight_layout()
    fig.savefig(OUT / "feature_variance_audit.png", dpi=150)
    plt.close(fig)

    print("\n=== Feature variance summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved to: {OUT}")

    max_span = df["log10_span"].max()
    print(f"\nMax log10-span across layers: {max_span:.2f}")
    if max_span >= 2:
        print(">>> Variances span >=2 decades. Recommend per-feature standardization "
              "(z-score each PC using training-fold stats) before ridge.")
    else:
        print(">>> Variance span is modest (<2 decades). Standardization likely "
              "won't change λ behavior much; residual bimodality is probably "
              "signal-absence, not method.")


if __name__ == "__main__":
    main()
