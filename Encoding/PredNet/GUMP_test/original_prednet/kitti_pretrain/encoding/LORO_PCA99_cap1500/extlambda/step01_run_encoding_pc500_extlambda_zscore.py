"""
PC500 variant: extended λ grid + per-feature z-scoring.

Uses the PC500 regressor tree (.../LORO_PCAcap_log10/layer4) as input.
Outputs under a new '_extlambda_zscore' sibling so nothing existing is
overwritten.

Same encoding logic as step01_run_encoding_extlambda_zscore.py, only
paths differ.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voxelwise_encoding_new import voxelwise_encoding_new


# =============================================================================
# Config
# =============================================================================

_SUFFIX = "_extlambda_zscore"

# PC500 tree
REG_BASE_DIR = (
    "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/"
    "PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/"
    "kitti_pretrain_result/LORO_PCAcap_log10/layer4"
)
BASE_SAVE_DIR = REG_BASE_DIR + _SUFFIX  # .../layer4_extlambda_zscore

FMRI_DATA_ROOT = (
    "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/"
    "PredNet/Titanic_trained/GUMP_test/fmri_new/"
)
CIFTI_TEMPLATE = (
    "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/"
    "sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/"
    "Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/"
    "ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"
)

ALL_RUNS = [1, 2, 3, 4, 5, 6, 7, 8]
N_FOLDS = len(ALL_RUNS)
NFOLD_INNER = 3
LAMBDAS = np.array(
    [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0], dtype=np.float32
)

SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15",
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20",
]


# =============================================================================
# Helpers (same as PC1500 variant)
# =============================================================================

def get_layer_key(name):
    return name[1:] if name.startswith("/") else name


def load_fmri_mat(fp):
    out = {}
    with h5py.File(fp, "r") as f:
        s = f["fmri_data_per_run"]
        for r in s.keys():
            out[r] = s[r][:].astype(np.float32)
    return out


def save_W_h5(W, fp):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(fp, "w") as f:
        f.create_dataset("W", data=W, compression="gzip")


def corr_vec(pred, actual):
    p = pred - pred.mean(0, keepdims=True)
    a = actual - actual.mean(0, keepdims=True)
    num = (p * a).sum(0)
    den = np.sqrt((p ** 2).sum(0) * (a ** 2).sum(0))
    return np.nan_to_num(num / (den + 1e-10))


def save_cifti_dscalar(data, template, out_path, feature_type):
    tpl = nib.load(template)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_maps, n_gray = data.shape
    brain_axis = tpl.header.get_axis(1)
    cortex_idx = []
    for name, slc, _ in brain_axis.iter_structures():
        if "CORTEX" in name:
            cortex_idx.extend(range(slc.start, slc.stop))
    if n_gray != len(cortex_idx):
        return False
    axis = brain_axis[np.array(cortex_idx)]
    scalar_axis = nib.cifti2.ScalarAxis([f"{feature_type}{i}" for i in range(n_maps)])
    hdr = nib.cifti2.Cifti2Header.from_axes((scalar_axis, axis))
    img = nib.Cifti2Image(data.astype(np.float32), header=hdr)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(img, out_path)
    return True


def summarize_hist(arr, grid):
    u, c = np.unique(arr, return_counts=True)
    pct = c / c.sum() * 100
    return ", ".join(f"{uu:.3g}:{pp:.1f}%" for uu, pp in zip(u, pct))


# =============================================================================
# Main
# =============================================================================

def run(feature_type, device):
    if feature_type == "Ahat":
        layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"]
    else:
        layer_names = ["/E0", "/E1", "/E2", "/E3"]
    num_layers = len(layer_names)

    reg_path = Path(REG_BASE_DIR) / feature_type / "regressors"
    save_root = Path(BASE_SAVE_DIR) / feature_type

    t_total = time.time()
    print(f"\n{'='*70}")
    print(f" PC500 EXT-λ + Z-SCORE  feat={feature_type}  {N_FOLDS}-fold LORO")
    print(f" λ grid ({len(LAMBDAS)}): {list(LAMBDAS)}")
    print(f" regressors : {reg_path}")
    print(f" save root  : {save_root}")
    print(f"{'='*70}")

    if not reg_path.exists():
        print(f" [ERROR] Regressors not found: {reg_path}")
        return

    for si, subj in enumerate(SUBJECTS):
        t_subj = time.time()
        subj_out = save_root / subj
        subj_out.mkdir(parents=True, exist_ok=True)
        out_mat = subj_out / "average_correlations_LORO.mat"
        if out_mat.exists():
            print(f"\n  {subj} already done. Skipping.")
            continue

        print(f"\n  Subject: {subj} ({si+1}/{len(SUBJECTS)})")

        fmri_path = Path(FMRI_DATA_ROOT) / subj / f"{subj}_per_run_fmri.mat"
        if not fmri_path.exists():
            print(f"    [WARN] missing fMRI for {subj}")
            continue
        fmri_struct = load_fmri_mat(fmri_path)

        regs = {L: {} for L in range(num_layers)}
        fmri = {}
        for r in ALL_RUNS:
            rp = reg_path / f"run{r}.h5"
            with h5py.File(rp, "r") as f:
                for L in range(num_layers):
                    k = get_layer_key(layer_names[L])
                    regs[L][r] = f[k][:].T.astype(np.float32)
            trs = regs[0][r].shape[0]
            fm = fmri_struct[f"run{r}"]
            if fm.shape[0] > trs:
                fm = fm[:trs, :]
            fmri[r] = fm

        all_layer_corrs = []
        for L in range(num_layers):
            t_layer = time.time()
            print(f"\n    Layer {L+1}/{num_layers} ({layer_names[L]})")
            fold_corrs, fold_W, fold_lam = [], [], []

            for fi, test_run in enumerate(ALL_RUNS):
                train_runs = [r for r in ALL_RUNS if r != test_run]
                X_tr = np.concatenate([regs[L][r] for r in train_runs], 0)
                Y_tr = np.concatenate([fmri[r] for r in train_runs], 0)
                X_te = regs[L][test_run].copy()
                Y_te = fmri[test_run].copy()

                # Per-feature z-score with TRAINING stats only
                mx = X_tr.mean(0, keepdims=True)
                sx = X_tr.std(0, keepdims=True)
                sx = np.where(sx < 1e-8, 1.0, sx)
                X_tr = (X_tr - mx) / sx
                X_te = (X_te - mx) / sx

                my = Y_tr.mean(0, keepdims=True)
                Y_tr = Y_tr - my
                Y_te = Y_te - my

                W, _, Lam = voxelwise_encoding_new(
                    X_tr, Y_tr, LAMBDAS, NFOLD_INNER, device=device
                )
                fold_W.append(W)
                fold_lam.append(Lam)
                pred = X_te @ W
                c = corr_vec(pred, Y_te)
                fold_corrs.append(c)
                print(
                    f"      Fold {fi+1}/{N_FOLDS} test=run{test_run}: "
                    f"mean r={np.nanmean(c):.4f}"
                )
                print(f"        λ hist: {summarize_hist(Lam, LAMBDAS)}")

            layer_r = np.mean(np.stack(fold_corrs), 0)
            all_layer_corrs.append(layer_r)
            W_mean = np.mean(np.stack(fold_W), 0)
            save_W_h5(W_mean, subj_out / f"W_layer{L+1}.h5")
            Lam_layer = np.stack(fold_lam).astype(np.float32)
            np.save(subj_out / f"Lambda_layer{L+1}.npy", Lam_layer)
            print(f"      Layer pooled λ: {summarize_hist(Lam_layer.ravel(), LAMBDAS)}")
            print(
                f"      Layer mean r={np.nanmean(layer_r):.4f} "
                f"(max {np.nanmax(layer_r):.4f}) [{time.time()-t_layer:.1f}s]"
            )

        if len(all_layer_corrs) == num_layers:
            final = np.stack(all_layer_corrs)
            sio.savemat(
                str(out_mat),
                {
                    "final_corr_matrix": final,
                    "n_folds": N_FOLDS,
                    "method": "LORO_extlambda_zscore_pc500",
                    "lambda_grid": LAMBDAS,
                },
                do_compression=True,
            )
            np.save(subj_out / "average_correlations_LORO.npy", final)
            save_cifti_dscalar(
                final, CIFTI_TEMPLATE,
                str(subj_out / "average_correlations_LORO.dscalar.nii"),
                feature_type,
            )
            print(
                f"\n    {subj} done in {time.time()-t_subj:.1f}s  "
                f"mean r={np.nanmean(final):.4f}"
            )

    print(f"\n=== {feature_type} ALL DONE in {(time.time()-t_total)/60:.1f} min ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=["Ahat", "E"], required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    run(args.feat, device)


if __name__ == "__main__":
    main()
