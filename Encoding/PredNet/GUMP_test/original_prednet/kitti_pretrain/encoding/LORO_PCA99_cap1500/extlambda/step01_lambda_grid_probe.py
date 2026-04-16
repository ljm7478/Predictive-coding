"""
Probe an extended log-spaced λ grid on a single subject (sub-01) for both
Ahat and E. Goal: check whether the pile-up at the current grid's upper
boundary (λ=0.9) migrates to larger λ when the grid is widened, and
whether mean r improves.

Writes to a separate '<BASE>_extlambda/' directory so existing results
are untouched.
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voxelwise_encoding_new import voxelwise_encoding_new


BASE_IN = Path(
    "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/"
    "PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/"
    "kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4"
)
BASE_OUT = BASE_IN.parent / (BASE_IN.name + "_extlambda")
FMRI_DATA_ROOT = Path(
    "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/"
    "PredNet/Titanic_trained/GUMP_test/fmri_new/"
)

SUBJECT = "sub-01"
ALL_RUNS = [1, 2, 3, 4, 5, 6, 7, 8]
NFOLD = 3

# Extended log-spaced grid — spans 4 decades.
LAMBDAS = np.array(
    [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0], dtype=np.float32
)


def get_layer_key(name):
    return name[1:] if name.startswith("/") else name


def load_fmri_mat(fp):
    out = {}
    with h5py.File(fp, "r") as f:
        s = f["fmri_data_per_run"]
        for r in s.keys():
            out[r] = s[r][:].astype(np.float32)
    return out


def summarize_hist(arr):
    uniq, cnt = np.unique(arr, return_counts=True)
    pct = cnt / cnt.sum() * 100
    return ", ".join(f"{u:.3g}:{p:.1f}%" for u, p in zip(uniq, pct))


def run_one(feature_type, device):
    if feature_type == "Ahat":
        layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"]
    else:
        layer_names = ["/E0", "/E1", "/E2", "/E3"]
    num_layers = len(layer_names)

    in_root = BASE_IN / feature_type
    out_root = BASE_OUT / feature_type / SUBJECT
    out_root.mkdir(parents=True, exist_ok=True)

    reg_path = in_root / "regressors"
    fmri_path = FMRI_DATA_ROOT / SUBJECT / f"{SUBJECT}_per_run_fmri.mat"

    print(f"\n{'='*70}\n {feature_type}  subject={SUBJECT}\n{'='*70}")
    print(f" λ grid ({len(LAMBDAS)}): {list(LAMBDAS)}")
    print(f" input  : {in_root}")
    print(f" output : {out_root}")

    fmri_struct = load_fmri_mat(fmri_path)

    regs = {L: {} for L in range(num_layers)}
    fmri = {}
    for r in ALL_RUNS:
        rp = reg_path / f"run{r}.h5"
        with h5py.File(rp, "r") as f:
            for L in range(num_layers):
                k = get_layer_key(layer_names[L])
                regs[L][r] = f[k][:].T.astype(np.float32)   # (TR, feat)
        trs = regs[0][r].shape[0]
        fm = fmri_struct[f"run{r}"]
        if fm.shape[0] > trs:
            fm = fm[:trs, :]
        fmri[r] = fm

    n_voxels = fmri[ALL_RUNS[0]].shape[1]
    layer_means = []

    for L in range(num_layers):
        t0 = time.time()
        print(f"\n  Layer {L+1}/{num_layers} ({layer_names[L]})")
        fold_corrs = []
        fold_lambdas = []

        for fi, test_run in enumerate(ALL_RUNS):
            train_runs = [r for r in ALL_RUNS if r != test_run]
            X_tr = np.concatenate([regs[L][r] for r in train_runs], 0)
            Y_tr = np.concatenate([fmri[r] for r in train_runs], 0)
            X_te = regs[L][test_run]
            Y_te = fmri[test_run]

            mx = X_tr.mean(0, keepdims=True); X_tr -= mx; X_te -= mx
            my = Y_tr.mean(0, keepdims=True); Y_tr -= my; Y_te -= my

            W, _, Lam = voxelwise_encoding_new(
                X_tr, Y_tr, LAMBDAS, NFOLD, device=device
            )
            pred = X_te @ W
            pc = pred - pred.mean(0, keepdims=True)
            ac = Y_te - Y_te.mean(0, keepdims=True)
            num = (pc * ac).sum(0)
            den = np.sqrt((pc**2).sum(0) * (ac**2).sum(0))
            corrs = np.nan_to_num(num / (den + 1e-10))

            fold_corrs.append(corrs)
            fold_lambdas.append(Lam)
            print(
                f"    Fold {fi+1}/8 test=run{test_run}: "
                f"mean r={np.nanmean(corrs):.4f}  λ hist: {summarize_hist(Lam)}"
            )

        Lam_layer = np.stack(fold_lambdas).astype(np.float32)  # (8, V)
        np.save(out_root / f"Lambda_layer{L+1}.npy", Lam_layer)
        layer_r = np.mean(np.stack(fold_corrs), 0)
        layer_means.append(layer_r)
        print(
            f"    Layer pooled λ hist: {summarize_hist(Lam_layer.ravel())}\n"
            f"    Layer mean r={np.nanmean(layer_r):.4f} "
            f"(max {np.nanmax(layer_r):.4f}) [{time.time()-t0:.1f}s]"
        )

    final = np.stack(layer_means)  # (4, V)
    np.save(out_root / "average_correlations_LORO.npy", final)
    print(f"\n  {feature_type} overall mean r={np.nanmean(final):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["Ahat", "E"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    t0 = time.time()
    for f in args.feats:
        run_one(f, device)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
