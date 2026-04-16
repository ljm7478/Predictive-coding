"""
Step 4: Direct Model Comparison between Ahat (prediction) and E (error)
        with Leave-One-Run-Out Cross-Validation.

Method — Direct comparison (no classical variance partitioning):

Unlike Popham et al. (2021) where visual and linguistic features are mixed
in a single movie stimulus and must be decomposed from each voxel's response,
PredNet produces Ahat (prediction) and E (error) as *already-separate* feature
streams. Classical variance partitioning (Unique = R2_Joint - R2_other) would
force most variance into "shared" because Ahat and E are structurally correlated
(E = ReLU(Input - Ahat)), making the unique contributions near-zero and
uninformative.

Instead, we fit two independent encoding models and compare them directly:

    Model_Ahat:  fMRI ~ Ahat features only   --> R2_Ahat
    Model_E:     fMRI ~ E    features only   --> R2_E

    PI_direct = (R2_Ahat - R2_E) / (R2_Ahat + R2_E)

    PI = +1 : purely prediction-driven (Ahat explains better)
    PI = -1 : purely error-driven      (E explains better)
    PI =  0 : equal explanatory power

This preserves the full information from both models without the
subtraction-induced cancellation of classical variance partitioning.

Conventions matching step01_run_encoding.py:
  * R^2 := square of mean Pearson r across LORO folds (negative r --> 0 before squaring)
  * Train-fold demeaning of regressors and fMRI (no zscoring) before ridge fit
  * Same lambda grid and inner-CV settings as step01

Outputs (per subject, under <BASE_SAVE_DIR>/direct_comparison_LORO/<subject>/):
  direct_comparison_LORO.mat / .npz
      R2_Ahat, R2_E, PI_direct, mean_r_Ahat, mean_r_E, R2_diff
      shape (num_layers, n_voxels)
  <metric>_LORO.dscalar.nii (CIFTI brain maps, 4 layers per file)
"""

import os
import numpy as np
import torch
import h5py
import scipy.io as sio
from pathlib import Path
import nibabel as nib
import time

from voxelwise_encoding_new import voxelwise_encoding_new


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Set to True to use log10-transformed features (from step0_*_log10_*.m)
USE_LOG10 = True

if USE_LOG10:
    BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4"
else:
    BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500/layer4"

FMRI_DATA_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new/"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Layer pairs (Ahat_L is matched against E_L for the same L)
ahat_layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"]
e_layer_names    = ["/E0",    "/E1",    "/E2",    "/E3"]
num_layers = len(ahat_layer_names)
assert num_layers == len(e_layer_names), "Ahat / E layer counts must match"

# LORO: All 8 runs used for cross-validation
all_runs = [1, 2, 3, 4, 5, 6, 7, 8]
n_folds = len(all_runs)  # 8-fold LORO

# Encoding parameters (matched to step01)
lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
nfold = 3  # Inner CV for lambda selection

# Gump subject IDs
subjects = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15",
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]

# GPU setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Output subdirectory under BASE_SAVE_DIR
OUTPUT_DIR_NAME = "direct_comparison_LORO"

# Numerical stability for the PI denominator
PI_DENOM_EPS = 1e-6


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def get_layer_key(layer_name):
    """Get the actual key name (with or without leading slash)."""
    return layer_name[1:] if layer_name.startswith('/') else layer_name


def load_fmri_mat(filepath):
    """Load fMRI data from MATLAB v7.3 .mat file (HDF5 format)."""
    fmri_data = {}
    with h5py.File(filepath, "r") as f:
        struct = f["fmri_data_per_run"]
        for run_name in struct.keys():
            data = struct[run_name][:].astype(np.float32)
            fmri_data[run_name] = data
    return fmri_data


def compute_correlation_vectorized(pred, actual):
    """Vectorized Pearson correlation across voxels (per-voxel column-wise)."""
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    actual_centered = actual - actual.mean(axis=0, keepdims=True)

    numerator = (pred_centered * actual_centered).sum(axis=0)
    denominator = np.sqrt(
        (pred_centered ** 2).sum(axis=0) * (actual_centered ** 2).sum(axis=0)
    )

    corrs = numerator / (denominator + 1e-10)
    corrs = np.nan_to_num(corrs, nan=0.0)
    return corrs


def save_cifti_dscalar(data, template_file, output_file, map_names):
    """Save as CIFTI dscalar with cortex only."""
    template = nib.load(template_file)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_maps, n_gray_data = data.shape
    brain_axis = template.header.get_axis(1)

    cortex_indices = []
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX' in name:
            cortex_indices.extend(range(slc.start, slc.stop))

    n_cortex = len(cortex_indices)

    if n_gray_data != n_cortex:
        print(f"    [WARN] Data ({n_gray_data}) != cortex ({n_cortex}); skipping CIFTI for {os.path.basename(output_file)}")
        return False

    cortex_brain_axis = brain_axis[np.array(cortex_indices)]
    scalar_axis = nib.cifti2.ScalarAxis(map_names)
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    return True


def load_layer_per_run(regressor_path, layer_key, all_runs):
    """
    Load one layer's regressor for every LORO run.

    Returns dict: {run_idx: ndarray (TRs, features)}
    Regressor h5 stores (features, TRs); we transpose to (TRs, features).
    """
    out = {}
    for run_idx in all_runs:
        r_path = regressor_path / f"run{run_idx}.h5"
        if not r_path.exists():
            raise FileNotFoundError(f"Missing regressor file: {r_path}")
        with h5py.File(r_path, "r") as f:
            if layer_key not in f:
                raise KeyError(f"Layer key '{layer_key}' not found in {r_path}")
            out[run_idx] = f[layer_key][:].T.astype(np.float32)
    return out


def fit_and_evaluate(X_train, fmri_train, X_test, fmri_test, alphas, nfold, device):
    """
    Fit ridge with internal CV and return per-voxel test-set Pearson r.
    All inputs must already be train-fold demeaned.
    """
    W, _, _ = voxelwise_encoding_new(
        X_train, fmri_train, alphas, nfold, device=device
    )
    pred_test = X_test @ W
    r = compute_correlation_vectorized(pred_test, fmri_test)
    return r


def signed_squared_to_R2(mean_r):
    """
    Convert mean correlation across folds to R^2.
    Negative correlations are set to 0 before squaring.
    """
    return np.where(mean_r > 0, mean_r ** 2, 0.0).astype(np.float32)


def compute_direct_PI(R2_ahat, R2_e, eps=PI_DENOM_EPS):
    """
    Direct Preference Index from individual model R^2 values.

    PI = (R2_Ahat - R2_E) / (R2_Ahat + R2_E)

    PI is set to 0 where the denominator is below `eps`
    (neither model has explanatory power).
    """
    denom = R2_ahat + R2_e
    pi = np.where(denom > eps, (R2_ahat - R2_e) / denom, 0.0)
    return pi.astype(np.float32)


# =============================================================================
# 3. PER-SUBJECT DIRECT COMPARISON
# =============================================================================

def direct_comparison_subject(subject, ahat_root, e_root, alphas):
    """
    Run LORO direct model comparison for one subject across all 4 layers.
    Fits Ahat-only and E-only models, compares R2 directly.
    """

    print(f"\n  Subject: {subject}")
    subj_start = time.time()

    # --- Load fMRI ---
    fmri_path = Path(FMRI_DATA_ROOT) / subject / f"{subject}_per_run_fmri.mat"
    if not fmri_path.exists():
        print(f"    [WARN] Missing fMRI for {subject}")
        return None

    fmri_data_struct = load_fmri_mat(fmri_path)
    print(f"    fMRI runs: {list(fmri_data_struct.keys())}")

    # Output containers, one row per layer
    R2_Ahat_all = []
    R2_E_all = []
    r_Ahat_all = []
    r_E_all = []

    for lay in range(num_layers):
        layer_start = time.time()
        ahat_key = get_layer_key(ahat_layer_names[lay])
        e_key    = get_layer_key(e_layer_names[lay])

        print(f"\n    Layer {lay + 1}/{num_layers}  ({ahat_key} vs {e_key})")

        # --- Preload this layer's regressors for all runs ---
        ahat_per_run = load_layer_per_run(ahat_root / "regressors", ahat_key, all_runs)
        e_per_run    = load_layer_per_run(e_root    / "regressors", e_key,    all_runs)

        n_feat_ahat = ahat_per_run[all_runs[0]].shape[1]
        n_feat_e    = e_per_run[all_runs[0]].shape[1]
        print(f"      Features: Ahat={n_feat_ahat}, E={n_feat_e}")

        # --- Align fMRI per run to regressor TR length ---
        fmri_per_run = {}
        for run_idx in all_runs:
            run_name = f"run{run_idx}"
            if run_name not in fmri_data_struct:
                raise RuntimeError(f"Missing fMRI {run_name} for {subject}")
            reg_trs = ahat_per_run[run_idx].shape[0]
            assert e_per_run[run_idx].shape[0] == reg_trs, \
                f"Ahat/E TR mismatch on run{run_idx}: {reg_trs} vs {e_per_run[run_idx].shape[0]}"
            fmri_run = fmri_data_struct[run_name]
            if fmri_run.shape[0] > reg_trs:
                fmri_run = fmri_run[:reg_trs, :]
            fmri_per_run[run_idx] = fmri_run

        n_voxels = fmri_per_run[all_runs[0]].shape[1]

        # --- LORO CV: fit two independent models per fold ---
        fold_r_Ahat = []
        fold_r_E = []

        for fold_idx, test_run in enumerate(all_runs):
            train_runs = [r for r in all_runs if r != test_run]

            # Concatenate training data
            X_ahat_train = np.concatenate([ahat_per_run[r] for r in train_runs], axis=0)
            X_e_train    = np.concatenate([e_per_run[r]    for r in train_runs], axis=0)
            fmri_train   = np.concatenate([fmri_per_run[r] for r in train_runs], axis=0)

            X_ahat_test = ahat_per_run[test_run]
            X_e_test    = e_per_run[test_run]
            fmri_test   = fmri_per_run[test_run]

            # --- Demean using training statistics ---
            ahat_mean = X_ahat_train.mean(axis=0, keepdims=True)
            X_ahat_train = X_ahat_train - ahat_mean
            X_ahat_test  = X_ahat_test  - ahat_mean

            e_mean = X_e_train.mean(axis=0, keepdims=True)
            X_e_train = X_e_train - e_mean
            X_e_test  = X_e_test  - e_mean

            fmri_mean = fmri_train.mean(axis=0, keepdims=True)
            fmri_train = fmri_train - fmri_mean
            fmri_test  = fmri_test  - fmri_mean

            # --- Fit each model independently ---
            r_a = fit_and_evaluate(
                X_ahat_train, fmri_train, X_ahat_test, fmri_test, alphas, nfold, device
            )
            r_e = fit_and_evaluate(
                X_e_train, fmri_train, X_e_test, fmri_test, alphas, nfold, device
            )

            fold_r_Ahat.append(r_a)
            fold_r_E.append(r_e)

            print(f"      Fold {fold_idx + 1}/{n_folds} (test=run{test_run}): "
                  f"r_Ahat={np.nanmean(r_a):.4f}  r_E={np.nanmean(r_e):.4f}")

        # --- Average correlations across folds ---
        mean_r_Ahat = np.mean(np.stack(fold_r_Ahat), axis=0)
        mean_r_E    = np.mean(np.stack(fold_r_E),    axis=0)

        # --- Convert to R^2 ---
        R2_Ahat = signed_squared_to_R2(mean_r_Ahat)
        R2_E    = signed_squared_to_R2(mean_r_E)

        r_Ahat_all.append(mean_r_Ahat.astype(np.float32))
        r_E_all.append(mean_r_E.astype(np.float32))
        R2_Ahat_all.append(R2_Ahat)
        R2_E_all.append(R2_E)

        layer_time = time.time() - layer_start
        print(f"      Layer {lay + 1} done in {layer_time:.1f}s | "
              f"R2 means: Ahat={R2_Ahat.mean():.4f}  E={R2_E.mean():.4f}")

    # =========================================================================
    # Stack across layers and compute direct PI
    # =========================================================================
    R2_Ahat_arr = np.stack(R2_Ahat_all)   # (num_layers, n_voxels)
    R2_E_arr    = np.stack(R2_E_all)

    # Direct preference index: compare R2 values without subtraction
    PI_direct = compute_direct_PI(R2_Ahat_arr, R2_E_arr)

    # R2 difference (signed, for visualization)
    R2_diff = (R2_Ahat_arr - R2_E_arr).astype(np.float32)

    print(f"\n    Subject {subject} done in {time.time() - subj_start:.1f}s")
    return dict(
        R2_Ahat=R2_Ahat_arr,
        R2_E=R2_E_arr,
        PI_direct=PI_direct,
        R2_diff=R2_diff,
        mean_r_Ahat=np.stack(r_Ahat_all),
        mean_r_E=np.stack(r_E_all),
    )


# =============================================================================
# 4. SAVE RESULTS PER SUBJECT
# =============================================================================

def save_results(results, out_dir, subject):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- .mat ----
    mat_path = out_dir / "direct_comparison_LORO.mat"
    sio.savemat(
        str(mat_path),
        {
            "R2_Ahat":     results["R2_Ahat"],
            "R2_E":        results["R2_E"],
            "PI_direct":   results["PI_direct"],
            "R2_diff":     results["R2_diff"],
            "mean_r_Ahat": results["mean_r_Ahat"],
            "mean_r_E":    results["mean_r_E"],
            "layer_names_Ahat": np.array(ahat_layer_names, dtype=object),
            "layer_names_E":    np.array(e_layer_names, dtype=object),
            "n_folds": n_folds,
            "method": "LORO_direct_comparison",
        },
        do_compression=True,
    )
    print(f"    Saved: {mat_path.name}")

    # ---- .npz ----
    npz_path = out_dir / "direct_comparison_LORO.npz"
    np.savez(
        str(npz_path),
        R2_Ahat=results["R2_Ahat"],
        R2_E=results["R2_E"],
        PI_direct=results["PI_direct"],
        R2_diff=results["R2_diff"],
        mean_r_Ahat=results["mean_r_Ahat"],
        mean_r_E=results["mean_r_E"],
    )
    print(f"    Saved: {npz_path.name}")

    # ---- CIFTI dscalars ----
    cifti_metrics = [
        ("R2_Ahat",    "R2_Ahat"),
        ("R2_E",       "R2_E"),
        ("PI_direct",  "PI_direct"),
        ("R2_diff",    "R2_diff"),
        ("mean_r_Ahat", "mean_r_Ahat"),
        ("mean_r_E",    "mean_r_E"),
    ]
    for fname_stem, key in cifti_metrics:
        data = results[key]  # (num_layers, n_voxels)
        map_names = [f"{fname_stem}_layer{i + 1}" for i in range(num_layers)]
        cifti_out = str(out_dir / f"{fname_stem}_LORO.dscalar.nii")
        save_cifti_dscalar(data, CIFTI_TEMPLATE, cifti_out, map_names)
    print(f"    Saved: {len(cifti_metrics)} CIFTI dscalar files")


# =============================================================================
# 5. MAIN PIPELINE
# =============================================================================

def run_direct_comparison(force=False):
    alphas = np.array(lambdas, dtype=np.float32)
    total_start = time.time()

    print(f"\n{'=' * 70}")
    print(f" Gump DIRECT COMPARISON (LORO CV): Ahat vs E")
    print(f" {n_folds}-fold CV, {num_layers} layers, {len(subjects)} subjects")
    print(f" Method: PI = (R2_Ahat - R2_E) / (R2_Ahat + R2_E)")
    print(f"{'=' * 70}")

    ahat_root = Path(BASE_SAVE_DIR) / "Ahat"
    e_root    = Path(BASE_SAVE_DIR) / "E"

    # --- Sanity: regressor dirs exist ---
    for tag, root in [("Ahat", ahat_root), ("E", e_root)]:
        reg_dir = root / "regressors"
        if not reg_dir.exists():
            print(f"  [ERROR] {tag} regressors not found at: {reg_dir}")
            return

    out_root = Path(BASE_SAVE_DIR) / OUTPUT_DIR_NAME
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\n  Output root: {out_root}")

    for subj_idx, subject in enumerate(subjects):
        out_dir = out_root / subject
        bundle_path = out_dir / "direct_comparison_LORO.mat"

        if bundle_path.exists() and not force:
            print(f"\n  [{subj_idx + 1}/{len(subjects)}] {subject} already done. Skipping. (Use --force to re-run.)")
            continue

        print(f"\n  [{subj_idx + 1}/{len(subjects)}] Processing {subject} ...")
        results = direct_comparison_subject(subject, ahat_root, e_root, alphas)

        if results is None:
            print(f"    [SKIP] {subject}")
            continue

        save_results(results, out_dir, subject)

        # Per-subject preview
        print(f"\n    Per-layer preview ({subject}):")
        for lay in range(num_layers):
            print(
                f"      L{lay + 1}: "
                f"R2_Ahat={results['R2_Ahat'][lay].mean():.4f}  "
                f"R2_E={results['R2_E'][lay].mean():.4f}  "
                f"R2_diff={results['R2_diff'][lay].mean():.4f}  "
                f"PI={results['PI_direct'][lay].mean():+.4f}"
            )

    total_time = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"=== ALL DONE in {total_time / 60:.1f} minutes ===")
    print(f"{'=' * 70}")


# =============================================================================
# 6. QUICK SHAPE CHECK
# =============================================================================

def quick_shape_check():
    """Sanity check that Ahat and E regressors exist and have matching TR counts."""
    print("\n" + "=" * 70)
    print(" QUICK SHAPE CHECK - Direct Comparison (LORO)")
    print("=" * 70)

    ahat_reg = Path(BASE_SAVE_DIR) / "Ahat" / "regressors"
    e_reg    = Path(BASE_SAVE_DIR) / "E"    / "regressors"

    print(f"\n  Ahat regressor dir: {ahat_reg}  exists={ahat_reg.exists()}")
    print(f"  E    regressor dir: {e_reg}     exists={e_reg.exists()}")

    if not (ahat_reg.exists() and e_reg.exists()):
        return

    for run_idx in all_runs:
        ahat_path = ahat_reg / f"run{run_idx}.h5"
        e_path    = e_reg    / f"run{run_idx}.h5"
        if not (ahat_path.exists() and e_path.exists()):
            print(f"  [MISSING] run{run_idx}")
            continue
        with h5py.File(ahat_path, "r") as fa, h5py.File(e_path, "r") as fe:
            print(f"\n  run{run_idx}.h5")
            for lay in range(num_layers):
                a_key = get_layer_key(ahat_layer_names[lay])
                e_key = get_layer_key(e_layer_names[lay])
                a_shape = fa[a_key].shape if a_key in fa else None
                e_shape = fe[e_key].shape if e_key in fe else None
                tr_match = (
                    a_shape is not None and e_shape is not None
                    and a_shape[1] == e_shape[1]
                )
                marker = "OK " if tr_match else "!! "
                print(f"    {marker}{a_key} {a_shape}   |   {e_key} {e_shape}")

    # fMRI sanity
    print("\n  Checking fMRI structure (sub-01)...")
    fmri_path = Path(FMRI_DATA_ROOT) / "sub-01" / "sub-01_per_run_fmri.mat"
    if fmri_path.exists():
        with h5py.File(fmri_path, "r") as f:
            struct = f["fmri_data_per_run"]
            for run_name in struct.keys():
                print(f"    {run_name}: {struct[run_name].shape}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gump LORO Direct Comparison (Ahat vs E)")
    parser.add_argument("--check", action="store_true", help="Quick shape check only")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if direct_comparison_LORO.mat already exists")
    args = parser.parse_args()

    if args.check:
        quick_shape_check()
    else:
        run_direct_comparison(force=args.force)
