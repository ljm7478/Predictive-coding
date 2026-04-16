"""
Step 1: GPU-accelerated Voxel-wise Encoding for Gump Dataset
Leave-One-Run-Out (LORO) Cross-Validation Version
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

FEATURE_TYPE = "Ahat"  # "E" for Error signals, "Ahat" for Predictions

# Set to True to use log10-transformed features (from step0_*_log10_*.m)
USE_LOG10 = True

if USE_LOG10:
    BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4"
else:
    BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/LORO_PCA99cap1500/layer4"
FMRI_DATA_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new/"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Layer configuration based on feature type
if FEATURE_TYPE == "Ahat":
    layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"] 
else:  # "E"
    layer_names = ["/E0", "/E1", "/E2", "/E3"]

num_layers = len(layer_names)

# LORO: All 8 runs used for cross-validation
all_runs = [1, 2, 3, 4, 5, 6, 7, 8]
n_folds = len(all_runs)  # 8-fold LORO

# Encoding parameters
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


def save_W_h5(W, filepath):
    """Save weights matrix as HDF5."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(filepath, "w") as f:
        f.create_dataset("W", data=W, compression="gzip")


def compute_correlation_vectorized(pred, actual):
    """Vectorized Pearson correlation across voxels."""
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    actual_centered = actual - actual.mean(axis=0, keepdims=True)
    
    numerator = (pred_centered * actual_centered).sum(axis=0)
    denominator = np.sqrt(
        (pred_centered ** 2).sum(axis=0) * (actual_centered ** 2).sum(axis=0)
    )
    
    corrs = numerator / (denominator + 1e-10)
    corrs = np.nan_to_num(corrs, nan=0.0)
    return corrs


def save_cifti_dscalar(data, template_file, output_file):
    """Save as CIFTI dscalar with cortex only."""
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_gray_data = data.shape
    print(f"    CIFTI input: {data.shape} (layers x grayordinates)")
    
    brain_axis = template.header.get_axis(1)
    
    cortex_indices = []
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX' in name:
            cortex_indices.extend(range(slc.start, slc.stop))
    
    n_cortex = len(cortex_indices)
    print(f"    Cortical vertices: {n_cortex}")
    
    if n_gray_data != n_cortex:
        print(f"    WARNING: Data ({n_gray_data}) != cortex ({n_cortex}), saving .mat only")
        return False
    
    cortex_brain_axis = brain_axis[np.array(cortex_indices)]
    scalar_axis = nib.cifti2.ScalarAxis([f'{FEATURE_TYPE}{i}' for i in range(n_maps)])
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    
    print(f"    Saved: {output_file}")
    return True


def check_regressor_shapes_loro(regressor_path, layer_names, all_runs):
    """Check regressor shapes for LORO CV."""
    print("\n  [Shape Check] Checking regressor dimensions for LORO...")
    
    shapes_per_layer = {layer: [] for layer in layer_names}
    
    for run_idx in all_runs:
        r_path = regressor_path / f"run{run_idx}.h5"
        
        if not r_path.exists():
            print(f"    [ERROR] Missing regressor file: {r_path}")
            return False, None
        
        with h5py.File(r_path, "r") as f:
            for layer_name in layer_names:
                key = layer_name if layer_name in f else get_layer_key(layer_name)
                if key not in f:
                    print(f"    [ERROR] Layer {layer_name} not found in {r_path}")
                    return False, None
                
                shape = f[key].shape
                shapes_per_layer[layer_name].append((run_idx, shape))
    
    # Print shape summary
    print("\n    Regressor shapes (MATLAB format: features x TRs):")
    n_features = None
    total_trs = 0
    trs_per_run = []
    
    for layer_name in layer_names[:1]:  # Just check first layer
        shapes = shapes_per_layer[layer_name]
        print(f"      {layer_name}:")
        
        for run_idx, shape in shapes:
            print(f"        run{run_idx}.h5: {shape}")
            trs_per_run.append(shape[1])
            
            if n_features is None:
                n_features = shape[0]
    
    total_trs = sum(trs_per_run)
    min_train_trs = total_trs - max(trs_per_run)  # Worst case: leave out longest run
    
    print(f"\n    LORO Summary:")
    print(f"      Features (PCA components): {n_features}")
    print(f"      Total TRs across all runs: {total_trs}")
    print(f"      Min training TRs (worst fold): {min_train_trs}")
    print(f"      Features/min_train_TRs ratio: {n_features/min_train_trs:.3f}")
    
    if n_features > min_train_trs:
        print(f"\n    [ERROR] Features ({n_features}) > Min Training TRs ({min_train_trs})")
        return False, None
    else:
        print(f"\n    [OK] Features < Training TRs in all folds")
    
    return True, n_features


# =============================================================================
# 3. MAIN ENCODING LOOP - LORO VERSION
# =============================================================================

def run_Gump_encoding_LORO():
    """Main encoding pipeline with Leave-One-Run-Out CV."""
    
    alphas = np.array(lambdas, dtype=np.float32)
    total_start = time.time()
    
    print(f"\n{'='*70}")
    print(f" Gump ENCODING (LORO CV): {FEATURE_TYPE}")
    print(f" {n_folds}-fold Cross-Validation")
    print(f"{'='*70}")
    
    SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE
    regressor_path = SAVE_ROOT / "regressors"
    
    # --- Check if regressors exist ---
    if not regressor_path.exists():
        print(f"  [ERROR] Regressors not found at: {regressor_path}")
        return
    
    # --- Validate regressor shapes ---
    is_valid, n_features = check_regressor_shapes_loro(
        regressor_path, layer_names, all_runs
    )
    
    if not is_valid:
        print(f"\n  [SKIP] Shape validation failed")
        return
    
    print(f"\n  Starting LORO encoding with {n_features} features...")
    
    # --- Process each subject ---
    for subj_idx, subject in enumerate(subjects):
        subject_start = time.time()
        
        subject_save_dir = SAVE_ROOT / subject
        subject_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already done
        out_mat = subject_save_dir / "average_correlations_LORO.mat"
        if out_mat.exists():
            print(f"\n  {subject} already done. Skipping.")
            continue
        
        print(f"\n  Subject: {subject} ({subj_idx + 1}/{len(subjects)})")
        
        # --- Load fMRI ---
        fmri_path = Path(FMRI_DATA_ROOT) / subject / f"{subject}_per_run_fmri.mat"
        if not fmri_path.exists():
            print(f"    [WARNING] Missing fMRI for {subject}")
            continue
        
        fmri_data_struct = load_fmri_mat(fmri_path)
        print(f"    fMRI runs available: {list(fmri_data_struct.keys())}")
        
        # --- Preload all regressors and fMRI (aligned) ---
        print(f"\n    Preloading data for all runs...")
        
        regressors_per_run = {lay: {} for lay in range(num_layers)}
        fmri_per_run = {}
        
        for run_idx in all_runs:
            run_name = f"run{run_idx}"
            
            # Load regressor
            r_path = regressor_path / f"run{run_idx}.h5"
            with h5py.File(r_path, "r") as f:
                for lay in range(num_layers):
                    layer_key = get_layer_key(layer_names[lay])
                    # MATLAB: (features, TRs) -> transpose to (TRs, features)
                    reg_data = f[layer_key][:].T.astype(np.float32)
                    regressors_per_run[lay][run_idx] = reg_data
            
            reg_trs = regressors_per_run[0][run_idx].shape[0]
            
            # Load and align fMRI
            if run_name not in fmri_data_struct:
                print(f"    [WARNING] Missing fMRI {run_name}")
                continue
            
            fmri_run = fmri_data_struct[run_name]
            fmri_trs = fmri_run.shape[0]
            
            # Truncate fMRI to match regressor
            if fmri_trs > reg_trs:
                fmri_run = fmri_run[:reg_trs, :]
            
            fmri_per_run[run_idx] = fmri_run
            print(f"      {run_name}: {fmri_run.shape[0]} TRs x {fmri_run.shape[1]} voxels")
        
        n_voxels = fmri_per_run[all_runs[0]].shape[1]
        
        # --- Layer Loop ---
        all_layer_corrs = []  # (num_layers, n_voxels)
        all_layer_W_mean = []  # For saving averaged W
        
        for lay in range(num_layers):
            layer_start = time.time()
            layer_name = layer_names[lay]
            print(f"\n    Layer {lay + 1}/{num_layers} ({layer_name})")
            
            # --- LORO CV: collect predictions for each fold ---
            all_fold_corrs = []  # (n_folds, n_voxels)
            all_fold_W = []  # For averaging W across folds

            for fold_idx, test_run in enumerate(all_runs):
                train_runs = [r for r in all_runs if r != test_run]

                # Concatenate training data
                X_train_list = [regressors_per_run[lay][r] for r in train_runs]
                fmri_train_list = [fmri_per_run[r] for r in train_runs]

                X_train = np.concatenate(X_train_list, axis=0)
                fmri_train = np.concatenate(fmri_train_list, axis=0)

                # Test data
                X_test = regressors_per_run[lay][test_run]
                fmri_test = fmri_per_run[test_run]

                # --- Demean using training statistics ---
                # Equivalent to Wen et al. Eq. 3 bias term (b_v)
                # Demeaning + ridge without intercept = unpenalized intercept
                reg_mean = X_train.mean(axis=0, keepdims=True)
                X_train = X_train - reg_mean
                X_test = X_test - reg_mean

                fmri_mean = fmri_train.mean(axis=0, keepdims=True)
                fmri_train = fmri_train - fmri_mean
                fmri_test = fmri_test - fmri_mean

                # Train encoding model
                W, _, _ = voxelwise_encoding_new(
                    X_train, fmri_train, alphas, nfold, device=device
                )

                all_fold_W.append(W)

                # Predict and correlate
                pred_fmri = X_test @ W
                fold_corrs = compute_correlation_vectorized(pred_fmri, fmri_test)
                all_fold_corrs.append(fold_corrs)

                print(f"      Fold {fold_idx+1}/{n_folds} (test=run{test_run}): "
                      f"train={X_train.shape[0]} TRs, test={X_test.shape[0]} TRs, "
                      f"mean r = {np.nanmean(fold_corrs):.4f}")
            
            # Average across folds
            layer_corrs = np.mean(np.stack(all_fold_corrs), axis=0)
            all_layer_corrs.append(layer_corrs)
            
            # Average W across folds and save
            W_mean = np.mean(np.stack(all_fold_W), axis=0)
            save_W_h5(W_mean, subject_save_dir / f"W_layer{lay + 1}.h5")
            
            layer_time = time.time() - layer_start
            print(f"      Layer avg (across {n_folds} folds): r = {np.nanmean(layer_corrs):.4f} "
                  f"(max: {np.nanmax(layer_corrs):.4f}) [{layer_time:.1f}s]")
        
        # --- Save Results ---
        if len(all_layer_corrs) == num_layers:
            final_corr_matrix = np.stack(all_layer_corrs)  # (4, 59412)
            print(f"\n    Final correlation matrix: {final_corr_matrix.shape}")
            
            # Save .mat
            sio.savemat(
                str(subject_save_dir / "average_correlations_LORO.mat"),
                {
                    "final_corr_matrix": final_corr_matrix,
                    "n_folds": n_folds,
                    "method": "LORO"
                },
                do_compression=True
            )
            print(f"    Saved: average_correlations_LORO.mat")
            
            # Save .npy
            np.save(
                str(subject_save_dir / "average_correlations_LORO.npy"),
                final_corr_matrix
            )
            print(f"    Saved: average_correlations_LORO.npy")
            
            # Save CIFTI
            cifti_output = str(subject_save_dir / "average_correlations_LORO.dscalar.nii")
            save_cifti_dscalar(final_corr_matrix, CIFTI_TEMPLATE, cifti_output)
            
            subject_time = time.time() - subject_start
            print(f"\n    Subject {subject} complete in {subject_time:.1f}s")
            print(f"    Mean r across layers: {np.nanmean(final_corr_matrix):.4f}")
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"=== ALL DONE in {total_time/60:.1f} minutes ===")
    print(f"{'='*70}")


# =============================================================================
# 4. QUICK SHAPE CHECK
# =============================================================================

def quick_shape_check():
    """Quick validation before encoding."""
    print("\n" + "="*70)
    print(f" QUICK SHAPE CHECK - Gump LORO ({FEATURE_TYPE})")
    print("="*70)
    
    SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE
    regressor_path = SAVE_ROOT / "regressors"
    
    if not regressor_path.exists():
        print(f"  [NOT FOUND] {regressor_path}")
        return
    
    print(f"\n  Regressor path: {regressor_path}")
    print(f"  Feature type: {FEATURE_TYPE}")
    print(f"  All runs (LORO): {all_runs}")
    
    check_regressor_shapes_loro(regressor_path, layer_names, all_runs)
    
    # Check fMRI
    print("\n  Checking fMRI structure...")
    test_subject = subjects[0]
    fmri_path = Path(FMRI_DATA_ROOT) / test_subject / f"{test_subject}_per_run_fmri.mat"
    
    if fmri_path.exists():
        with h5py.File(fmri_path, "r") as f:
            struct = f["fmri_data_per_run"]
            print(f"  Available runs: {list(struct.keys())}")
            for run_name in struct.keys():
                shape = struct[run_name].shape
                print(f"    {run_name}: {shape}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gump LORO Encoding Pipeline")
    parser.add_argument("--check", action="store_true", help="Quick shape check only")
    args = parser.parse_args()
    
    if args.check:
        quick_shape_check()
    else:
        run_Gump_encoding_LORO()