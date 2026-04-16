"""
Step 1: GPU-accelerated Voxel-wise Encoding for Gump Dataset
Adapted from Gump encoding pipeline
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

BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4"
FMRI_DATA_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new/"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

# Layer configuration based on feature type
if FEATURE_TYPE == "Ahat":
    layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"] 
else:  # "E"
    layer_names = ["/E0", "/E1", "/E2", "/E3"]

num_layers = len(layer_names)

train_runs = [1, 2, 3, 4]
test_runs = [5, 6, 7, 8]

# Encoding parameters (matching MATLAB)
lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
nfold = 3

# Gump subject IDs (actual naming from your preprocessing script)
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
    """
    Load fMRI data from MATLAB v7.3 .mat file (HDF5 format).
    
    MATLAB saves as: fmri_data_per_run.run1, .run2, etc.
    MATLAB format: (59412 voxels, TRs)
    h5py reads as: (TRs, 59412 voxels) due to row/column major difference
    So NO transpose needed!
    """
    fmri_data = {}
    with h5py.File(filepath, "r") as f:
        struct = f["fmri_data_per_run"]
        for run_name in struct.keys():
            # h5py already reads as (TRs, voxels) due to MATLAB/Python ordering
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
    """
    Save as CIFTI dscalar with cortex only.
    
    Input data: (n_maps, n_grayordinates) = (4, 59412)
    """
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_gray_data = data.shape
    print(f"    CIFTI input: {data.shape} (layers x grayordinates)")
    
    # Get brain axis from template
    brain_axis = template.header.get_axis(1)
    
    # Find cortical indices only
    cortex_indices = []
    for name, slc, bm in brain_axis.iter_structures():
        if 'CORTEX' in name:
            cortex_indices.extend(range(slc.start, slc.stop))
    
    n_cortex = len(cortex_indices)
    print(f"    Cortical vertices: {n_cortex}")
    
    # Check dimension match
    if n_gray_data != n_cortex:
        print(f"    WARNING: Data ({n_gray_data}) != cortex ({n_cortex}), saving .mat only")
        return False
    
    # Create cortex-only brain axis
    cortex_brain_axis = brain_axis[np.array(cortex_indices)]
    
    # Create scalar axis with layer names (use actual feature type)
    scalar_axis = nib.cifti2.ScalarAxis([f'{FEATURE_TYPE}{i}' for i in range(n_maps)])
    
    # Create new header
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    
    # Create and save image
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    
    print(f"    Saved: {output_file}")
    return True


def check_regressor_shapes(regressor_path, layer_names, train_runs, test_runs):
    """Check regressor shapes before encoding."""
    print("\n  [Shape Check] Checking regressor dimensions...")
    
    all_runs = train_runs + test_runs
    shapes_per_layer = {layer: [] for layer in layer_names}
    
    for run_idx in all_runs:
        r_path = regressor_path / f"run{run_idx}.h5"
        
        if not r_path.exists():
            print(f"    [ERROR] Missing regressor file: {r_path}")
            return False, None
        
        with h5py.File(r_path, "r") as f:
            for layer_name in layer_names:
                # Try with and without leading slash
                key = layer_name if layer_name in f else get_layer_key(layer_name)
                if key not in f:
                    print(f"    [ERROR] Layer {layer_name} not found in {r_path}")
                    print(f"    Available keys: {list(f.keys())}")
                    return False, None
                
                shape = f[key].shape
                shapes_per_layer[layer_name].append((run_idx, shape))
    
    # Print shape summary
    print("\n    Regressor shapes (MATLAB format: features x TRs):")
    n_features = None
    
    for layer_name in layer_names:
        shapes = shapes_per_layer[layer_name]
        print(f"      {layer_name}:")
        
        for run_idx, shape in shapes:
            run_type = "train" if run_idx in train_runs else "test"
            print(f"        run{run_idx}.h5 ({run_type}): {shape}")
            
            if n_features is None:
                n_features = shape[0]
            elif shape[0] != n_features:
                print(f"    [WARNING] Inconsistent feature count: {shape[0]} vs {n_features}")
    
    # Calculate total training TRs
    train_trs = sum(
        shapes_per_layer[layer_names[0]][i][1][1]
        for i, (run_idx, _) in enumerate(shapes_per_layer[layer_names[0]])
        if run_idx in train_runs
    )
    
    print(f"\n    Summary:")
    print(f"      Features (PCA components): {n_features}")
    print(f"      Total training TRs: {train_trs}")
    print(f"      Features/TRs ratio: {n_features/train_trs:.3f}")
    
    if n_features > train_trs:
        print(f"\n    [ERROR] Features ({n_features}) > Training TRs ({train_trs})")
        print(f"    This will cause overfitting!")
        return False, None
    elif n_features > train_trs * 0.5:
        print(f"\n    [WARNING] Features/TRs ratio > 0.5, may have some overfitting")
    else:
        print(f"\n    [OK] Features < TRs, encoding should work properly")
    
    return True, n_features


# =============================================================================
# 3. MAIN ENCODING LOOP
# =============================================================================

def run_Gump_encoding():
    """Main encoding pipeline for Gump dataset."""
    
    alphas = np.array(lambdas, dtype=np.float32)
    total_start = time.time()
    
    print(f"\n{'='*70}")
    print(f" Gump ENCODING: {FEATURE_TYPE}")
    print(f"{'='*70}")
    
    SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE
    regressor_path = SAVE_ROOT / "regressors"
    
    # --- Check if regressors exist ---
    if not regressor_path.exists():
        print(f"  [ERROR] Regressors not found at: {regressor_path}")
        return
    
    # --- Validate regressor shapes ---
    is_valid, n_features = check_regressor_shapes(
        regressor_path, layer_names, train_runs, test_runs
    )
    
    if not is_valid:
        print(f"\n  [SKIP] Shape validation failed")
        return
    
    print(f"\n  Starting encoding with {n_features} features...")
    
    # --- Process each subject ---
    for subj_idx, subject in enumerate(subjects):
        subject_start = time.time()
        
        subject_save_dir = SAVE_ROOT / subject
        subject_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already done
        out_mat = subject_save_dir / "average_correlations.mat"
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
        
        # --- Concatenate Training fMRI (with TR alignment) ---
        # Regressors have fewer TRs due to HRF convolution edge trimming
        # Load regressor lengths first to align fMRI
        fmri_train_list = []
        valid_train = True
        
        for run_idx in train_runs:
            run_name = f"run{run_idx}"
            if run_name not in fmri_data_struct:
                print(f"    [WARNING] Missing fMRI {run_name}")
                valid_train = False
                break
            
            # Get regressor length for this run (use first layer)
            r_path = regressor_path / f"run{run_idx}.h5"
            with h5py.File(r_path, "r") as f:
                first_layer_key = get_layer_key(layer_names[0])
                reg_trs = f[first_layer_key].shape[1]
            
            fmri_run = fmri_data_struct[run_name]
            fmri_trs = fmri_run.shape[0]
            
            # Truncate fMRI to match regressor (trim from end)
            if fmri_trs > reg_trs:
                fmri_run = fmri_run[:reg_trs, :]
                print(f"    {run_name}: {fmri_trs} -> {reg_trs} TRs (truncated to match regressors)")
            else:
                print(f"    {run_name}: {fmri_trs} TRs")
            
            fmri_train_list.append(fmri_run)
        
        if not valid_train:
            continue
        
        fmri_train_data = np.concatenate(fmri_train_list, axis=0)
        print(f"\n    Concatenated fMRI (train): {fmri_train_data.shape}")
        
        # --- Layer Loop ---
        all_subject_corrs = []
        
        for lay in range(num_layers):
            layer_start = time.time()
            layer_name = layer_names[lay]
            layer_key = get_layer_key(layer_name)
            print(f"\n    Layer {lay + 1}/{num_layers} ({layer_name})")
            
            # --- Load Training Regressors ---
            train_reg_list = []
            for run_idx in train_runs:
                r_path = regressor_path / f"run{run_idx}.h5"
                with h5py.File(r_path, "r") as f:
                    # MATLAB saves as (features, TRs), transpose to (TRs, features)
                    reg_data = f[layer_key][:].T.astype(np.float32)
                    train_reg_list.append(reg_data)
                    print(f"      run{run_idx}.h5: {reg_data.shape} (TRs x features)")
            
            X_train = np.concatenate(train_reg_list, axis=0)
            print(f"      X_train: {X_train.shape} (TRs x features)")
            print(f"      fMRI_train: {fmri_train_data.shape} (TRs x voxels)")
            
            # --- Shape Validation ---
            if X_train.shape[0] != fmri_train_data.shape[0]:
                print(f"      [ERROR] Shape mismatch: X={X_train.shape[0]}, fMRI={fmri_train_data.shape[0]}")
                continue
            
            print(f"      [OK] Training samples match")
            
            # --- Train Encoding Model ---
            print(f"      Training encoding model...", end=" ", flush=True)
            W, _, _ = voxelwise_encoding_new(
                X_train, fmri_train_data, alphas, nfold, device=device
            )
            print("done")
            
            # Save W matrix
            save_W_h5(W, subject_save_dir / f"W_layer{lay + 1}.h5")
            
            # --- Test ---
            test_corrs_per_run = []
            
            for run_idx in test_runs:
                run_name = f"run{run_idx}"
                r_path = regressor_path / f"run{run_idx}.h5"
                
                with h5py.File(r_path, "r") as f:
                    X_test = f[layer_key][:].T.astype(np.float32)
                
                if run_name not in fmri_data_struct:
                    print(f"      [WARNING] Missing fMRI {run_name}")
                    continue
                    
                fmri_test = fmri_data_struct[run_name]
                
                # Truncate to shorter length (regressor or fMRI)
                min_len = min(X_test.shape[0], fmri_test.shape[0])
                X_test = X_test[:min_len, :]
                fmri_test = fmri_test[:min_len, :]
                
                # Predict
                pred_fmri = X_test @ W
                
                corrs = compute_correlation_vectorized(pred_fmri, fmri_test)
                test_corrs_per_run.append(corrs)
                print(f"      {run_name}: mean r = {np.nanmean(corrs):.4f} ({min_len} TRs)")
            
            if len(test_corrs_per_run) > 0:
                layer_corrs = np.mean(np.stack(test_corrs_per_run), axis=0)
                all_subject_corrs.append(layer_corrs)
                
                layer_time = time.time() - layer_start
                mean_r = np.nanmean(layer_corrs)
                max_r = np.nanmax(layer_corrs)
                print(f"      Layer avg: r = {mean_r:.4f} (max: {max_r:.4f}) [{layer_time:.1f}s]")
            else:
                print(f"      [WARNING] No test data processed for layer {lay}")
                all_subject_corrs.append(np.zeros(fmri_train_data.shape[1]))
        
        # --- Save Results ---
        if len(all_subject_corrs) == num_layers:
            final_corr_matrix = np.stack(all_subject_corrs)  # (4, 59412)
            print(f"\n    Final correlation matrix: {final_corr_matrix.shape}")
            
            # Save .mat
            sio.savemat(
                str(subject_save_dir / "average_correlations.mat"),
                {"final_corr_matrix": final_corr_matrix},
                do_compression=True
            )
            print(f"    Saved: average_correlations.mat")
            
            # Save .npy
            np.save(
                str(subject_save_dir / "average_correlations.npy"),
                final_corr_matrix
            )
            print(f"    Saved: average_correlations.npy")
            
            # Save CIFTI dscalar
            cifti_output = str(subject_save_dir / "average_correlations.dscalar.nii")
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
    """Quick validation of MATLAB outputs before encoding."""
    print("\n" + "="*70)
    print(f" QUICK SHAPE CHECK - Gump Dataset ({FEATURE_TYPE})")
    print("="*70)
    
    SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE
    regressor_path = SAVE_ROOT / "regressors"
    
    if not regressor_path.exists():
        print(f"  [NOT FOUND] {regressor_path}")
        return
    
    print(f"\n  Regressor path: {regressor_path}")
    print(f"  Feature type: {FEATURE_TYPE}")
    print(f"  Layer names: {layer_names}")
    print(f"  Train runs: {train_runs} -> run{train_runs[0]}.h5 to run{train_runs[-1]}.h5")
    print(f"  Test runs: {test_runs} -> run{test_runs[0]}.h5 to run{test_runs[-1]}.h5")
    
    # Check first file
    first_file = regressor_path / f"run{train_runs[0]}.h5"
    if not first_file.exists():
        print(f"  [NOT FOUND] {first_file}")
        return
    
    with h5py.File(first_file, "r") as f:
        print(f"\n  Datasets in {first_file.name}: {list(f.keys())}")
        for key in f.keys():
            shape = f[key].shape
            print(f"    {key}: {shape} (features={shape[0]}, TRs={shape[1]})")
    
    # Full validation
    check_regressor_shapes(regressor_path, layer_names, train_runs, test_runs)
    
    # Also check one fMRI file
    print("\n  Checking fMRI structure...")
    test_subject = subjects[0]
    fmri_path = Path(FMRI_DATA_ROOT) / test_subject / f"{test_subject}_per_run_fmri.mat"
    
    if fmri_path.exists():
        with h5py.File(fmri_path, "r") as f:
            print(f"  fMRI file: {fmri_path.name}")
            struct = f["fmri_data_per_run"]
            print(f"  Available runs: {list(struct.keys())}")
            for run_name in struct.keys():
                shape = struct[run_name].shape
                # h5py reads as (TRs, voxels) due to MATLAB/Python ordering
                print(f"    {run_name}: {shape} (TRs={shape[0]}, voxels={shape[1]})")
        
        # Compare with regressor TRs
        print("\n  TR alignment check (fMRI vs Regressors):")
        first_layer_key = get_layer_key(layer_names[0])
        
        for run_idx in train_runs + test_runs:
            run_name = f"run{run_idx}"
            r_file = regressor_path / f"run{run_idx}.h5"
            
            with h5py.File(fmri_path, "r") as f:
                fmri_trs = f["fmri_data_per_run"][run_name].shape[0]
            
            if r_file.exists():
                with h5py.File(r_file, "r") as f:
                    reg_trs = f[first_layer_key].shape[1]  # (features, TRs)
                diff = fmri_trs - reg_trs
                status = "OK" if diff == 0 else f"fMRI has {diff} extra TRs (will truncate)"
                print(f"    {run_name}: fMRI={fmri_trs}, Reg={reg_trs} -> {status}")
    else:
        print(f"  [NOT FOUND] {fmri_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gump Encoding Pipeline")
    parser.add_argument("--check", action="store_true", help="Quick shape check only")
    args = parser.parse_args()
    
    if args.check:
        quick_shape_check()
    else:
        run_Gump_encoding()