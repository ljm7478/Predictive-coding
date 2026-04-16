"""
Step 3: GPU-accelerated Voxel-wise Encoding
With comprehensive data shape validation
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

BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/longer_time/four_train_four_test/layer4"
FMRI_DATA_ROOT = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/fmri_new/"
CIFTI_TEMPLATE = "/combinelab2/03_user/jungmin/02_data/01_Gump/01_FG_preprocessed/sub-01/sub-01_ciftify/ciftify/sub-01/MNINonLinear/Results_MNI152NLin2009cAsym/ses-movie_task-movie_run-1/ses-movie_task-movie_run-1_Atlas_s0.dtseries.nii"

TEMPORAL_TYPES = ["indirect_1s_extrap", "indirect_2s_extrap", "indirect_5s_extrap", "indirect_10s_extrap"]


# Layer configuration based on feature type
FEATURE_TYPE = "E"

if FEATURE_TYPE == "Ahat":
    layer_names = ["/Ahat0", "/Ahat1", "/Ahat2", "/Ahat3"]
else:  # "E"
    layer_names = ["/E0", "/E1", "/E2", "/E3"]

num_layers = len(layer_names)

train_runs = [0, 1, 2, 3]
test_runs = [4, 5, 6, 7]

lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
nfold = 3

subjects = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", 
    "sub-06", "sub-09", "sub-10", "sub-14", "sub-15", 
    "sub-16", "sub-17", "sub-18", "sub-19", "sub-20"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

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
    """
    Save as CIFTI dscalar with cortex only (like MATLAB ciftisavereset).
    
    Input data: (n_maps, n_grayordinates) = (4, 59412)
    """
    template = nib.load(template_file)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    n_maps, n_gray_data = data.shape
    print(f"    CIFTI input: {data.shape} (layers × grayordinates)")
    
    # Get brain axis from template
    brain_axis = template.header.get_axis(1)
    
    # Find cortical indices only (remove subcortical)
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
    
    # Create scalar axis with layer names
    scalar_axis = nib.cifti2.ScalarAxis([f'Layer_{i+1}' for i in range(n_maps)])
    
    # Create new header
    new_header = nib.cifti2.Cifti2Header.from_axes((scalar_axis, cortex_brain_axis))
    
    # Create and save image
    new_img = nib.Cifti2Image(data.astype(np.float32), header=new_header)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    nib.save(new_img, output_file)
    
    print(f"    Saved: {output_file}")
    return True


def check_regressor_shapes(regressor_path, layer_names, train_runs, test_runs):
    """
    Check regressor shapes before encoding.
    Returns True if shapes are valid, False otherwise.
    """
    print("\n  [Shape Check] Checking regressor dimensions...")
    
    all_runs = train_runs + test_runs
    shapes_per_layer = {layer: [] for layer in layer_names}
    
    for run_idx in all_runs:
        r_path = regressor_path / f"run{run_idx + 1}.h5"
        
        if not r_path.exists():
            print(f"    [ERROR] Missing regressor file: {r_path}")
            return False, None
        
        with h5py.File(r_path, "r") as f:
            for layer_name in layer_names:
                if layer_name not in f and layer_name[1:] not in f:
                    # Try without leading slash
                    actual_key = layer_name[1:] if layer_name.startswith('/') else layer_name
                    if actual_key not in f:
                        print(f"    [ERROR] Layer {layer_name} not found in {r_path}")
                        print(f"    Available keys: {list(f.keys())}")
                        return False, None
                
                # Get shape (MATLAB saves as features x time)
                shape = f[layer_name].shape
                shapes_per_layer[layer_name].append((run_idx, shape))
    
    # Print shape summary
    print("\n    Regressor shapes (MATLAB format: features × TRs):")
    n_features = None
    
    for layer_name in layer_names:
        shapes = shapes_per_layer[layer_name]
        print(f"      {layer_name}:")
        
        for run_idx, shape in shapes:
            run_type = "train" if run_idx in train_runs else "test"
            print(f"        run{run_idx + 1} ({run_type}): {shape}")
            
            # After transpose in Python: (TRs, features)
            # MATLAB saves as (features, TRs), so shape[0] = features
            if n_features is None:
                n_features = shape[0]
            elif shape[0] != n_features:
                print(f"    [WARNING] Inconsistent feature count: {shape[0]} vs {n_features}")
    
    # Calculate total training TRs
    train_trs = sum(shapes_per_layer[layer_names[0]][i][1][1] 
                    for i, (run_idx, _) in enumerate(shapes_per_layer[layer_names[0]]) 
                    if run_idx in train_runs)
    
    print(f"\n    Summary:")
    print(f"      Features (PCA components): {n_features}")
    print(f"      Total training TRs: {train_trs}")
    print(f"      Features/TRs ratio: {n_features/train_trs:.3f}")
    
    # Check for overfitting risk
    if n_features > train_trs:
        print(f"\n    [ERROR] Features ({n_features}) > Training TRs ({train_trs})")
        print(f"    This will cause overfitting! Re-run MATLAB PCA with MAX_PCA_COMPONENTS < {train_trs}")
        return False, None
    elif n_features > train_trs * 0.5:
        print(f"\n    [WARNING] Features/TRs ratio > 0.5, may have some overfitting")
    else:
        print(f"\n    [OK] Features < TRs, encoding should work properly")
    
    return True, n_features


def check_fmri_shapes(fmri_data_struct, train_runs, test_runs):
    """
    Check fMRI data shapes.
    Returns (train_trs, n_voxels) if valid.
    """
    print("\n    [Shape Check] fMRI dimensions:")
    
    train_trs_list = []
    test_trs_list = []
    n_voxels = None
    
    for run_idx in train_runs + test_runs:
        run_name = f"run{run_idx + 1}"
        
        if run_name not in fmri_data_struct:
            print(f"      [WARNING] Missing {run_name}")
            continue
        
        data = fmri_data_struct[run_name]
        shape = data.shape
        run_type = "train" if run_idx in train_runs else "test"
        print(f"      {run_name} ({run_type}): {shape}")
        
        # Determine orientation
        # Expected: (TRs, voxels) after loading
        if shape[0] < shape[1]:
            # Likely (TRs, voxels) - correct
            trs, voxels = shape
        else:
            # Likely (voxels, TRs) - need info
            voxels, trs = shape
            print(f"        Note: Data may be (voxels × TRs), will handle accordingly")
        
        if run_idx in train_runs:
            train_trs_list.append(trs)
        else:
            test_trs_list.append(trs)
        
        if n_voxels is None:
            n_voxels = voxels
    
    total_train_trs = sum(train_trs_list)
    total_test_trs = sum(test_trs_list)
    
    print(f"\n      Summary:")
    print(f"        Voxels: {n_voxels}")
    print(f"        Training TRs: {total_train_trs} ({len(train_trs_list)} runs)")
    print(f"        Test TRs: {total_test_trs} ({len(test_trs_list)} runs)")
    
    return total_train_trs, n_voxels


# =============================================================================
# 3. MAIN ENCODING LOOP
# =============================================================================

def run_step3_encoding():
    """Main encoding pipeline with shape validation."""
    
    alphas = np.array(lambdas, dtype=np.float32)
    total_start = time.time()
    
    for current_temp_type in TEMPORAL_TYPES[0:1]:
        print(f"\n{'='*70}")
        print(f" PROCESSING: {current_temp_type} ({FEATURE_TYPE})")
        print(f"{'='*70}")
        
        SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE / current_temp_type
        regressor_path = SAVE_ROOT / "regressors"
        
        # --- Check if regressors exist ---
        if not regressor_path.exists():
            print(f"  [ERROR] Regressors not found at: {regressor_path}")
            continue
        
        # --- Validate regressor shapes BEFORE processing subjects ---
        is_valid, n_features = check_regressor_shapes(
            regressor_path, layer_names, train_runs, test_runs
        )
        
        if not is_valid:
            print(f"\n  [SKIP] Skipping {current_temp_type} due to shape issues")
            continue
        
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
            
            # --- Check fMRI shapes ---
            total_train_trs, n_voxels = check_fmri_shapes(
                fmri_data_struct, train_runs, test_runs
            )
            
            # --- Concatenate Training fMRI ---
            fmri_train_cell = []
            valid_train = True
            
            for run_idx in train_runs:
                run_name = f"run{run_idx + 1}"
                if run_name in fmri_data_struct:
                    fmri_train_cell.append(fmri_data_struct[run_name])
                else:
                    print(f"    [WARNING] Missing {run_name}")
                    valid_train = False
                    break
            
            if not valid_train:
                continue
            
            # Concatenate along time axis
            fmri_train_data = np.concatenate(fmri_train_cell, axis=0)
            print(f"\n    Concatenated fMRI (train): {fmri_train_data.shape}")
            
            # Verify shapes match
            if fmri_train_data.shape[0] != total_train_trs:
                print(f"    [WARNING] Mismatch: concatenated={fmri_train_data.shape[0]}, expected={total_train_trs}")
            
            # --- Layer Loop ---
            all_subject_corrs = []
            
            for lay in range(num_layers):
                layer_start = time.time()
                layer_name = layer_names[lay]
                print(f"\n    Layer {lay + 1}/{num_layers} ({layer_name})")
                
                # --- Load Training Regressors ---
                train_reg_cell = []
                for run_idx in train_runs:
                    r_path = regressor_path / f"run{run_idx + 1}.h5"
                    with h5py.File(r_path, "r") as f:
                        # MATLAB saves as (features, time), transpose to (time, features)
                        reg_data = f[layer_name][:].T.astype(np.float32)
                        train_reg_cell.append(reg_data)
                        print(f"      run{run_idx + 1}: {f[layer_name].shape} -> {reg_data.shape} (after .T)")
                
                Y_train = np.concatenate(train_reg_cell, axis=0)
                print(f"      Y_train: {Y_train.shape} (TRs × features)")
                print(f"      fMRI_train: {fmri_train_data.shape} (TRs × voxels)")
                
                # --- Final Shape Validation ---
                if Y_train.shape[0] != fmri_train_data.shape[0]:
                    print(f"      [ERROR] Shape mismatch: Y={Y_train.shape[0]}, fMRI={fmri_train_data.shape[0]}")
                    continue
                
                if Y_train.shape[1] > Y_train.shape[0]:
                    print(f"      [ERROR] Features ({Y_train.shape[1]}) > Samples ({Y_train.shape[0]})")
                    print(f"      This will cause overfitting! Skipping...")
                    continue
                
                print(f"      [OK] Features ({Y_train.shape[1]}) < Samples ({Y_train.shape[0]})")
                
                # --- Train Encoding Model ---
                print(f"      Training encoding model...", end=" ")
                W, _, _ = voxelwise_encoding_new(
                    Y_train, fmri_train_data, alphas, nfold, device=device
                )
                
                # Save W matrix
                save_W_h5(W, subject_save_dir / f"W_layer{lay + 1}.h5")
                
                # --- Test ---
                test_corrs_per_run = []
                
                for run_idx in test_runs:
                    run_name = f"run{run_idx + 1}"
                    r_path = regressor_path / f"{run_name}.h5"
                    
                    with h5py.File(r_path, "r") as f:
                        Y_test = f[layer_name][:].T.astype(np.float32)
                    
                    if run_name not in fmri_data_struct:
                        continue
                    fmri_test = fmri_data_struct[run_name]
                    
                    if isinstance(W, torch.Tensor):
                        W = W.cpu().numpy()
                    
                    pred_fmri = Y_test @ W
                    min_len = min(pred_fmri.shape[0], fmri_test.shape[0])
                    
                    corrs = compute_correlation_vectorized(
                        pred_fmri[:min_len, :],
                        fmri_test[:min_len, :]
                    )
                    test_corrs_per_run.append(corrs)
                
                layer_corrs = np.mean(np.stack(test_corrs_per_run), axis=0)
                all_subject_corrs.append(layer_corrs)
                
                layer_time = time.time() - layer_start
                mean_r = np.nanmean(layer_corrs)
                max_r = np.nanmax(layer_corrs)
                print(f"r = {mean_r:.4f} (max: {max_r:.4f}) ({layer_time:.1f}s)")
            
            # --- Save Results ---
            final_corr_matrix = np.stack(all_subject_corrs)  # (4, 59412)
            print(f"\n    Final correlation matrix: {final_corr_matrix.shape}")
            
            # Save .mat
            sio.savemat(
                str(subject_save_dir / "average_correlations.mat"),
                {"final_corr_matrix": final_corr_matrix},
                do_compression=True
            )
            print(f"    Saved: average_correlations.mat")
            
            # Save .npy for Python
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
    print(f"=== ALL DONE in {total_time/3600:.2f} hours ===")
    print(f"{'='*70}")


# =============================================================================
# 4. QUICK SHAPE CHECK (run separately to validate before encoding)
# =============================================================================

def quick_shape_check():
    """
    Quick check of all regressor files without running encoding.
    Use this to verify MATLAB output before starting the long encoding process.
    """
    print("\n" + "="*70)
    print(" QUICK SHAPE CHECK - Validating MATLAB outputs")
    print("="*70)
    
    for current_temp_type in TEMPORAL_TYPES[0:1]:
        print(f"\n--- {current_temp_type} ---")
        
        SAVE_ROOT = Path(BASE_SAVE_DIR) / FEATURE_TYPE / current_temp_type
        regressor_path = SAVE_ROOT / "regressors"
        
        if not regressor_path.exists():
            print(f"  [NOT FOUND] {regressor_path}")
            continue
        
        # Check first run file
        r_path = regressor_path / "run1.h5"
        if not r_path.exists():
            print(f"  [NOT FOUND] {r_path}")
            continue
        
        with h5py.File(r_path, "r") as f:
            print(f"  Datasets: {list(f.keys())}")
            for key in f.keys():
                shape = f[key].shape
                n_features = shape[0]  # MATLAB format: (features, TRs)
                n_trs = shape[1]
                print(f"    {key}: {shape} (features={n_features}, TRs={n_trs})")
                
                # Quick overfitting check (assuming ~1800 total training TRs)
                expected_train_trs = 1818  # Adjust for your dataset
                if n_features > expected_train_trs:
                    print(f"      [WARNING] Features ({n_features}) > expected TRs ({expected_train_trs})")
                    print(f"      Re-run MATLAB PCA with MAX_PCA_COMPONENTS = 500")
                else:
                    print(f"      [OK] Features < expected training TRs")


if __name__ == "__main__":
    # Option 1: Quick check only (uncomment to use)
    # quick_shape_check()
    
    # Option 2: Full encoding
    run_step3_encoding()