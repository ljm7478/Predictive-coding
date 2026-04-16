"""
Diagnostic script to check regressor files from MATLAB Step 2.
"""

import h5py
import numpy as np
from pathlib import Path

# Path to regressors
BASE_SAVE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/longer_time/four_train_four_test/layer4"
FEATURE_TYPE = "Ahat"
TEMPORAL_TYPE = "indirect_1s_extrap"

regressor_path = Path(BASE_SAVE_DIR) / FEATURE_TYPE / TEMPORAL_TYPE / "regressors"

print(f"Checking regressors at: {regressor_path}\n")

# Check all runs
for run_file in sorted(regressor_path.glob("*.h5")):
    print(f"\n{'='*50}")
    print(f"File: {run_file.name}")
    print(f"{'='*50}")
    
    with h5py.File(run_file, "r") as f:
        print(f"Datasets in file: {list(f.keys())}")
        
        for key in f.keys():
            data = f[key][:]
            print(f"\n  {key}:")
            print(f"    Shape (as stored): {data.shape}")
            print(f"    Shape (transposed): {data.T.shape}")
            print(f"    Dtype: {data.dtype}")
            print(f"    Min: {data.min():.4f}, Max: {data.max():.4f}")
            print(f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            print(f"    NaN count: {np.isnan(data).sum()}")
            print(f"    Zero count: {(data == 0).sum()} / {data.size}")

# Also check PCA components file
print(f"\n\n{'='*50}")
print("Checking PCA components...")
print(f"{'='*50}")

pca_dir = Path(BASE_SAVE_DIR) / FEATURE_TYPE / TEMPORAL_TYPE

for lay in range(1, 5):
    pca_file = pca_dir / f"pca_components_layer{lay}.mat"
    if pca_file.exists():
        print(f"\nLayer {lay} PCA file: {pca_file.name}")
        try:
            # Try scipy.io first (for older .mat files)
            import scipy.io as sio
            mat = sio.loadmat(pca_file)
            for key in mat.keys():
                if not key.startswith('_'):
                    print(f"  {key}: {mat[key].shape}")
        except:
            # Fall back to h5py (for v7.3 .mat files)
            with h5py.File(pca_file, "r") as f:
                for key in f.keys():
                    if not key.startswith('#'):
                        data = f[key]
                        if hasattr(data, 'shape'):
                            print(f"  {key}: {data.shape}")
    else:
        print(f"\nLayer {lay} PCA file NOT FOUND: {pca_file}")