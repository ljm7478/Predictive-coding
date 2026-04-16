"""
Check Raw W Matrix Distribution
Purpose: Understand the actual weight patterns before PC1 transformation
"""

import numpy as np
import h5py
import scipy.io as sio
from pathlib import Path
import nibabel as nib

# =============================================================================
# CONFIGURATION
# =============================================================================

# GUMP dataset
BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"

# Or Budapest dataset
# BASE_DIR = "/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/Budapest_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test/layer4"

SUBJECT = "sub-01"

# Glasser parcellation for ROI
GLASSER_PATH = "/combinelab/02_data/99_parcellation/Glasser2016/Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"

# V1/V2 labels in Glasser (bilateral)
V1V2_LABELS = [1, 4, 181, 184]

# Higher visual / association areas
MT_LABELS = [23, 203]  # MT bilateral
PFC_LABELS = [60, 61, 62, 240, 241, 242]  # dlPFC bilateral (example)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_W_matrix(filepath):
    """Load W matrix from .mat or .h5 file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.mat':
        try:
            # Try HDF5 format first (MATLAB v7.3)
            with h5py.File(filepath, 'r') as f:
                W = f['W'][:].astype(np.float32)
        except:
            # Fall back to scipy for older .mat format
            data = sio.loadmat(str(filepath))
            W = data['W'].astype(np.float32)
    elif filepath.suffix == '.h5':
        with h5py.File(filepath, 'r') as f:
            W = f['W'][:].astype(np.float32)
    else:
        raise ValueError(f"Unknown file format: {filepath.suffix}")
    
    return W


def load_glasser_parcellation(path, n_cortex=59412):
    """Load Glasser parcellation labels."""
    img = nib.load(path)
    data = img.get_fdata().flatten()
    if len(data) > n_cortex:
        data = data[:n_cortex]
    return data.astype(int)


def analyze_W_layer(W, layer_name, glasser_labels=None):
    """Analyze W matrix statistics for one layer."""
    
    # Check shape and transpose if needed
    # We want W to be (features, voxels) for consistency
    # But if it's (voxels, features), we transpose
    if W.shape[0] == 59412:  # voxels first
        n_voxels, n_features = W.shape
        W_for_mean = W  # (voxels, features)
        W_mean = W.mean(axis=1)  # mean across features -> (voxels,)
    else:  # features first
        n_features, n_voxels = W.shape
        W_for_mean = W.T  # transpose to (voxels, features)
        W_mean = W.mean(axis=0)  # mean across features -> (voxels,)
    
    print(f"\n  {layer_name}: shape {W.shape} -> {n_voxels} voxels, {n_features} features")
    print(f"  {'-'*50}")
    
    # Overall statistics
    print(f"    Overall:  min={W.min():.4f}, max={W.max():.4f}, mean={W.mean():.4f}")
    print(f"              std={W.std():.4f}, %neg={100*np.sum(W<0)/W.size:.1f}%")
    
    # Mean across features (this is what we care about for visualization)
    print(f"\n    Mean(W) per voxel: shape={W_mean.shape}")
    print(f"              min={W_mean.min():.4f}, max={W_mean.max():.4f}, mean={W_mean.mean():.4f}")
    print(f"              %neg={100*np.sum(W_mean<0)/len(W_mean):.1f}%")
    
    # ROI analysis if parcellation available
    if glasser_labels is not None and len(W_mean) == len(glasser_labels):
        v1v2_mask = np.isin(glasser_labels, V1V2_LABELS)
        mt_mask = np.isin(glasser_labels, MT_LABELS)
        
        if np.sum(v1v2_mask) > 0:
            v1_W_mean = W_mean[v1v2_mask].mean()
            v1_pct_neg = 100 * np.sum(W_mean[v1v2_mask] < 0) / np.sum(v1v2_mask)
            print(f"\n    V1/V2 ROI ({np.sum(v1v2_mask)} voxels):")
            print(f"              mean(W_mean)={v1_W_mean:.4f}, %neg={v1_pct_neg:.1f}%")
        
        if np.sum(mt_mask) > 0:
            mt_W_mean = W_mean[mt_mask].mean()
            mt_pct_neg = 100 * np.sum(W_mean[mt_mask] < 0) / np.sum(mt_mask)
            print(f"    MT ROI ({np.sum(mt_mask)} voxels):")
            print(f"              mean(W_mean)={mt_W_mean:.4f}, %neg={mt_pct_neg:.1f}%")
    elif glasser_labels is not None:
        print(f"\n    [WARNING] W_mean length ({len(W_mean)}) != parcellation ({len(glasser_labels)})")
    
    return W_mean


def main():
    print("\n" + "="*70)
    print(" RAW W MATRIX ANALYSIS")
    print(" Purpose: Check weight distribution before PC1 transformation")
    print("="*70)
    
    # Load Glasser parcellation
    try:
        glasser_labels = load_glasser_parcellation(GLASSER_PATH)
        print(f"\nLoaded Glasser parcellation: {len(glasser_labels)} vertices")
        print(f"V1/V2 vertices: {np.sum(np.isin(glasser_labels, V1V2_LABELS))}")
    except Exception as e:
        print(f"\n[WARNING] Could not load Glasser parcellation: {e}")
        glasser_labels = None
    
    # ==========================================================================
    # PREDICTION (Ahat)
    # ==========================================================================
    print("\n" + "="*70)
    print(f" PREDICTION (Ahat) - {SUBJECT}")
    print("="*70)
    
    ahat_dir = Path(BASE_DIR) / "Ahat" / SUBJECT
    ahat_W_means = []
    
    for layer in range(1, 5):
        w_path = ahat_dir / f"W_layer{layer}.mat"
        if w_path.exists():
            W = load_W_matrix(w_path)
            W_mean = analyze_W_layer(W, f"Layer {layer}", glasser_labels)
            ahat_W_means.append(W_mean)
        else:
            print(f"  [NOT FOUND] {w_path}")
            ahat_W_means.append(None)
    
    # ==========================================================================
    # ERROR (E)
    # ==========================================================================
    print("\n" + "="*70)
    print(f" ERROR (E) - {SUBJECT}")
    print("="*70)
    
    e_dir = Path(BASE_DIR) / "E" / SUBJECT
    e_W_means = []
    
    for layer in range(1, 5):
        w_path = e_dir / f"W_layer{layer}.mat"
        if w_path.exists():
            W = load_W_matrix(w_path)
            W_mean = analyze_W_layer(W, f"Layer {layer}", glasser_labels)
            e_W_means.append(W_mean)
        else:
            print(f"  [NOT FOUND] {w_path}")
            e_W_means.append(None)
    
    # ==========================================================================
    # KEY COMPARISON: V1 in L1 vs L4
    # ==========================================================================
    if glasser_labels is not None and all(w is not None for w in ahat_W_means + e_W_means):
        print("\n" + "="*70)
        print(" KEY COMPARISON: V1/V2 Mean Weight Across Layers")
        print("="*70)
        
        v1v2_mask = np.isin(glasser_labels, V1V2_LABELS)
        
        print(f"\n{'Layer':<10} {'Ahat V1 mean':>15} {'E V1 mean':>15} {'Ahat-E':>15}")
        print("-"*60)
        
        for i, layer in enumerate(['L1', 'L2', 'L3', 'L4']):
            ahat_v1 = ahat_W_means[i][v1v2_mask].mean()
            e_v1 = e_W_means[i][v1v2_mask].mean()
            diff = ahat_v1 - e_v1
            print(f"{layer:<10} {ahat_v1:>+15.4f} {e_v1:>+15.4f} {diff:>+15.4f}")
        
        # L4 - L1 change (hierarchy effect)
        print(f"\n{'='*60}")
        print(f"L4 - L1 Change (Hierarchy Effect in V1):")
        ahat_change = ahat_W_means[3][v1v2_mask].mean() - ahat_W_means[0][v1v2_mask].mean()
        e_change = e_W_means[3][v1v2_mask].mean() - e_W_means[0][v1v2_mask].mean()
        print(f"  Ahat: {ahat_change:+.4f}  {'(suppression ?)' if ahat_change < 0 else '(no suppression)'}")
        print(f"  E:    {e_change:+.4f}")
        
        # ==========================================================================
        # RECOMMENDATION
        # ==========================================================================
        print("\n" + "="*70)
        print(" RECOMMENDATION FOR VISUALIZATION")
        print("="*70)
        
        print(f"""
Based on the analysis:

1. USE mean(W) instead of PC1:
   - PC1 centers the data, losing absolute magnitude information
   - mean(W) preserves whether weights are positive or negative overall

2. Alternative beta calculation:
   beta_mean = W.mean(axis=0)  # Simple mean across features
   
   Or weighted by variance:
   beta_weighted = (W * np.var(W, axis=0)).sum(axis=0) / np.var(W, axis=0).sum()

3. For visualization:
   - Use the mean(W) values computed above
   - Apply symmetric color range centered at 0
   - This should show V1 suppression in Ahat L4 if present
""")


if __name__ == "__main__":
    main()