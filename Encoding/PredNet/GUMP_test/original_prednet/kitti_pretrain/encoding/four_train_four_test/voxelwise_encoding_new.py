"""
voxelwise_encoding_new.py
Exact translation of voxelwise_encoding_new.m for GPU acceleration.
With batched processing for memory efficiency.
"""

import numpy as np
import torch


def voxelwise_encoding_new(Y, X, lambdas, nfold, device="cuda", batch_size=10000):
    """
    Voxel-wise ridge regression with cross-validation.
    
    EXACT match to MATLAB voxelwise_encoding_new.m
    Batched for GPU memory efficiency.
    
    Parameters:
    -----------
    Y : ndarray (Nt, Nc)
        Regressors, each column is a regressor (should be zero-mean)
    X : ndarray (Nt, Nv)
        Response data, each column is a voxel time series (should be zero-mean)
    lambdas : list or ndarray
        Candidate regularization parameters
    nfold : int
        Number of CV folds
    device : str
        'cuda' for GPU, 'cpu' for CPU
    batch_size : int
        Number of voxels to process at once (reduce if OOM)
        
    Returns:
    --------
    W : ndarray (Nc, Nv)
        Optimal encoding weights for each voxel
    Rmat : ndarray (Nv, n_lambdas, nfold)
        Validation correlations
    Lambda : ndarray (Nv,)
        Optimal regularization parameter for each voxel
    """
    
    Nt, Nc = Y.shape  # timepoints, components
    Nv = X.shape[1]   # voxels
    n_lambdas = len(lambdas)
    
    lambdas_arr = np.array(lambdas)
    
    # Move Y to GPU (regressors are shared across all voxels)
    Y_gpu = torch.tensor(Y, dtype=torch.float32, device=device)
    
    # Fold indices (match MATLAB's round(linspace(...)))
    fold_indices = np.round(np.linspace(1, Nt + 1, nfold + 1)).astype(int)
    fold_indices = fold_indices - 1  # Convert to 0-indexed
    
    # Pre-compute Y splits for each fold (these are shared)
    Y_train_folds = []
    Y_val_folds = []
    train_indices = []
    val_indices = []
    
    for nf in range(nfold):
        idx_v_start = fold_indices[nf]
        idx_v_end = fold_indices[nf + 1]
        idx_v = list(range(idx_v_start, idx_v_end))
        idx_t = list(set(range(Nt)) - set(idx_v))
        idx_t.sort()
        
        train_indices.append(idx_t)
        val_indices.append(idx_v)
        Y_train_folds.append(Y_gpu[idx_t, :])
        Y_val_folds.append(Y_gpu[idx_v, :])
    
    # Pre-compute YTY and regularization matrices for each fold/lambda
    # These are independent of voxels
    eye_Nc = torch.eye(Nc, dtype=torch.float32, device=device)
    
    # Results arrays (CPU to save GPU memory)
    Rmat = np.zeros((Nv, n_lambdas, nfold), dtype=np.float32)
    
    print("Validating regularization parameters (batched)..")
    
    # Process voxels in batches
    n_batches = (Nv + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start_v = batch_idx * batch_size
        end_v = min((batch_idx + 1) * batch_size, Nv)
        batch_voxels = end_v - start_v
        
        print(f"  Batch {batch_idx + 1}/{n_batches} (voxels {start_v}-{end_v})")
        
        # Load batch of voxels to GPU
        X_batch = torch.tensor(X[:, start_v:end_v], dtype=torch.float32, device=device)
        
        for nf in range(nfold):
            Y_t = Y_train_folds[nf]
            Y_v = Y_val_folds[nf]
            idx_t = train_indices[nf]
            idx_v = val_indices[nf]
            
            X_t = X_batch[idx_t, :]
            X_v = X_batch[idx_v, :]
            
            T = Y_t.shape[0]
            YTY = Y_t.T @ Y_t
            YTX = Y_t.T @ X_t
            
            for k, lmb in enumerate(lambdas_arr):
                # Ridge regression
                reg_matrix = YTY + lmb * T * eye_Nc
                M = torch.linalg.solve(reg_matrix, YTX)
                
                # Predict
                X_vp = Y_v @ M
                
                # Correlation
                R = batch_correlation(X_v, X_vp)
                R = torch.nan_to_num(R, nan=0.0)
                Rmat[start_v:end_v, k, nf] = R.cpu().numpy()
        
        # Clear batch from GPU
        del X_batch
        torch.cuda.empty_cache()
    
    # Choose optimal lambda for each voxel
    mean_R = Rmat.mean(axis=2)  # (Nv, n_lambdas)
    max_idx = mean_R.argmax(axis=1)  # (Nv,)
    Lambda = lambdas_arr[max_idx]
    
    # Train final model with optimal parameters on all data
    print("Training final encoding models with optimal parameters (batched)..")
    
    YTY_full = Y_gpu.T @ Y_gpu
    W = np.zeros((Nc, Nv), dtype=np.float32)
    
    unique_lambdas = np.unique(Lambda)
    
    for batch_idx in range(n_batches):
        start_v = batch_idx * batch_size
        end_v = min((batch_idx + 1) * batch_size, Nv)
        
        print(f"  Batch {batch_idx + 1}/{n_batches} (voxels {start_v}-{end_v})")
        
        # Load batch
        X_batch = torch.tensor(X[:, start_v:end_v], dtype=torch.float32, device=device)
        YTX_batch = Y_gpu.T @ X_batch
        Lambda_batch = Lambda[start_v:end_v]
        
        W_batch = torch.zeros((Nc, end_v - start_v), dtype=torch.float32, device=device)
        
        for lmb in unique_lambdas:
            voxel_mask = torch.tensor(Lambda_batch == lmb, device=device)
            
            if voxel_mask.sum() == 0:
                continue
            
            reg_matrix = YTY_full + lmb * Nt * eye_Nc
            M = torch.linalg.solve(reg_matrix, YTX_batch[:, voxel_mask])
            W_batch[:, voxel_mask] = M
        
        W[:, start_v:end_v] = W_batch.cpu().numpy()
        
        del X_batch, YTX_batch, W_batch
        torch.cuda.empty_cache()
    
    return W, Rmat, Lambda


def batch_correlation(X, Y):
    """
    Compute correlation between corresponding columns of X and Y.
    Equivalent to diag(corr(X, Y)) in MATLAB.
    """
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)
    
    num = (X_centered * Y_centered).sum(dim=0)
    denom = torch.sqrt(
        (X_centered ** 2).sum(dim=0) * (Y_centered ** 2).sum(dim=0)
    )
    
    R = num / (denom + 1e-10)
    return R


# ============== TEST ==============

if __name__ == "__main__":
    # Test with random data
    Nt = 500
    Nc = 100
    Nv = 1000
    
    Y = np.random.randn(Nt, Nc).astype(np.float32)
    X = np.random.randn(Nt, Nv).astype(np.float32)
    
    Y = Y - Y.mean(axis=0)
    X = X - X.mean(axis=0)
    
    lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    nfold = 3
    
    print("Testing voxelwise_encoding_new (batched)...")
    W, Rmat, Lambda = voxelwise_encoding_new(Y, X, lambdas, nfold, batch_size=500)
    
    print(f"\nResults:")
    print(f"  W shape: {W.shape}")
    print(f"  Rmat shape: {Rmat.shape}")
    print(f"  Lambda shape: {Lambda.shape}")
    print(f"  Mean validation R: {Rmat.mean():.4f}")