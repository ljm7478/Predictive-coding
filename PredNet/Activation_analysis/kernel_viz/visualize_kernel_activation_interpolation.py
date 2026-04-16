# -*- coding: utf-8 -*-
# Python 2.7
# Panel: (Ahat0 frame as input) -> (kernels of a chosen block) -> (activations)
# Works with PredNet HDF5 weights + your saved (B,T,H,W,C) npy feature maps.

import os, re, math
import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= HARD-CODED PATHS / CONFIG =========
# Weights
H5_PATH = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_Titanic/weights/prednet/Titanic_train_250epoch_layers4_32channel.hdf5"

# Which block??s kernels to visualize (e.g., "layer_ahat_1", "layer_ahat_2", "layer_ahat_3")
BLOCK_NAME = "layer_ahat_1"

# Use Ahat0 as the input image
AHAT0_NPY_PATH = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/layer4/1frame_prediction/seg1/Ahat0.npy"

# (Optional) Real activations to show on the right (B,T,H,W,C) for the same block as BLOCK_NAME
FEATUREMAP_NPY_PATH = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/layer4/1frame_prediction/seg1/Ahat1.npy"
# If you don't have it, set FEATUREMAP_NPY_PATH = None and the script will compute a linear response on the Ahat0 frame.

SAMPLE_IDX = 0
TIMESTEP   = 1   # choose the frame index you want to visualize (e.g., 1, 250, 500)
SAVE_DIR   = "./kernel_vis"
MAX_COLS   = 8
TILE_SZ    = (64, 64)  # w,h of small tiles in the kernel/activation mosaics
# =============================================

def makedirs_safe(p):
    if not os.path.exists(p):
        try: os.makedirs(p)
        except OSError: pass

def norm01(x):
    x = x.astype(np.float32)
    m, M = x.min(), x.max()
    return (x - m) / (M - m + 1e-8)

def grid_dims(n, maxc=8):
    cols = min(maxc, max(1, n))
    rows = int(math.ceil(float(n)/cols))
    return rows, cols

# ---- HDF5: load kernel tensor for a block (to Cout,Cin,kH,kW) ----
def _find_kernel_in_group(grp):
    keys = list(grp.keys())
    for kname in ("kernel:0", "kernel", "W:0", "W"):
        if kname in keys and isinstance(grp[kname], h5py.Dataset):
            return np.array(grp[kname])
    for sub in keys:
        if isinstance(grp[sub], h5py.Group):
            subkeys = list(grp[sub].keys())
            for kname in ("kernel:0", "kernel", "W:0", "W"):
                if kname in subkeys and isinstance(grp[sub][kname], h5py.Dataset):
                    return np.array(grp[sub][kname])
    return None

def load_block_kernel(h5_path, block_name):
    with h5py.File(h5_path, "r") as f:
        root = f["model_weights"]["pred_net_1"]
        if "pred_net_1" in root:
            root = root["pred_net_1"]
        if block_name not in root:
            raise ValueError("Block '{}' not found under /model_weights/pred_net_1/pred_net_1".format(block_name))
        grp = root[block_name]
        Wraw = _find_kernel_in_group(grp)
        if Wraw is None:
            raise ValueError("No kernel found in block '{}' (note: E blocks have no kernels)".format(block_name))
    s = Wraw.shape
    if len(s) != 4:
        raise ValueError("Kernel for '{}' is not 4D (shape={})".format(block_name, s))
    if s[0] <= 63 and s[1] <= 63:  # (kH,kW,Cin,Cout) -> Keras-style
        kH, kW, Cin, Cout = s
        W = np.transpose(Wraw, (3,2,0,1)).astype(np.float32)
    else:                           # (Cout,Cin,kH,kW)
        Cout, Cin, kH, kW = s
        W = Wraw.astype(np.float32)
    return W, (Cout, Cin, kH, kW)

# ---- Build kernel tiles: RGB if Cin==3 else grayscale effective ----
def kernel_tiles(W):
    Cout, Cin, kH, kW = W.shape
    tiles = []
    if Cin == 3:
        for j in range(Cout):
            R = norm01(W[j,0]); G = norm01(W[j,1]); B = norm01(W[j,2])
            rgb = np.stack([R,G,B], axis=2)  # (kH,kW,3)
            tiles.append(rgb)
    else:
        for j in range(Cout):
            K = np.sqrt((W[j]**2).sum(axis=0))  # effective 2D
            tiles.append(norm01(K))
    return tiles

def mosaic_from_tiles(tiles, cols, tile_sz=(64,64)):
    n = len(tiles)
    rows = int(math.ceil(float(n)/cols))
    is_rgb = (tiles[0].ndim == 3 and tiles[0].shape[2] == 3)
    Ht, Wt = tile_sz[1], tile_sz[0]
    if is_rgb:
        grid = np.zeros((rows*(Ht+2), cols*(Wt+2), 3), dtype=np.float32)
        for i, t in enumerate(tiles):
            r, c = i // cols, i % cols
            t = cv2.resize(t, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
            y0, x0 = r*(Ht+2), c*(Wt+2)
            grid[y0:y0+Ht, x0:x0+Wt, :] = t
    else:
        grid = np.zeros((rows*(Ht+2), cols*(Wt+2)), dtype=np.float32)
        for i, t in enumerate(tiles):
            r, c = i // cols, i % cols
            t = cv2.resize(t, (Wt, Ht), interpolation=cv2.INTER_CUBIC)
            y0, x0 = r*(Ht+2), c*(Wt+2)
            grid[y0:y0+Ht, x0:x0+Wt] = t
    return grid

# ---- Ahat0 frame as "input image" ----
def load_ahat0_frame(npy_path, sample_idx=0, timestep=1):
    """
    npy: (B,T,H,W,C). Return RGB (H,W,3) or gray (H,W,1) float [0,1] depending on C.
    """
    X = np.load(npy_path)  # (B,T,H,W,C)
    frame = X[sample_idx, timestep]  # (H,W,C)
    if frame.ndim == 2:
        img = frame.astype(np.float32)
        img = img[..., None]
        return img
    C = frame.shape[-1]
    img = frame.astype(np.float32)
    # Normalize if values not in [0,1]
    m, M = img.min(), img.max()
    if M > 1.0 or m < 0.0:
        img = norm01(img)
    if C == 3:
        return img  # (H,W,3)
    else:
        return img[..., None]  # (H,W,1)

# ---- Linear conv (fallback when no true feature map provided) ----
def linear_response(img, W):
    """
    img: (H,W,3) or (H,W,1) float [0,1]
    W: (Cout,Cin,kH,kW)
    Returns (H,W,Cout)
    """
    if img.ndim == 2:
        img = img[..., None]
    H, Wd, Cin_img = img.shape
    Cout, Cin_w, kH, kW = W.shape
    if Cin_img != Cin_w:
        if Cin_w == 1 and Cin_img == 3:
            gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
            img = gray[..., None]; Cin_img = 1
        else:
            raise ValueError("Image channels {} != kernel Cin {}".format(Cin_img, Cin_w))
    resp = np.zeros((H, Wd, Cout), dtype=np.float32)
    for j in range(Cout):
        acc = np.zeros((H, Wd), dtype=np.float32)
        for i in range(Cin_w):
            acc += cv2.filter2D(img[..., i], cv2.CV_32F, W[j,i])
        resp[..., j] = acc
    return resp

def featuremap_tiles_from_npy(npy_path, sample_idx, timestep, tile_sz=(64,64)):
    X = np.load(npy_path)                    # (B,T,H,W,C)
    fmap = np.transpose(X, (0,1,4,2,3))[sample_idx, timestep]  # (C,H,W)
    tiles = []
    for c in range(fmap.shape[0]):
        ch = norm01(fmap[c])
        tiles.append(cv2.resize(ch, tile_sz, interpolation=cv2.INTER_CUBIC))
    return tiles

def tiles_from_responses(R, tile_sz=(64,64)):
    H, Wd, C = R.shape
    tiles = []
    for j in range(C):
        ch = norm01(R[..., j])
        tiles.append(cv2.resize(ch, tile_sz, interpolation=cv2.INTER_CUBIC))
    return tiles

# ---- Build the full panel ----
def make_panel():
    makedirs_safe(SAVE_DIR)

    # 1) Load kernels for target block
    W, (Cout, Cin, kH, kW) = load_block_kernel(H5_PATH, BLOCK_NAME)
    print("[Kernel] {} shape (Cout={}, Cin={}, kH={}, kW={})".format(BLOCK_NAME, Cout, Cin, kH, kW))

    # 2) Kernel mosaic
    k_tiles = kernel_tiles(W)
    cols = min(MAX_COLS, Cout)
    k_grid = mosaic_from_tiles(k_tiles, cols, TILE_SZ)

    # 3) Input image = Ahat0 frame
    img = load_ahat0_frame(AHAT0_NPY_PATH, SAMPLE_IDX, TIMESTEP)  # (H,W,1 or 3)

    # 4) Right panel: prefer real feature map for the same block, else linear response
    if FEATUREMAP_NPY_PATH and os.path.exists(FEATUREMAP_NPY_PATH):
        act_tiles = featuremap_tiles_from_npy(FEATUREMAP_NPY_PATH, SAMPLE_IDX, TIMESTEP, TILE_SZ)
        act_grid  = mosaic_from_tiles(act_tiles, cols, TILE_SZ)
        act_title = "activation"
    else:
        R = linear_response(img, W)
        act_tiles = tiles_from_responses(R, TILE_SZ)
        act_grid  = mosaic_from_tiles(act_tiles, cols, TILE_SZ)
        act_title = "linear response (conv only)"

    # 5) Draw panel: Input (left) | Kernels (middle) | Activations (right)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # left: input (Ahat0 frame)
    if img.shape[-1] == 3:
        axes[0].imshow(img)
    else:
        axes[0].imshow(img[...,0], cmap='gray')
    axes[0].set_title("Input (Ahat0)  |  s{} t{}".format(SAMPLE_IDX, TIMESTEP), fontsize=11, pad=6)
    axes[0].axis('off')

    # middle: kernels
    if k_grid.ndim == 3:
        axes[1].imshow(k_grid)
        kdesc = "RGB kernels" if Cin == 3 else "effective kernels"
    else:
        axes[1].imshow(k_grid, cmap='gray')
        kdesc = "effective kernels"
    axes[1].set_title("{}".format(BLOCK_NAME, kdesc), fontsize=11, pad=6)
    axes[1].axis('off')

    # right: activations
    axes[2].imshow(act_grid if act_grid.ndim == 3 else act_grid, cmap=None if act_grid.ndim==3 else 'viridis')
    axes[2].set_title("{} | {}".format(BLOCK_NAME, act_title), fontsize=11, pad=6)
    axes[2].axis('off')

    plt.suptitle("Ahat0 -> {} kernels -> {}  (Cout={}, Cin={}, k={}?{})".format(
        BLOCK_NAME, act_title, Cout, Cin, kH, kW), fontsize=14, y=1.03)
    plt.tight_layout(rect=[0,0,1,0.96])

    out_path = os.path.join(SAVE_DIR, "{}_t{}.png".format(BLOCK_NAME, TIMESTEP))
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
    print("[Saved] {}".format(out_path))

if __name__ == "__main__":
    make_panel()
