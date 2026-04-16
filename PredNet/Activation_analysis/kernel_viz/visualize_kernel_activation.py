# -*- coding: utf-8 -*-
# Python 2.7
# Batch panels: Ahat0 (input) -> Kernels (std) -> Activations (std)
# Runs for layer4 (Ahat0..3) and layer7 (Ahat0..6), using each layer's own H5.

import os, math
import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========= PATH TEMPLATES (edit if needed) =========
# One checkpoint per layer_tag (layer4, layer7)
H5_PATH_TEMPLATE = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/02_Titanic/weights/prednet/Titanic_train_250epoch_{layer_tag}_32channel.hdf5"

# NPY root (B,T,H,W,C) files live under: .../npy/{layer_tag}/1frame_prediction/seg1
NPY_DIR_TEMPLATE = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/CNN_like/npy/{layer_tag}/1frame_prediction/seg1"
# ===================================================

# ========= WHAT TO RUN =========
RUN_SPECS = [
    {"layer_tag": "layer4", "levels": [0, 1, 2, 3]},                    # Ahat0..3
    {"layer_tag": "layer7", "levels": [0, 1, 2, 3, 4, 5, 6]},           # Ahat0..6
]
TIMESTEPS  = [1, 250]   # frames to visualize
SAMPLE_IDX = 0
SAVE_DIR   = "./kernel_vis/modified_prednet"
MAX_COLS   = 8
TILE_SZ    = (64, 64)        # (w,h) tile size for mosaics
# ==============================

def makedirs_safe(p):
    if not os.path.exists(p):
        try: os.makedirs(p)
        except OSError: pass

def norm01(x):
    x = x.astype(np.float32)
    m, M = x.min(), x.max()
    return (x - m) / (M - m + 1e-8)

def grid_cols(n, maxc=8):
    return min(maxc, max(1, n))

# ---- HDF5: load kernels for a block as (Cout,Cin,kH,kW) ----
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
    if not os.path.exists(h5_path):
        raise IOError("H5 not found: {}".format(h5_path))
    with h5py.File(h5_path, "r") as f:
        if "model_weights" not in f or "pred_net_1" not in f["model_weights"]:
            raise ValueError("Unexpected H5 structure (missing /model_weights/pred_net_1): {}".format(h5_path))
        root = f["model_weights"]["pred_net_1"]
        if "pred_net_1" in root:  # some Keras versions
            root = root["pred_net_1"]
        if block_name not in root:
            raise ValueError("Block '{}' not found in {}".format(block_name, h5_path))
        grp = root[block_name]
        Wraw = _find_kernel_in_group(grp)
        if Wraw is None:
            raise ValueError("No kernel in '{}' (E blocks have none)".format(block_name))

    s = Wraw.shape
    if len(s) != 4:
        raise ValueError("Kernel for '{}' not 4D: {}".format(block_name, s))
    # Keras: (kH,kW,Cin,Cout) -> transpose to (Cout,Cin,kH,kW)
    if s[0] <= 63 and s[1] <= 63:
        kH, kW, Cin, Cout = s
        W = np.transpose(Wraw, (3,2,0,1)).astype(np.float32)
    else:  # already (Cout,Cin,kH,kW)
        Cout, Cin, kH, kW = s
        W = Wraw.astype(np.float32)
    return W, (Cout, Cin, kH, kW)

# ---- Kernel tiles (std: per-filter norm, RGB if Cin==3, else L2) ----
def kernel_tiles_std(W):
    Cout, Cin, kH, kW = W.shape
    tiles = []
    if Cin == 3:
        for j in range(Cout):
            R = norm01(W[j,0]); G = norm01(W[j,1]); B = norm01(W[j,2])
            tiles.append(np.stack([R,G,B], axis=2))
    else:
        for j in range(Cout):
            K = np.sqrt((W[j]**2).sum(axis=0))   # L2 collapse over Cin
            tiles.append(norm01(K))
    return tiles

def mosaic_from_tiles_nearest(tiles, cols, tile_sz=(64,64)):
    n = len(tiles)
    rows = int(math.ceil(float(n)/cols))
    is_rgb = (tiles[0].ndim == 3 and tiles[0].shape[2] == 3)
    Ht, Wt = tile_sz[1], tile_sz[0]
    interp = cv2.INTER_NEAREST
    if is_rgb:
        grid = np.zeros((rows*(Ht+2), cols*(Wt+2), 3), dtype=np.float32)
        for i, t in enumerate(tiles):
            r, c = i // cols, i % cols
            t = cv2.resize(t, (Wt, Ht), interpolation=interp)
            y0, x0 = r*(Ht+2), c*(Wt+2)
            grid[y0:y0+Ht, x0:x0+Wt, :] = t
    else:
        grid = np.zeros((rows*(Ht+2), cols*(Wt+2)), dtype=np.float32)
        for i, t in enumerate(tiles):
            r, c = i // cols, i % cols
            t = cv2.resize(t, (Wt, Ht), interpolation=interp)
            y0, x0 = r*(Ht+2), c*(Wt+2)
            grid[y0:y0+Ht, x0:x0+Wt] = t
    return grid

# ---- Ahat0 frame as input ----
def load_ahat0_frame(npy_path, sample_idx=0, timestep=1):
    if not os.path.exists(npy_path):
        raise IOError("Missing Ahat0.npy at {}".format(npy_path))
    X = np.load(npy_path)  # (B,T,H,W,C)
    frame = X[sample_idx, timestep]  # (H,W,C)
    img = frame.astype(np.float32)
    m, M = img.min(), img.max()
    if M > 1.0 or m < 0.0:
        img = norm01(img)
    if img.ndim == 2:
        return img[..., None]
    if img.shape[-1] in (1, 3):
        return img
    # more than 3 channels -> take first as gray
    return img[..., :1]

# ---- Linear conv (fallback) ----
def linear_response(img, W):
    if img.ndim == 2:
        img = img[..., None]
    H, Wd, Cin_img = img.shape
    Cout, Cin_w, kH, kW = W.shape
    if Cin_img != Cin_w:
        if Cin_w == 1 and Cin_img == 3:
            gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
            img = gray[..., None]
            Cin_img = 1
        else:
            raise ValueError("Image channels {} != kernel Cin {}".format(Cin_img, Cin_w))
    resp = np.zeros((H, Wd, Cout), dtype=np.float32)
    for j in range(Cout):
        acc = np.zeros((H, Wd), dtype=np.float32)
        for i in range(Cin_w):
            acc += cv2.filter2D(img[..., i], cv2.CV_32F, W[j,i])
        resp[..., j] = acc
    return resp

# ---- Tiles for activations (std: per-channel norm, grayscale) ----
def featuremap_tiles_from_npy_gray(npy_path, sample_idx, timestep, tile_sz=(64,64)):
    if not os.path.exists(npy_path):
        return None
    X = np.load(npy_path)                       # (B,T,H,W,C)
    fmap = np.transpose(X, (0,1,4,2,3))[sample_idx, timestep]  # (C,H,W)
    tiles = []
    for c in range(fmap.shape[0]):
        ch = norm01(fmap[c])
        tiles.append(cv2.resize(ch, tile_sz, interpolation=cv2.INTER_NEAREST))
    return tiles

def tiles_from_responses_gray(R, tile_sz=(64,64)):
    H, Wd, C = R.shape
    tiles = []
    for j in range(C):
        ch = norm01(R[..., j])
        tiles.append(cv2.resize(ch, tile_sz, interpolation=cv2.INTER_NEAREST))
    return tiles

def make_one_panel(h5_path, npy_dir, level_idx, timestep, out_dir):
    """Build one panel for a specific Ahat{level_idx} at given timestep."""
    makedirs_safe(out_dir)

    block_name = "layer_ahat_{}".format(level_idx)

    # 1) Kernels
    W, (Cout, Cin, kH, kW) = load_block_kernel(h5_path, block_name)
    k_tiles = kernel_tiles_std(W)
    cols   = grid_cols(Cout, MAX_COLS)
    k_grid = mosaic_from_tiles_nearest(k_tiles, cols, TILE_SZ)

    # 2) Input (Ahat0 frame)
    ahat0_path = os.path.join(npy_dir, "Ahat0.npy")
    img = load_ahat0_frame(ahat0_path, SAMPLE_IDX, timestep)

    # 3) Activation: prefer true feature map for this level
    fmap_path = os.path.join(npy_dir, "Ahat{}.npy".format(level_idx))
    act_tiles = featuremap_tiles_from_npy_gray(fmap_path, SAMPLE_IDX, timestep, TILE_SZ)
    if act_tiles is None:
        # fallback: linear conv response
        R = linear_response(img, W)
        act_tiles = tiles_from_responses_gray(R, TILE_SZ)
        act_title = "activation"
    else:
        act_title = "activation"
    act_grid = mosaic_from_tiles_nearest(act_tiles, cols, TILE_SZ)

    # 4) Draw three-column panel
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: input
    if img.shape[-1] == 3:
        axes[0].imshow(img)
    else:
        axes[0].imshow(img[...,0], cmap='gray')
    axes[0].set_title("Input (Ahat0) | s{} t{}".format(SAMPLE_IDX, timestep), fontsize=11, pad=6)
    axes[0].axis('off')

    # Middle: kernels
    if k_grid.ndim == 3:
        axes[1].imshow(k_grid)
        kdesc = "RGB kernels"
    else:
        axes[1].imshow(k_grid, cmap='gray')
        kdesc = "kernels"
    axes[1].set_title("{} | {}".format(block_name, kdesc), fontsize=11, pad=6)
    axes[1].axis('off')

    # Right: activations
    axes[2].imshow(act_grid, cmap='gray')
    axes[2].set_title("{} | {}".format(block_name, act_title), fontsize=11, pad=6)
    axes[2].axis('off')

    plt.suptitle("input image-> {} kernels -> {}  (Cout={}, Cin={}, k={}x{})".format(
        block_name, act_title, Cout, Cin, kH, kW), fontsize=14, y=1.04)
    plt.tight_layout(rect=[0,0,1,0.96])

    out_path = os.path.join(out_dir, "{}_t{}.png".format(block_name, timestep))
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()
    print("[Saved] {}".format(out_path))

def main():
    makedirs_safe(SAVE_DIR)
    for spec in RUN_SPECS:
        layer_tag = spec["layer_tag"]                  # "layer4" or "layer7"
        levels    = spec["levels"]

        h5_path = H5_PATH_TEMPLATE.format(layer_tag=layer_tag)
        npy_dir = NPY_DIR_TEMPLATE.format(layer_tag=layer_tag)
        out_dir = os.path.join(SAVE_DIR, layer_tag)
        makedirs_safe(out_dir)

        if not os.path.exists(h5_path):
            print("[Skip ALL for {}] Missing H5: {}".format(layer_tag, h5_path))
            continue
        if not os.path.exists(npy_dir):
            print("[Skip ALL for {}] Missing NPY dir: {}".format(layer_tag, npy_dir))
            continue

        for t in TIMESTEPS:
            for lvl in levels:
                try:
                    make_one_panel(h5_path, npy_dir, lvl, t, out_dir)
                except Exception as e:
                    print("[Skip] {} Ahat{} t{} due to: {}".format(layer_tag, lvl, t, str(e)))

if __name__ == "__main__":
    main()
