# -*- coding: utf-8 -*-
"""
Simple visualization: Actual, Predicted, Error for each condition.
Based on original PredNet plotting style.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py

# --- Configuration ---
ORIGINAL_INPUT_PATH = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/longer-time_result/prednet_original/h5/pt_kitti_ft_Lall_nt150/original_video.npy'
ANALYSIS_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_prediction_error_frame_viz/original_prednet/pt_kitti_ft_Titanic_Lall_nt150/longer_time_visualization/'
FRAME_START = 1400
NT = 20  # Number of frames to show in one image

BASE_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/longer-time_result/prednet_original/h5/pt_kitti_ft_Lall_nt150'
RUN_ID = 'seg0'

CONDITIONS = [
    {'path': '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4', 'label': '0.04s_prediction'},
    {'path': os.path.join(BASE_RESULTS_DIR, 'indirect_1s_extrap/layer4'), 'label': '1s_prediction'},
    {'path': os.path.join(BASE_RESULTS_DIR, 'indirect_2s_extrap/layer4'), 'label': '2s_prediction'},
    {'path': os.path.join(BASE_RESULTS_DIR, 'indirect_5s_extrap/layer4'), 'label': '5s_prediction'},
    {'path': os.path.join(BASE_RESULTS_DIR, 'indirect_10s_extrap/layer4'), 'label': '10s_prediction'},
]
# --- End Configuration ---


def load_original_video(path):
    if not os.path.exists(path):
        print("Not found: {0}".format(path))
        return None
    print("Loading: {0}".format(path))
    data = np.load(path)
    print("  Shape: {0}".format(data.shape))
    # (1, T, H, W, C) -> (T, H, W, C)
    if data.ndim == 5:
        data = data[0]
    return data


def load_ahat_and_e(base_dir, run_id):
    """Load Ahat0 and E0 from h5 files."""
    ahat = None
    error = None
    
    ahat_file = os.path.join(base_dir, '{0}_Ahat.h5'.format(run_id))
    if os.path.exists(ahat_file):
        with h5py.File(ahat_file, 'r') as hf:
            if 'Ahat0' in hf:
                ahat = hf['Ahat0'][:]
                print("  Ahat0 shape: {0}".format(ahat.shape))
                # (1, T, H, W, C) -> (T, H, W, C)
                if ahat.ndim == 5:
                    ahat = ahat[0]
    
    e_file = os.path.join(base_dir, '{0}_E.h5'.format(run_id))
    if os.path.exists(e_file):
        with h5py.File(e_file, 'r') as hf:
            if 'E0' in hf:
                error = hf['E0'][:]
                print("  E0 shape: {0}".format(error.shape))
                # (1, T, H, W, C) -> (T, H, W, C)
                if error.ndim == 5:
                    error = error[0]
    
    return ahat, error


def normalize_for_display(img):
    """Normalize image to [0, 1] for display."""
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def plot_actual_predicted_error(X_actual, X_hat, X_error, save_path, nt, start_frame=0):
    """
    Plot 3 rows: Actual, Predicted, Error
    Data shape: (T, H, W, C)
    """
    # Get aspect ratio from data
    aspect_ratio = float(X_actual.shape[1]) / X_actual.shape[2]  # H / W
    
    plt.figure(figsize=(nt, 3 * aspect_ratio))
    gs = gridspec.GridSpec(3, nt)
    gs.update(wspace=0., hspace=0.)
    
    for t in range(nt):
        frame_idx = start_frame + t
        
        # Row 0: Actual
        ax = plt.subplot(gs[t])
        actual_frame = normalize_for_display(X_actual[frame_idx])
        plt.imshow(actual_frame, interpolation='none')
        ax.axis('off')
        if t == 0:
            ax.set_ylabel('Actual', fontsize=10)
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Row 1: Predicted
        ax = plt.subplot(gs[t + nt])
        pred_frame = normalize_for_display(X_hat[frame_idx, :, :, :3])  # Take first 3 channels
        plt.imshow(pred_frame, interpolation='none')
        ax.axis('off')
        if t == 0:
            ax.set_ylabel('Predicted', fontsize=10)
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        # Row 2: Error
        ax = plt.subplot(gs[t + 2*nt])
        # E0 has 6 channels, take mean for grayscale display
        error_frame = np.mean(X_error[frame_idx], axis=2)  # (H, W)
        error_frame = normalize_for_display(error_frame)
        plt.imshow(error_frame, cmap='gray', interpolation='none')
        ax.axis('off')
        if t == 0:
            ax.set_ylabel('Error', fontsize=10)
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()
    print("Saved: {0}".format(save_path))


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Visualization: Actual / Predicted / Error")
    print("=" * 60)
    
    if not os.path.exists(ANALYSIS_SAVE_DIR):
        os.makedirs(ANALYSIS_SAVE_DIR)
    
    # Load original video
    print("\nLoading original video...")
    X_actual = load_original_video(ORIGINAL_INPUT_PATH)
    
    if X_actual is None:
        print("ERROR: Could not load original video!")
        exit(1)
    
    # Process each condition
    for cond in CONDITIONS:
        print("\n" + "-" * 40)
        print("Processing: {0}".format(cond['label']))
        print("-" * 40)
        
        X_hat, X_error = load_ahat_and_e(cond['path'], RUN_ID)
        
        if X_hat is None or X_error is None:
            print("Warning: Missing data for {0}".format(cond['label']))
            continue
        
        save_path = os.path.join(ANALYSIS_SAVE_DIR, '{0}.png'.format(cond['label']))
        
        plot_actual_predicted_error(
            X_actual, X_hat, X_error,
            save_path, 
            nt=NT, 
            start_frame=FRAME_START
        )
    
    print("\n" + "=" * 60)
    print("Done! Output: {0}".format(ANALYSIS_SAVE_DIR))
    print("=" * 60)