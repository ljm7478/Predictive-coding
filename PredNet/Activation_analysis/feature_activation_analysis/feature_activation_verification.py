"""
Deep-Dive Verification:
1. Log-Histogram to check if Prediction is truly 'Zero'.
2. Raster Plot to see the temporal distribution of the 'Active 2.5%'.

(Fixed: Removed invalid arguments for eventplot)
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4/'
SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_activation_analysis/test_GUMP/pt_kitti_ft_Titanic_Lall_nt150/original/layer4/'
SEG_NAME = 'seg0'
LAYER = 3

def plot_verification_analysis(ahat_path, error_path, save_dir, layer_idx=3):
    print(f"Analyzing Layer {layer_idx} for deep verification...")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load Data
    with h5py.File(ahat_path, 'r') as f:
        ahat = f[f'Ahat{layer_idx}'][:]
    with h5py.File(error_path, 'r') as f:
        error = f[f'E{layer_idx}'][:]
        
    # Flatten spatial dims for histogram: (Total_Points,)
    ahat_flat = ahat.flatten()
    error_flat = error.flatten()
    
    # --- 1. Log-Histogram Analysis ---
    print("Generating Log-Histograms...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    epsilon = 1e-12
    
    # Ahat Histogram
    axes[0].hist(np.abs(ahat_flat) + epsilon, bins=100, log=True, color='blue', alpha=0.7)
    axes[0].set_title(f'Prediction (Ahat{layer_idx}) Value Distribution (Log Scale)')
    axes[0].set_xlabel('Absolute Activation Value')
    axes[0].set_ylabel('Frequency (Log)')
    axes[0].axvline(x=1e-9, color='k', linestyle='--', label='Zero Threshold (1e-9)')
    axes[0].legend()
    
    # Error Histogram
    axes[1].hist(np.abs(error_flat) + epsilon, bins=100, log=True, color='red', alpha=0.7)
    axes[1].set_title(f'Error (E{layer_idx}) Value Distribution (Log Scale)')
    axes[1].set_xlabel('Absolute Activation Value')
    axes[1].axvline(x=1e-9, color='k', linestyle='--', label='Zero Threshold (1e-9)')
    axes[1].legend()
    
    plt.suptitle(f'Is Prediction truly "Zero"? (Layer {layer_idx})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Log_Histogram.png'))
    plt.close()

    # --- 2. Raster Plot (Temporal Distribution) ---
    print("Generating Raster Plots...")
    B, T, H, W, C = ahat.shape
    ahat_time = ahat.reshape(B, T, -1)[0]  # (T, Feat)
    error_time = error.reshape(B, T, -1)[0] # (T, Feat)
    
    # Downsample features (random 500 units)
    np.random.seed(42)
    sample_indices = np.random.choice(ahat_time.shape[1], 500, replace=False)
    
    ahat_sample = ahat_time[:, sample_indices]
    error_sample = error_time[:, sample_indices]
    
    # Define "Active" events
    def get_spike_times(data):
        spike_times = []
        # Dynamic threshold: Mean + 2*Std
        threshold = np.mean(np.abs(data)) + 2 * np.std(data)
        threshold = max(threshold, 1e-6) 
        
        for unit_idx in range(data.shape[1]):
            unit_ts = np.abs(data[:, unit_idx])
            spikes = np.where(unit_ts > threshold)[0]
            spike_times.append(spikes)
        return spike_times

    ahat_spikes = get_spike_times(ahat_sample)
    error_spikes = get_spike_times(error_sample)
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # Fix: Removed 'marker' and 'markersize', added 'linelengths'
    # Plot Ahat Raster
    if len(ahat_spikes) > 0:
        axes[0].eventplot(ahat_spikes, color='blue', linelengths=0.8, alpha=0.5)
    axes[0].set_title(f'Prediction (Ahat{layer_idx}) Spike Raster (500 Sample Units)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Unit Index')
    
    # Plot Error Raster
    if len(error_spikes) > 0:
        axes[1].eventplot(error_spikes, color='red', linelengths=0.8, alpha=0.5)
    axes[1].set_title(f'Error (E{layer_idx}) Spike Raster (500 Sample Units)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (Frames)')
    axes[1].set_ylabel('Unit Index')
    
    # Highlight comparison
    axes[0].text(0.02, 0.9, "Are spikes clustered or distributed?", transform=axes[0].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'Raster_Plot.png'))
    plt.close()
    
    print("Verification plots saved.")

if __name__ == "__main__":
    ahat_path = os.path.join(BASE_DIR, f'{SEG_NAME}_Ahat.h5')
    error_path = os.path.join(BASE_DIR, f'{SEG_NAME}_E.h5')
    
    plot_verification_analysis(ahat_path, error_path, SAVE_DIR, LAYER)