"""
Comprehensive Time-Series Visualization
- All layers (L0-L3)
- Average across all units
- Three time segments: Early / Middle / Late movie
"""

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4/'
SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_activation_trend/test_GUMP/pt_kitti_ft_Titanic_Lall_nt150/original/layer4/'
SEG_NAME = 'seg0'

# Total frames: 22550, FPS: 25
# 22550 / 25 = 902 seconds = ~15 minutes
TOTAL_FRAMES = 22550
FPS = 25

# Split into 3 segments (Early / Middle / Late)
SEGMENT_SIZE = TOTAL_FRAMES // 3  # ~7516 frames each (~5 min)
SEGMENTS = {
    'Early (0-5 min)': (0, SEGMENT_SIZE),
    'Middle (5-10 min)': (SEGMENT_SIZE, 2 * SEGMENT_SIZE),
    'Late (10-15 min)': (2 * SEGMENT_SIZE, TOTAL_FRAMES)
}


def get_average_timeseries(data):
    """Get average time-series across all units."""
    B, T, H, W, C = data.shape
    data_flat = data.reshape(B, T, -1)[0]  # (T, features)
    avg_ts = np.mean(data_flat, axis=1)  # (T,)
    return avg_ts


def z_score(series):
    """Standardize to mean=0, std=1."""
    return (series - np.mean(series)) / (np.std(series) + 1e-12)


def plot_all_layers_all_segments(ahat_path, error_path, save_dir):
    """
    Create plots for all layers (L0-L3) and all time segments (Early/Middle/Late).
    """
    print("Loading all layers...")
    
    # Load all layers
    ahat_data = {}
    error_data = {}
    
    with h5py.File(ahat_path, 'r') as f:
        for l in range(4):
            key = 'Ahat{}'.format(l)
            if key in f.keys():
                ahat_data[l] = f[key][:]
                print("  Loaded {}: shape {}".format(key, ahat_data[l].shape))
    
    with h5py.File(error_path, 'r') as f:
        for l in range(4):
            key = 'E{}'.format(l)
            if key in f.keys():
                error_data[l] = f[key][:]
                print("  Loaded {}: shape {}".format(key, error_data[l].shape))
    
    # Compute average time-series for all layers
    print("\nComputing average time-series...")
    ahat_avg = {}
    error_avg = {}
    
    for l in range(4):
        ahat_avg[l] = z_score(get_average_timeseries(ahat_data[l]))
        error_avg[l] = z_score(get_average_timeseries(error_data[l]))
        print("  L{}: Ahat range [{:.2f}, {:.2f}], Error range [{:.2f}, {:.2f}]".format(
            l, ahat_avg[l].min(), ahat_avg[l].max(), error_avg[l].min(), error_avg[l].max()))
    
    # Create output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # --- Plot 1: All Layers, One Segment per Figure ---
    for seg_name, (t_start, t_end) in SEGMENTS.items():
        print("\nPlotting segment: {}".format(seg_name))
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        
        time_sec = np.arange(t_end - t_start) / FPS  # Convert to seconds
        
        for l in range(4):
            # Ahat (left column)
            ax = axes[l, 0]
            ax.plot(time_sec, ahat_avg[l][t_start:t_end], 'b-', linewidth=0.8, alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title('Ahat L{} - Average'.format(l), fontsize=11, fontweight='bold')
            ax.set_ylabel('Z-scored')
            if l == 3:
                ax.set_xlabel('Time (seconds)')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f0f0ff')
            
            # Error (right column)
            ax = axes[l, 1]
            ax.plot(time_sec, error_avg[l][t_start:t_end], 'r-', linewidth=0.8, alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title('Error L{} - Average'.format(l), fontsize=11, fontweight='bold')
            ax.set_ylabel('Z-scored')
            if l == 3:
                ax.set_xlabel('Time (seconds)')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#fff0f0')
        
        plt.suptitle('Prediction vs Error: {} (Frames {}-{})'.format(seg_name, t_start, t_end), 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        seg_filename = seg_name.split(' ')[0].lower()  # 'early', 'middle', 'late'
        save_path = os.path.join(save_dir, 'AllLayers_{}.png'.format(seg_filename))
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(save_path))
    
    # --- Plot 2: Overlay for each layer (Ahat vs Error) ---
    for seg_name, (t_start, t_end) in SEGMENTS.items():
        print("\nPlotting overlay for segment: {}".format(seg_name))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        time_sec = np.arange(t_end - t_start) / FPS
        
        for l in range(4):
            ax = axes[l]
            ax.plot(time_sec, ahat_avg[l][t_start:t_end], 'b-', linewidth=0.8, alpha=0.7, label='Prediction (Ahat)')
            ax.plot(time_sec, error_avg[l][t_start:t_end], 'r-', linewidth=0.8, alpha=0.7, label='Error (E)')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title('Layer {} - Overlay'.format(l), fontsize=12, fontweight='bold')
            ax.set_ylabel('Z-scored Activation')
            ax.set_xlabel('Time (seconds)')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Ahat vs Error Overlay: {}'.format(seg_name), fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        seg_filename = seg_name.split(' ')[0].lower()
        save_path = os.path.join(save_dir, 'Overlay_{}.png'.format(seg_filename))
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(save_path))
    
    # --- Plot 3: Layer Comparison (All layers in one plot) ---
    for seg_name, (t_start, t_end) in SEGMENTS.items():
        print("\nPlotting layer comparison for segment: {}".format(seg_name))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        
        time_sec = np.arange(t_end - t_start) / FPS
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # Blue, Green, Orange, Red
        
        # Ahat - all layers
        ax = axes[0]
        for l in range(4):
            ax.plot(time_sec, ahat_avg[l][t_start:t_end], color=colors[l], 
                    linewidth=0.8, alpha=0.7, label='L{}'.format(l))
        ax.set_title('Prediction (Ahat) - All Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z-scored Activation')
        ax.set_xlabel('Time (seconds)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f8ff')
        
        # Error - all layers
        ax = axes[1]
        for l in range(4):
            ax.plot(time_sec, error_avg[l][t_start:t_end], color=colors[l], 
                    linewidth=0.8, alpha=0.7, label='L{}'.format(l))
        ax.set_title('Error (E) - All Layers', fontsize=12, fontweight='bold')
        ax.set_ylabel('Z-scored Activation')
        ax.set_xlabel('Time (seconds)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#fff8f8')
        
        plt.suptitle('Layer Comparison: {}'.format(seg_name), fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        seg_filename = seg_name.split(' ')[0].lower()
        save_path = os.path.join(save_dir, 'LayerComparison_{}.png'.format(seg_filename))
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print("  Saved: {}".format(save_path))
    
    print("\n" + "="*60)
    print("All plots saved to: {}".format(save_dir))
    print("="*60)


if __name__ == "__main__":
    ahat_path = os.path.join(BASE_DIR, '{}_Ahat.h5'.format(SEG_NAME))
    error_path = os.path.join(BASE_DIR, '{}_E.h5'.format(SEG_NAME))
    
    print("Input files:")
    print("  Ahat: {}".format(ahat_path))
    print("  Error: {}".format(error_path))
    print()
    
    if os.path.exists(ahat_path) and os.path.exists(error_path):
        plot_all_layers_all_segments(ahat_path, error_path, SAVE_DIR)
    else:
        print("Files not found!")
        if not os.path.exists(ahat_path):
            print("  Missing: {}".format(ahat_path))
        if not os.path.exists(error_path):
            print("  Missing: {}".format(error_path))