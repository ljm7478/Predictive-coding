"""
Time-Series Visualization: AllLayers only
- All layers (L0-L3)
- Average across all units
- Three time segments: Early / Middle / Late movie
- Y-axis range matched between Prediction and Error for fair comparison
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

# Total frames: 22550, FPS: 25
TOTAL_FRAMES = 22550
FPS = 25

# Split into 3 segments (Early / Middle / Late)
SEGMENT_SIZE = TOTAL_FRAMES // 3
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


def plot_all_layers(ahat_path, error_path, save_dir):
    """
    Create AllLayers plots with matched Y-axis range.
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
    
    # --- Plot: All Layers, One Segment per Figure ---
    for seg_name, (t_start, t_end) in SEGMENTS.items():
        print("\nPlotting segment: {}".format(seg_name))
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 14))
        
        time_sec = np.arange(t_end - t_start) / FPS  # Convert to seconds
        
        # Calculate GLOBAL Y-axis range across ALL layers for this segment
        global_min = float('inf')
        global_max = float('-inf')
        
        for l in range(4):
            ahat_segment = ahat_avg[l][t_start:t_end]
            error_segment = error_avg[l][t_start:t_end]
            
            global_min = min(global_min, ahat_segment.min(), error_segment.min())
            global_max = max(global_max, ahat_segment.max(), error_segment.max())
        
        # Add padding (10%)
        y_range = global_max - global_min
        global_min -= y_range * 0.05
        global_max += y_range * 0.05
        
        print("  Global Y-axis range: [{:.2f}, {:.2f}]".format(global_min, global_max))
        
        # Plot each layer with GLOBAL range
        for l in range(4):
            ahat_segment = ahat_avg[l][t_start:t_end]
            error_segment = error_avg[l][t_start:t_end]
            
            # Ahat (left column)
            ax = axes[l, 0]
            ax.plot(time_sec, ahat_segment, 'b-', linewidth=0.8, alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title('Ahat L{} - Average'.format(l), fontsize=11, fontweight='bold')
            ax.set_ylabel('Z-scored')
            ax.set_ylim(global_min, global_max)  # GLOBAL Y-axis
            if l == 3:
                ax.set_xlabel('Time (seconds)')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f0f0ff')
            
            # Error (right column)
            ax = axes[l, 1]
            ax.plot(time_sec, error_segment, 'r-', linewidth=0.8, alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_title('Error L{} - Average'.format(l), fontsize=11, fontweight='bold')
            ax.set_ylabel('Z-scored')
            ax.set_ylim(global_min, global_max)  # GLOBAL Y-axis (same!)
            if l == 3:
                ax.set_xlabel('Time (seconds)')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#fff0f0')
        
        plt.suptitle('Prediction vs Error: {} (Frames {}-{})'.format(seg_name, t_start, t_end), 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save
        seg_filename = seg_name.split(' ')[0].lower()  # 'early', 'middle', 'late'
        save_path = os.path.join(save_dir, 'AllLayers_{}.png'.format(seg_filename))
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
        plot_all_layers(ahat_path, error_path, SAVE_DIR)
    else:
        print("Files not found!")
        if not os.path.exists(ahat_path):
            print("  Missing: {}".format(ahat_path))
        if not os.path.exists(error_path):
            print("  Missing: {}".format(error_path))