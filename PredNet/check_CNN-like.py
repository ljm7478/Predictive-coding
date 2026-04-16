# -*- coding: utf-8 -*-
from __future__ import print_function, division # Python 2.7 compatibility (print function and float division)

import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_maps(file_path, layer_index=0, sample_idx=0, time_idx=5, num_channels=5):
    """
    Visualizes feature maps from an HDF5 file. (Python 2.7 compatible)
    
    Args:
        file_path: Path to the .h5 file.
        layer_index: Layer index to visualize (depends on file structure).
        sample_idx: Index of the sample in the batch.
        time_idx: Index of the time step (frame).
        num_channels: Number of channels to visualize (first n channels).
    """
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. Check internal keys of the file
            # In Python 2, keys() returns a list, so list() wrapper is optional but safe
            print("File Keys: {}".format(f.keys()))
            
            # Load data (Assumes the first key contains the main data)
            key = f.keys()[0]
            data = f[key][:] 
            
            print("Data Shape: {}".format(data.shape))
            # Expected Shape: (samples, time, channels, height, width) or (samples, time, height, width, channels)
            
            # 2. Check and adjust dimension order (Handle Channels First vs Last)
            # Assuming data is 5D
            if data.ndim == 5:
                # If shape is (Batch, Time, H, W, Channels) -> Transpose to (Batch, Time, Channels, H, W)
                if data.shape[-1] < data.shape[2]: # Heuristic check if channels are at the last dimension
                    print("Detected Channels Last format. Transposing...")
                    data = np.transpose(data, (0, 1, 4, 2, 3))
            
            # Extract specific sample and time step
            # Resulting shape should be (Channels, H, W)
            feature_maps = data[sample_idx, time_idx] 
            
            print("Plotting Sample {}, Time {}".format(sample_idx, time_idx))
            print("Feature Map Range: min={:.5f}, max={:.5f}".format(feature_maps.min(), feature_maps.max()))

            # 3. Visualization
            fig, axes = plt.subplots(1, num_channels, figsize=(15, 3))
            
            # Handle case where num_channels is 1 (axes is not a list)
            if num_channels == 1: 
                axes = [axes]
            
            for i in range(num_channels):
                # Select channel
                fm = feature_maps[i]
                
                # Normalization (Crucial: Scale values to 0-1 for visualization)
                # Ensure float division by converting one operand to float (though __future__ division handles this)
                denom = float(fm.max() - fm.min())
                if denom > 0:
                    fm_norm = (fm - fm.min()) / denom
                else:
                    fm_norm = fm # Prevent division by zero if map is uniform
                
                ax = axes[i]
                ax.imshow(fm_norm, cmap='gray') # Options: 'gray', 'inferno', 'viridis'
                ax.set_title("Ch {}".format(i))
                ax.axis('off')
            
            plt.suptitle("Feature Maps (Sample {}, Time {})".format(sample_idx, time_idx), fontsize=14)
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print("Error occurred: {}".format(e))
        print("Please check the file path or structure.")

# --- Execution ---
if __name__ == "__main__":
    # Updated file path
    file_path = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/CNN_like/h5/pt_kitti_ft_Lall_nt150/layer4/seg0_Ahat.h5'

    # Run visualization
    # You can adjust sample_idx or time_idx if needed
    visualize_feature_maps(file_path, sample_idx=0, time_idx=5, num_channels=5)