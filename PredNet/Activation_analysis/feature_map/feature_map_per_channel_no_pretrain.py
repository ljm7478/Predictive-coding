import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend, suitable for servers
import matplotlib.pyplot as plt
import cv2
import math
import re

# --- Configuration ---
BASE_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump'
PLOTS_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/activation_verification_jm/featureMap/original_layer4/no_pretrain_same_kitti_pt_setting'

RUN_TO_PLOT = 'run1'
TIMESTEPS_TO_PLOT = [5000]

def _flatten_axes(axes):
    """Safely flatten matplotlib axes to a 1D list."""
    return np.atleast_1d(axes).ravel()

def compare_ahat_vs_e(layer_data_dict, run_name, timestep, save_dir):
    """
    Compares Prediction (Ahat) and Error (E) for all layers side-by-side.
    """
    print("  Plotting Ahat vs. E comparison for Timestep {}...".format(timestep))
    num_layers = 4  # Assuming a 4-layer model

    # Determine the max number of channels to plot for a consistent grid size
    max_channels = 0
    for l in range(num_layers):
        if 'Ahat{}'.format(l) in layer_data_dict:
            # Data shape is (B, T, C, H, W)
            max_channels = max(max_channels, layer_data_dict['Ahat{}'.format(l)].shape[2])

    fig, axes = plt.subplots(num_layers * 2, max_channels, figsize=(max_channels * 1.2, num_layers * 2.4))
    if max_channels == 1: # Handle single channel case
        axes = axes.reshape(num_layers * 2, 1)

    fig.suptitle('Prediction (Ahat) vs. Error (E) | Run: {} | Timestep: {}'.format(run_name, timestep), fontsize=16)

    for l in range(num_layers):
        ahat_key = 'Ahat{}'.format(l)
        e_key = 'E{}'.format(l)

        if ahat_key in layer_data_dict:
            ahat_map = layer_data_dict[ahat_key][0, timestep]  # (C, H, W)
            for i in range(ahat_map.shape[0]):
                axes[l * 2, i].imshow(ahat_map[i], cmap='viridis', interpolation='none')
            axes[l * 2, 0].set_ylabel('Ahat{}\n(Pred)'.format(l), rotation=0, labelpad=40, verticalalignment='center')

        if e_key in layer_data_dict:
            e_map = layer_data_dict[e_key][0, timestep]  # (C, H, W)
            for i in range(e_map.shape[0]):
                axes[l * 2 + 1, i].imshow(e_map[i], cmap='plasma', interpolation='none')
            axes[l * 2 + 1, 0].set_ylabel('E{}\n(Error)'.format(l), rotation=0, labelpad=40, verticalalignment='center')
    
    # Turn off unused axes and ticks for all axes
    for ax in _flatten_axes(axes):
        if not ax.images:
            ax.axis('off')
        else:
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    comparison_save_dir = os.path.join(save_dir, 'Ahat_vs_E_comparison')
    if not os.path.exists(comparison_save_dir):
        os.makedirs(comparison_save_dir)
        
    save_path = os.path.join(comparison_save_dir, 'comparison_t{}.png'.format(timestep))
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("    Saved Ahat vs. E plot to: {}".format(save_path))


def visualize_all_channels(data, layer_name, timestep, save_dir, resize_to=(64, 64)):
    """
    Visualizes all channels of a single feature map (e.g., all channels of 'E3').
    """
    print("  Visualizing all channels for {} at Timestep {}...".format(layer_name, timestep))

    # CRITICAL: Assumes data is already in channels_first format (B, T, C, H, W)
    data = np.transpose(data, (0, 1, 4, 2, 3))
    fmap = data[0, timestep]  # Get data for the specific timestep -> (C, H, W)
    num_channels = fmap.shape[0]

    n_cols = min(16, num_channels)
    n_rows = int(np.ceil(float(num_channels) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = _flatten_axes(axes)

    for idx in range(num_channels):
        ch_map = fmap[idx]
        # Normalize for visualization
        denom = (ch_map.max() - ch_map.min() + 1e-5)
        norm = (ch_map - ch_map.min()) / denom
        img_resized = cv2.resize(norm, resize_to, interpolation=cv2.INTER_CUBIC)
        axes[idx].imshow(img_resized, cmap='viridis')
        axes[idx].set_title("Ch {}".format(idx), fontsize=8, pad=2)
        axes[idx].axis('off')

    # Turn off empty subplots
    for k in range(num_channels, len(axes)):
        axes[k].axis('off')

    fig.suptitle('All Channels: {} | Timestep {}'.format(layer_name, timestep), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Create subdirectories for each feature family (e.g., Ahat, E)
    layer_family_match = re.match(r'^([A-Za-z]+)', layer_name)
    if layer_family_match:
        layer_family = layer_family_match.group(1)
        family_save_dir = os.path.join(save_dir, 'all_channels', layer_family)
        if not os.path.exists(family_save_dir):
            os.makedirs(family_save_dir)

        save_path = os.path.join(family_save_dir, '{}_t{}.png'.format(layer_name, timestep))
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("    Saved all-channels plot to: {}".format(save_path))


if __name__ == "__main__":
    print("Starting feature map visualization...")
    if not os.path.exists(PLOTS_SAVE_DIR):
        os.makedirs(PLOTS_SAVE_DIR)

    # --- Load all necessary feature data into a dictionary ---
    layers_to_load = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'E0', 'E1', 'E2', 'E3']
    layer_data_dict = {}

    run_dir = os.path.join(BASE_RESULTS_DIR, RUN_TO_PLOT)
    if not os.path.exists(run_dir):
        print("Error: Run directory not found at {}".format(run_dir))
        exit()

    for layer_name in layers_to_load:
        npy_path = os.path.join(run_dir, "{}.npy".format(layer_name))
        if os.path.exists(npy_path):
            print("Loading {}...".format(npy_path))
            # The loaded data should have shape (1, nt, C, H, W)
            layer_data_dict[layer_name] = np.load(npy_path)
        else:
            print("Warning: Could not find {}. Skipping.".format(npy_path))

    if not layer_data_dict:
        print("Error: No data loaded. Please check BASE_RESULTS_DIR and RUN_TO_PLOT.")
        exit()

    # --- Run Visualizations for each specified timestep ---
    for t in TIMESTEPS_TO_PLOT:
        # 1. Direct Comparison of Ahat vs. E (Most Important)
        # compare_ahat_vs_e(layer_data_dict, RUN_TO_PLOT, t, PLOTS_SAVE_DIR)

        # 2. Visualize all channels for all loaded layers
        for layer_name in sorted(layer_data_dict.keys()):
             visualize_all_channels(layer_data_dict[layer_name], layer_name, t, PLOTS_SAVE_DIR)

    print("\nVisualization complete. Plots saved in: {}".format(PLOTS_SAVE_DIR))