# -*- coding: utf-8 -*-
"""
Calculates and plots the mean activation trends for two separate PredNet models
(e.g., fine-tuned vs. from-scratch) on individual graphs, ensuring the y-axis
scale is consistent across both plots for fair comparison.
Loads data directly from HDF5 files.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import h5py # Import h5py for reading HDF5 files

# --- Configuration ---
# --- Model 1: Gump test---
GUMP_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4/'
GUMP_RUN = 'seg0' # Assumes segX_Ahat.h5/segX_E.h5 naming
GUMP_LABEL = 'Gump Test'

# --- Model 2: Budapest test---
BP_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/08_Budapest/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4/'
BP_RUN = 'seg1' # Assumes segX_Ahat.h5/segX_E.h5 naming
BP_LABEL = 'Budapest Test'

# --- General Settings ---
ANALYSIS_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_activation_trend/test_GUMP/compare_diff_test_data_pt_kitti_ft_Titanic'
NUM_LAYERS = 4 # Assuming 4 layers for both models
# --- End Configuration ---


def calculate_mean_activation(data):
    """Calculates the mean absolute activation value."""
    if data.ndim == 4:
        data = np.expand_dims(data, axis=0)
    return np.mean(np.abs(data))

def load_activations_from_h5(base_dir, run_id, num_layers):
    """Loads Ahat and E activations from HDF5 files for a given run."""
    activations = {}
    ahat_file = os.path.join(base_dir, '{}_Ahat.h5'.format(run_id))
    if os.path.exists(ahat_file):
        print("  Loading Ahat from: {}".format(ahat_file))
        with h5py.File(ahat_file, 'r') as hf:
            for l in range(num_layers):
                layer_key = '/Ahat{}'.format(l)
                if layer_key in hf:
                    activations['Ahat{}'.format(l)] = calculate_mean_activation(hf[layer_key][:])
                else: print("  Warning: Key '{}' not found.".format(layer_key))
    else: print("  Warning: File not found: {}".format(ahat_file))
        
    e_file = os.path.join(base_dir, '{}_E.h5'.format(run_id))
    if os.path.exists(e_file):
        print("  Loading E from: {}".format(e_file))
        with h5py.File(e_file, 'r') as hf:
            for l in range(num_layers):
                layer_key = '/E{}'.format(l)
                if layer_key in hf:
                    activations['E{}'.format(l)] = calculate_mean_activation(hf[layer_key][:])
                else: print("  Warning: Key '{}' not found.".format(layer_key))
    else: print("  Warning: File not found: {}".format(e_file))
    return activations


def plot_single_activation_trend(activation_dict, label, save_path, ylim=None):
    """Plots activation trends for a single model."""
    num_layers = 4
    layers = ['L{}'.format(l) for l in range(num_layers)]
    ahat_activations = [activation_dict.get('Ahat{}'.format(l), np.nan) for l in range(num_layers)]
    e_activations = [activation_dict.get('E{}'.format(l), np.nan) for l in range(num_layers)]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, ahat_activations, 'bo-', label='Prediction (Ahat) Activation')
    plt.plot(layers, e_activations, 'ro-', label='Error (E) Activation')
    
    plt.title('Mean Feature Activation across Layers ({})'.format(label))
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Activation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')
    
    # Apply common y-axis limits if provided
    if ylim:
        plt.ylim(ylim)
        
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Activation plot saved to: {}".format(save_path))


if __name__ == "__main__":
    print("Starting activation comparison (separate plots)...")
    if not os.path.exists(ANALYSIS_SAVE_DIR):
        os.makedirs(ANALYSIS_SAVE_DIR)

    # --- Load data for both models ---
    print("\nLoading activations for Model 1: {}".format(GUMP_LABEL))
    GUMP_activations = load_activations_from_h5(GUMP_RESULTS_DIR, GUMP_RUN, NUM_LAYERS)
    print("\nLoading activations for Model 2: {}".format(BP_LABEL))
    BP_activations = load_activations_from_h5(BP_RESULTS_DIR, BP_RUN, NUM_LAYERS)

    # --- Determine common y-axis limits ---
    all_vals = []
    if GUMP_activations:
        all_vals.extend(GUMP_activations.values())
    if BP_activations:
        all_vals.extend(BP_activations.values())
        
    valid_vals = [v for v in all_vals if not np.isnan(v) and v > 0]
    common_ylim = None
    if valid_vals:
         min_val = min(valid_vals) * 0.5
         max_val = max(valid_vals) * 2
         common_ylim = (min_val, max_val)

    # --- Plot graphs separately with common y-limits ---
    if GUMP_activations:
        plot_path_ft = os.path.join(ANALYSIS_SAVE_DIR, 'GUMP.png')
        plot_single_activation_trend(GUMP_activations, GUMP_LABEL, plot_path_ft, ylim=common_ylim)
    else:
        print("\nSkipping plot for fine-tuned model: No data loaded.")

    if BP_activations:
        plot_path_fs = os.path.join(ANALYSIS_SAVE_DIR, 'BP.png')
        plot_single_activation_trend(BP_activations, BP_LABEL, plot_path_fs, ylim=common_ylim)
    else:
        print("\nSkipping plot for from-scratch model: No data loaded.")

    print("\nAnalysis complete.")
