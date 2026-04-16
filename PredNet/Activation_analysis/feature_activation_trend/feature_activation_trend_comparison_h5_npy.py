# -*- coding: utf-8 -*-
"""
Performs a direct quantitative analysis comparison between .npy files (No Pooling)
and .h5 files (Pooling) without any intermediate conversion.
"""
import os
import h5py  # Requires h5py library
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# --- Configuration ---

# 1. 'No Pooling' (NPY) configuration
NOPOOL_NPY_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/pt_kitti_ft_Lall_nt150/layer5_nopool/'
NOPOOL_RUN = 'run1'  # The subdir containing the .npy files

# 2. 'Pooling' (H5) configuration
POOLING_H5_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer5_pool/'
POOLING_AHAT_FILE = 'seg0_Ahat.h5'
POOLING_E_FILE = 'seg0_E.h5'

# 3. Output configuration
ANALYSIS_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/feature_activation_comparison/pt_kitti_ft_Lall_nt150/layer5'
# ---------------------

# List of layers we want to analyze
LAYERS_TO_ANALYZE = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'E0', 'E1', 'E2', 'E3']
# LAYERS_TO_ANALYZE = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'Ahat4', 'E0', 'E1', 'E2', 'E3', 'E4']

NUM_LAYERS = 5


def calculate_mean_activation(data):
    """Calculates the mean absolute activation value of the given data."""
    # Data shape is assumed to be (B, T, H, W, C) or similar
    return np.mean(np.abs(data))


def plot_comparison_trends(activations_nopool, activations_pooling, save_path):
    """Plots and saves the trend of activation values across layers for comparison."""
    layers = ['L{}'.format(l) for l in range(NUM_LAYERS)]
    
    # Get activations for 'No Pooling'
    ahat_nopool = [activations_nopool.get('Ahat{}'.format(l), np.nan) for l in range(NUM_LAYERS)]
    e_nopool = [activations_nopool.get('E{}'.format(l), np.nan) for l in range(NUM_LAYERS)]
    
    # Get activations for 'Pooling'
    ahat_pooling = [activations_pooling.get('Ahat{}'.format(l), np.nan) for l in range(NUM_LAYERS)]
    e_pooling = [activations_pooling.get('E{}'.format(l), np.nan) for l in range(NUM_LAYERS)]

    plt.figure(figsize=(12, 7))
    
    # Plot 'No Pooling' (dashed lines)
    plt.plot(layers, ahat_nopool, 'bo--', label='Ahat (No Pooling)')
    plt.plot(layers, e_nopool, 'ro--', label='E (No Pooling)')
    
    # Plot 'Pooling' (solid lines)
    plt.plot(layers, ahat_pooling, 'bs-', label='Ahat (Pooling)')
    plt.plot(layers, e_pooling, 'rs-', label='E (Pooling)')
    
    plt.title('Mean Activation Comparison (Pooling vs. No Pooling)')
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Activation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')  # Log scale is helpful for convergence
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print "Activation comparison plot saved to: {}".format(save_path)


def analyze_npy_run(base_dir, run_name):
    """Loads all .npy files for a given run and returns an activation dict."""
    print "Analyzing NPY run: %s at %s" % (run_name, base_dir)
    run_dir = os.path.join(base_dir, run_name)
    activations = {}
    
    if not os.path.exists(run_dir):
        print "  Error: NPY Run directory not found at {}".format(run_dir)
        return activations

    for layer_name in LAYERS_TO_ANALYZE:
        npy_path = os.path.join(run_dir, "{}.npy".format(layer_name))
        if os.path.exists(npy_path):
            print "  Loading and calculating {}...".format(layer_name)
            data = np.load(npy_path)
            activations[layer_name] = calculate_mean_activation(data)
        else:
            print "  Warning: Could not find {}. Skipping.".format(npy_path)
            
    return activations


def analyze_h5_run(h5_dir, ahat_filename, e_filename):
    """Loads data directly from H5 files and returns an activation dict."""
    print "Analyzing H5 run at %s" % h5_dir
    activations = {}
    
    # --- Process Ahat ---
    ahat_h5_path = os.path.join(h5_dir, ahat_filename)
    try:
        with h5py.File(ahat_h5_path, 'r') as f_ahat:
            print "  Processing %s..." % ahat_filename
            for layer_name in [l for l in LAYERS_TO_ANALYZE if 'Ahat' in l]:
                if layer_name in f_ahat:
                    print "    Extracting and calculating {}...".format(layer_name)
                    data = f_ahat[layer_name][:]  # Load data into memory
                    activations[layer_name] = calculate_mean_activation(data)
                else:
                    print "    Warning: Layer %s not found in %s" % (layer_name, ahat_filename)
    except IOError as e:
        print "  Error: Could not open H5 file at %s. (%s)" % (ahat_h5_path, e)
        return activations  # Return empty dict on error

    # --- Process E ---
    e_h5_path = os.path.join(h5_dir, e_filename)
    try:
        with h5py.File(e_h5_path, 'r') as f_e:
            print "  Processing %s..." % e_filename
            for layer_name in [l for l in LAYERS_TO_ANALYZE if 'E' in l]:
                if layer_name in f_e:
                    print "    Extracting and calculating {}...".format(layer_name)
                    data = f_e[layer_name][:]  # Load data into memory
                    activations[layer_name] = calculate_mean_activation(data)
                else:
                    print "    Warning: Layer %s not found in %s" % (layer_name, e_filename)
    except IOError as e:
        print "  Error: Could not open H5 file at %s. (%s)" % (e_h5_path, e)
        
    return activations


if __name__ == "__main__":
    print "Starting quantitative analysis comparison..."
    if not os.path.exists(ANALYSIS_SAVE_DIR):
        try:
            os.makedirs(ANALYSIS_SAVE_DIR)
        except OSError as e:
            print "Warning: Could not create save directory (may exist): %s" % e

    # 1. Analyze the 'No Pooling' run (from .npy files)
    activations_nopool = analyze_npy_run(NOPOOL_NPY_DIR, NOPOOL_RUN)
    
    print "---"
    
    # 2. Analyze the 'Pooling' run (from .h5 files)
    activations_pooling = analyze_h5_run(POOLING_H5_DIR, POOLING_AHAT_FILE, POOLING_E_FILE)

    # 3. Plot and save the comparison
    if not activations_nopool or not activations_pooling:
        print "\nError: Could not load data for one or both runs. Cannot create comparison plot."
        exit()
        
    plot_filename = 'activation_comparison_pooling_vs_nopool_direct2.png'
    plot_path = os.path.join(ANALYSIS_SAVE_DIR, plot_filename)
    
    plot_comparison_trends(activations_nopool, activations_pooling, plot_path)

    print "\nAnalysis complete. Comparison plot saved in: %s" % ANALYSIS_SAVE_DIR