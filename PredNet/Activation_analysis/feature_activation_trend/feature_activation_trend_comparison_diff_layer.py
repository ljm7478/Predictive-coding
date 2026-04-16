# -*- coding: utf-8 -*-
"""
Performs a direct quantitative analysis comparison between:
1. Layer 4 model (.h5 files)
2. Layer 5 model (.npy files)
"""
import os
import h5py  # Requires h5py library
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# --- Configuration ---

# 1. 'Layer 5' (NPY) configuration
L5_NPY_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/pt_kitti_ft_Lall_nt150/layer5_nopool/'
L5_RUN = 'run1'  # The subdir containing the .npy files

# 2. 'Layer 4' (H5) configuration
L4_H5_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4/'
L4_AHAT_FILE = 'seg0_Ahat.h5' # Assuming same filename pattern
L4_E_FILE = 'seg0_E.h5'       # Assuming same filename pattern

# 3. Output configuration
ANALYSIS_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/feature_activation_comparison/pt_kitti_ft_Lall_nt150/'
# ---------------------

# Define which layers to analyze for each model
LAYERS_TO_ANALYZE_L4 = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'E0', 'E1', 'E2', 'E3']
LAYERS_TO_ANALYZE_L5 = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'Ahat4', 'E0', 'E1', 'E2', 'E3', 'E4']
MAX_NUM_LAYERS = 5 # The maximum number of layers (L0 to L4)


def calculate_mean_activation(data):
    """Calculates the mean absolute activation value of the given data."""
    return np.mean(np.abs(data))


def plot_comparison_trends(activations_l4, activations_l5, save_path):
    """Plots and saves the trend of activation values across layers for comparison."""
    # X-axis labels go up to the max layer count
    layers = ['L{}'.format(l) for l in range(MAX_NUM_LAYERS)]
    
    # Get activations for 'Layer 4' model
    # Will result in [val0, val1, val2, val3, np.nan]
    ahat_l4 = [activations_l4.get('Ahat{}'.format(l), np.nan) for l in range(MAX_NUM_LAYERS)]
    e_l4 = [activations_l4.get('E{}'.format(l), np.nan) for l in range(MAX_NUM_LAYERS)]
    
    # Get activations for 'Layer 5' model
    # Will result in [val0, val1, val2, val3, val4]
    ahat_l5 = [activations_l5.get('Ahat{}'.format(l), np.nan) for l in range(MAX_NUM_LAYERS)]
    e_l5 = [activations_l5.get('E{}'.format(l), np.nan) for l in range(MAX_NUM_LAYERS)]

    plt.figure(figsize=(12, 7))
    
    # Plot 'Layer 4' (dashed lines) - will stop at L3
    plt.plot(layers, ahat_l4, 'bo--', label='Ahat (Layer 4)')
    plt.plot(layers, e_l4, 'ro--', label='E (Layer 4)')
    
    # Plot 'Layer 5' (solid lines)
    plt.plot(layers, ahat_l5, 'bs-', label='Ahat (Layer 5)')
    plt.plot(layers, e_l5, 'rs-', label='E (Layer 5)')
    
    plt.title('Mean Activation Comparison (Layer 4 vs. Layer 5)')
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Activation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yscale('log')  # Log scale is helpful for convergence
    
    # Ensure x-axis shows all labels L0-L4
    plt.xticks(layers) 
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print ("Activation comparison plot saved to: {}".format(save_path))


def analyze_npy_run(base_dir, run_name, layers_to_analyze):
    """Loads all .npy files for a given run and returns an activation dict."""
    print ("Analyzing NPY run: %s at %s" % (run_name, base_dir))
    run_dir = os.path.join(base_dir, run_name)
    activations = {}
    
    if not os.path.exists(run_dir):
        print (" Error: NPY Run directory not found at {}".format(run_dir))
        return activations

    for layer_name in layers_to_analyze:
        npy_path = os.path.join(run_dir, "{}.npy".format(layer_name))
        if os.path.exists(npy_path):
            print (" Loading and calculating {}...".format(layer_name))
            data = np.load(npy_path)
            activations[layer_name] = calculate_mean_activation(data)
        else:
            print (" Warning: Could not find {}. Skipping.".format(npy_path))
            
    return activations


def analyze_h5_run(h5_dir, ahat_filename, e_filename, layers_to_analyze):
    """Loads data directly from H5 files and returns an activation dict."""
    print ("Analyzing H5 run at %s" % h5_dir)
    activations = {}
    
    # --- Process Ahat ---
    ahat_h5_path = os.path.join(h5_dir, ahat_filename)
    try:
        with h5py.File(ahat_h5_path, 'r') as f_ahat:
            print ("  Processing %s..." % ahat_filename)
            for layer_name in [l for l in layers_to_analyze if 'Ahat' in l]:
                if layer_name in f_ahat:
                    print (" Extracting and calculating {}...".format(layer_name))
                    data = f_ahat[layer_name][:]  # Load data into memory
                    activations[layer_name] = calculate_mean_activation(data)
                else:
                    print (" Warning: Layer %s not found in %s" % (layer_name, ahat_filename))
    except IOError as e:
        print (" Error: Could not open H5 file at %s. (%s)" % (ahat_h5_path, e))
        # Continue to E file even if Ahat fails
        pass 

    # --- Process E ---
    e_h5_path = os.path.join(h5_dir, e_filename)
    try:
        with h5py.File(e_h5_path, 'r') as f_e:
            print ("  Processing %s..." % e_filename)
            for layer_name in [l for l in layers_to_analyze if 'E' in l]:
                if layer_name in f_e:
                    print (" Extracting and calculating {}...".format(layer_name))
                    data = f_e[layer_name][:]  # Load data into memory
                    activations[layer_name] = calculate_mean_activation(data)
                else:
                    print (" Warning: Layer %s not found in %s" % (layer_name, e_filename))
    except IOError as e:
        print (" Error: Could not open H5 file at %s. (%s)" % (e_h5_path, e))
        
    return activations


if __name__ == "__main__":
    print ("Starting quantitative analysis comparison (L4 vs L5)...")
    if not os.path.exists(ANALYSIS_SAVE_DIR):
        try:
            os.makedirs(ANALYSIS_SAVE_DIR)
        except OSError as e:
            print ("Warning: Could not create save directory (may exist): %s" % e)

    # 1. Analyze the 'Layer 5' run (from .npy files)
    activations_l5 = analyze_npy_run(L5_NPY_DIR, L5_RUN, LAYERS_TO_ANALYZE_L5)
    
    print ("---")
    
    # 2. Analyze the 'Layer 4' run (from .h5 files)
    activations_l4 = analyze_h5_run(L4_H5_DIR, L4_AHAT_FILE, L4_E_FILE, LAYERS_TO_ANALYZE_L4)

    # 3. Plot and save the comparison
    if not activations_l4 or not activations_l5:
        print ("\nError: Could not load data for one or both runs. Cannot create comparison plot.")
        exit()
        
    plot_filename = 'activation_comparison_L4_vs_L5_direct.png'
    plot_path = os.path.join(ANALYSIS_SAVE_DIR, plot_filename)
    
    plot_comparison_trends(activations_l4, activations_l5, plot_path)

    print ("\nAnalysis complete. Comparison plot saved in: %s" % ANALYSIS_SAVE_DIR)