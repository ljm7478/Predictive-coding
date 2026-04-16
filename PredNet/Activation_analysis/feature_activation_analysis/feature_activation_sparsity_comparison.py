"""
Sparsity Analysis: Calculates and plots the proportion of 'dead' (zero) units.
(Insight Text Box Removed Version)
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

def calculate_zero_proportion(data, threshold=1e-9):
    """
    Calculates the proportion of activations that are effectively zero.
    Values smaller than threshold are considered zero due to floating point precision.
    """
    total_elements = data.size
    # Count zeros (activations < threshold)
    zero_count = np.sum(np.abs(data) < threshold)
    return zero_count / total_elements

def plot_sparsity_comparison_clean(ahat_path, error_path, save_dir):
    print("Starting Sparsity Analysis (Clean Version)...")
    
    num_layers = 4
    ahat_sparsity = []
    error_sparsity = []
    layers = range(num_layers)
    
    # 1. Load Ahat Data
    if os.path.exists(ahat_path):
        with h5py.File(ahat_path, 'r') as f:
            for l in layers:
                key = 'Ahat{}'.format(l)
                if key in f.keys():
                    data = f[key][:]
                    prop = calculate_zero_proportion(data)
                    ahat_sparsity.append(prop)
                else:
                    ahat_sparsity.append(np.nan)
    else:
        print("Error: File not found - {}".format(ahat_path))
        return

    # 2. Load Error Data
    if os.path.exists(error_path):
        with h5py.File(error_path, 'r') as f:
            for l in layers:
                key = 'E{}'.format(l)
                if key in f.keys():
                    data = f[key][:]
                    prop = calculate_zero_proportion(data)
                    error_sparsity.append(prop)
                else:
                    error_sparsity.append(np.nan)
    else:
        print("Error: File not found - {}".format(error_path))
        return

    # 3. Plotting
    plt.figure(figsize=(10, 7))
    
    # Plot lines
    plt.plot(layers, ahat_sparsity, 'bo-', linewidth=2.5, markersize=10, label='Prediction (Ahat)')
    plt.plot(layers, error_sparsity, 'ro-', linewidth=2.5, markersize=10, label='Error (E)')
    
    # Styling
    plt.title('Feature Sparsity Comparison across Layers\n(Proportion of Inactive Units)', fontsize=14, fontweight='bold')
    plt.xlabel('Layer', fontsize=12)
    plt.ylabel('Proportion of Zeros (0.0 - 1.0)', fontsize=12)
    plt.xticks(layers, ['L0', 'L1', 'L2', 'L3'])
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='center right')
    
    # Add value annotations
    for i, val in enumerate(ahat_sparsity):
        plt.text(i, val + 0.03, '{:.1f}%'.format(val*100), 
                 ha='center', color='blue', fontweight='bold', fontsize=10)
        
    for i, val in enumerate(error_sparsity):
        plt.text(i, val - 0.05, '{:.1f}%'.format(val*100), 
                 ha='center', color='red', fontweight='bold', fontsize=10)

    # --- Insight Text Box Removed ---
    # The code block adding the text box has been removed here.
    # --------------------------------

    plt.tight_layout()
    
    # Save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'Sparsity_Comparison_Clean.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Clean sparsity plot saved to: {}".format(save_path))

if __name__ == "__main__":
    ahat_path = os.path.join(BASE_DIR, '{}_Ahat.h5'.format(SEG_NAME))
    error_path = os.path.join(BASE_DIR, '{}_E.h5'.format(SEG_NAME))
    
    plot_sparsity_comparison_clean(ahat_path, error_path, SAVE_DIR)