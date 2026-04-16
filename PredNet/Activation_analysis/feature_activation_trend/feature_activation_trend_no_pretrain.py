"""
Performs quantitative analysis on PredNet features.

This script calculates the mean activation of Prediction (Ahat) and Error (E)
signals for each layer and visualizes the trend to show convergence towards zero
in higher layers, which is a key indicator of a well-trained model.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend, suitable for servers
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/07_CamCAN/result/'
RUN_TO_ANALYZE = 'run1'
ANALYSIS_SAVE_DIR = os.path.join(BASE_RESULTS_DIR, 'analysis_plots')

def calculate_mean_activation(data):
    """Calculates the mean absolute activation value of the given data."""
    # Data shape is assumed to be (B, T, H, W, C)
    # The mean is calculated across time, height, width, and channels.
    return np.mean(np.abs(data))


def plot_activation_trends(activation_dict, save_path):
    """Plots and saves the trend of activation values across layers."""
    num_layers = 4
    ahat_activations = [activation_dict.get('Ahat{}'.format(l), 0) for l in range(num_layers)]
    e_activations = [activation_dict.get('E{}'.format(l), 0) for l in range(num_layers)]
    layers = ['L{}'.format(l) for l in range(num_layers)]

    plt.figure(figsize=(10, 6))
    plt.plot(layers, ahat_activations, 'bo-', label='Prediction (Ahat) Activation')
    plt.plot(layers, e_activations, 'ro-', label='Error (E) Activation')
    
    plt.title('Mean Feature Activation across Layers for {}'.format(RUN_TO_ANALYZE))
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Activation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    # Using a log scale is helpful as values converge to zero rapidly.
    plt.yscale('log')
    
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Activation trend plot saved to: {}".format(save_path))


if __name__ == "__main__":
    print("Starting quantitative analysis for run: {}".format(RUN_TO_ANALYZE))
    if not os.path.exists(ANALYSIS_SAVE_DIR):
        os.makedirs(ANALYSIS_SAVE_DIR)

    run_dir = os.path.join(BASE_RESULTS_DIR, RUN_TO_ANALYZE)
    if not os.path.exists(run_dir):
        print("Error: Run directory not found at {}".format(run_dir))
        exit()

    layers_to_analyze = ['Ahat0', 'Ahat1', 'Ahat2', 'Ahat3', 'E0', 'E1', 'E2', 'E3']
    activations = {}

    for layer_name in layers_to_analyze:
        npy_path = os.path.join(run_dir, "{}.npy".format(layer_name))
        if os.path.exists(npy_path):
            print("  Calculating activation for {}...".format(layer_name))
            data = np.load(npy_path)
            activations[layer_name] = calculate_mean_activation(data)
        else:
            print("  Warning: Could not find {}. Skipping.".format(npy_path))
    
    # Plot and save the results
    plot_path = os.path.join(ANALYSIS_SAVE_DIR, 'activation_trends_{}.png'.format(RUN_TO_ANALYZE))
    plot_activation_trends(activations, plot_path)

    print("\nAnalysis complete. Plot saved in: {}".format(ANALYSIS_SAVE_DIR))
