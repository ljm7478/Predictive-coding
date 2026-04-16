"""
Simplified CV Analysis: Mean Activation + CV + Summary
For presentation: Show this first, then Time-series for visual validation
"""
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_RESULTS_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/'
ANALYSIS_SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_activation_analysis/test_GUMP/pt_kitti_ft_Titanic_Lall_nt150/original/layer4/'
RUN_TO_ANALYZE = 'layer4'
SEG_TO_ANALYZE = 'seg0'


def analyze_temporal_dynamics(data, layer_name):
    """Compute temporal dynamics metrics."""
    B, T, H, W, C = data.shape
    data_flat = data.reshape(B, T, -1)
    
    mean_act = np.mean(np.abs(data))
    temporal_var_per_batch = np.var(data_flat, axis=1)
    temporal_var = np.mean(temporal_var_per_batch)
    temporal_std = np.sqrt(temporal_var)
    mean_abs = np.mean(np.abs(data_flat), axis=(0, 1)).mean()
    cv = temporal_std / (mean_abs + 1e-10)
    
    return {
        'mean_activation': mean_act,
        'temporal_std': temporal_std,
        'cv': cv,
        'shape': data.shape
    }


def plot_simplified_cv_analysis(ahat_metrics, error_metrics, save_dir):
    """
    Simplified plot: Mean Activation + CV + Comprehensive Summary
    """
    num_layers = len(ahat_metrics)
    layers = ['L{}'.format(l) for l in range(num_layers)]
    
    # Extract data
    ahat_mean = [ahat_metrics[l]['mean_activation'] for l in range(num_layers)]
    error_mean = [error_metrics[l]['mean_activation'] for l in range(num_layers)]
    ahat_cv = [ahat_metrics[l]['cv'] for l in range(num_layers)]
    error_cv = [error_metrics[l]['cv'] for l in range(num_layers)]
    
    # Create figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # --- 1. Mean Absolute Activation ---
    ax = axes[0]
    ax.plot(layers, ahat_mean, 'bo-', label='Prediction (Ahat)', linewidth=2.5, markersize=10)
    ax.plot(layers, error_mean, 'ro-', label='Error (E)', linewidth=2.5, markersize=10)
    ax.set_yscale('log')
    ax.set_title('Mean Absolute Activation', fontsize=13, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('Activation (log scale)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    
    # --- 2. Coefficient of Variation (CV) ---
    ax = axes[1]
    ax.plot(layers, ahat_cv, 'bo-', label='Prediction (Ahat)', linewidth=2.5, markersize=10)
    ax.plot(layers, error_cv, 'ro-', label='Error (E)', linewidth=2.5, markersize=10)
    ax.set_title('Coefficient of Variation (CV)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Layer', fontsize=11)
    ax.set_ylabel('CV (std / mean)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#ffffee')  # Highlight
    
    # --- 3. Comprehensive Summary ---
    ax = axes[2]
    ax.axis('off')
    
    # Build summary text
    summary_lines = [
        "SUMMARY",
        "=" * 50,
        "",
        "CV = std / mean  (relative variability)",
        "",
        "-" * 50,
        "",
        "ERROR (E):",
        "{:<6} {:>12} {:>12}".format("Layer", "Mean Act", "CV"),
        "-" * 35,
    ]
    
    for l in range(num_layers):
        change = ""
        if l > 0:
            mean_chg = ((error_mean[l] - error_mean[l-1]) / error_mean[l-1]) * 100
            cv_chg = ((error_cv[l] - error_cv[l-1]) / error_cv[l-1]) * 100
        summary_lines.append("L{}     {:>12.6f} {:>12.2f}".format(l, error_mean[l], error_cv[l]))
    
    summary_lines.extend([
        "",
        "PREDICTION (Ahat):",
        "{:<6} {:>12} {:>12}".format("Layer", "Mean Act", "CV"),
        "-" * 35,
    ])
    
    for l in range(num_layers):
        summary_lines.append("L{}     {:>12.6f} {:>12.2f}".format(l, ahat_mean[l], ahat_cv[l]))
    
    # Trend analysis
    summary_lines.extend([
        "",
        "-" * 50,
        "TREND ANALYSIS:",
        "",
        "Error:      Mean {:.1f}x decrease, CV {:.1f}x increase".format(
            error_mean[0] / error_mean[-1], error_cv[-1] / error_cv[0]),
        "Prediction: Mean {:.1f}x decrease, CV {:.1f}x increase".format(
            ahat_mean[0] / ahat_mean[-1], ahat_cv[-1] / ahat_cv[0]),
        "",
        "-" * 50,
        "KEY INSIGHT:",
        "",
        "Both Mean activation DECREASES",
        "but CV (relative variability) INCREASES!",
        "",
        "-> Encoding captures temporal fluctuation",
        "-> Higher CV = more distinctive pattern",
        "-> Explains why beta increases with layer",
    ])
    
    summary_text = "\n".join(summary_lines)
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for suptitle
    plt.suptitle('Temporal Dynamics Analysis: Original PredNet ({})'.format(SEG_TO_ANALYZE), 
                 fontsize=14, fontweight='bold', y=0.98)
    
    save_path = os.path.join(save_dir, 'CV_analysis_simplified_{}.png'.format(SEG_TO_ANALYZE))
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("Simplified CV plot saved to: {}".format(save_path))
    
    return save_path


def print_summary(ahat_metrics, error_metrics):
    """Print concise summary."""
    num_layers = len(ahat_metrics)
    
    print("\n" + "=" * 60)
    print("CV ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("\n{:<8} {:>12} {:>10} {:>12} {:>10}".format(
        "Layer", "E_Mean", "E_CV", "Ahat_Mean", "Ahat_CV"))
    print("-" * 55)
    
    for l in range(num_layers):
        print("L{:<7} {:>12.6f} {:>10.2f} {:>12.6f} {:>10.2f}".format(
            l,
            error_metrics[l]['mean_activation'],
            error_metrics[l]['cv'],
            ahat_metrics[l]['mean_activation'],
            ahat_metrics[l]['cv']))
    
    print("\n" + "=" * 60)
    print("INTERPRETATION:")
    print("  - Mean decreases with layer (prediction succeeds)")
    print("  - CV increases with layer (relative variability grows)")
    print("  - Encoding captures CV pattern -> beta increases")
    print("=" * 60)


if __name__ == "__main__":
    print("Starting simplified CV analysis...")
    
    if not os.path.exists(ANALYSIS_SAVE_DIR):
        os.makedirs(ANALYSIS_SAVE_DIR)
    
    run_dir = os.path.join(BASE_RESULTS_DIR, RUN_TO_ANALYZE)
    ahat_h5_path = os.path.join(run_dir, "{}_Ahat.h5".format(SEG_TO_ANALYZE))
    error_h5_path = os.path.join(run_dir, "{}_E.h5".format(SEG_TO_ANALYZE))
    
    print("Loading from:")
    print("  Ahat: {}".format(ahat_h5_path))
    print("  Error: {}".format(error_h5_path))
    
    if not os.path.exists(ahat_h5_path) or not os.path.exists(error_h5_path):
        print("Error: Files not found!")
        exit()
    
    # Analyze all layers
    num_layers = 4
    ahat_metrics = {}
    error_metrics = {}
    
    print("\n--- Analyzing features ---")
    with h5py.File(ahat_h5_path, 'r') as f_ahat, h5py.File(error_h5_path, 'r') as f_err:
        for l in range(num_layers):
            ahat_key = "Ahat{}".format(l)
            error_key = "E{}".format(l)
            
            if ahat_key in f_ahat.keys():
                data = f_ahat[ahat_key][:]
                ahat_metrics[l] = analyze_temporal_dynamics(data, ahat_key)
                print("  {}: Mean={:.6f}, CV={:.2f}".format(
                    ahat_key, ahat_metrics[l]['mean_activation'], ahat_metrics[l]['cv']))
            
            if error_key in f_err.keys():
                data = f_err[error_key][:]
                error_metrics[l] = analyze_temporal_dynamics(data, error_key)
                print("  {}: Mean={:.6f}, CV={:.2f}".format(
                    error_key, error_metrics[l]['mean_activation'], error_metrics[l]['cv']))
    
    # Print summary
    print_summary(ahat_metrics, error_metrics)
    
    # Plot
    if ahat_metrics and error_metrics:
        plot_path = plot_simplified_cv_analysis(ahat_metrics, error_metrics, ANALYSIS_SAVE_DIR)
    
    print("\n[Done] Analysis complete!")
    print("\nNEXT STEP: Run time-series analysis to visually validate the CV findings!")