"""
Temporal Pattern Analysis: Prediction vs Error
- Quantify sparsity/continuity differences
- Evidence for: Prediction = sparse/event-driven, Error = continuous/dense

Key metrics:
1. Temporal Sparsity (% near-zero timepoints)
2. Non-zero Ratio (% active timepoints above threshold)
3. Coefficient of Variation (CV = std/mean)
4. Temporal Autocorrelation (lag-1 correlation)
5. Power Spectrum Analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats
from scipy import signal
import h5py
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

# PredNet feature paths (H5 format)
FEATURE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4'

SAVE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4/Results/Q1/temporal_analysis'

N_LAYERS = 4
SEGMENTS = ['seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_features(feature_dir, signal_type, layer, segments):
    """
    Load PredNet features from H5 files
    
    Parameters:
    -----------
    feature_dir : str
        Directory containing H5 files
    signal_type : str
        'Ahat' for Prediction, 'E' for Error
    layer : int
        Layer index (1-4)
    segments : list
        List of segment names ['seg0', 'seg1', ...]
    
    Returns:
    --------
    features : np.array
        (n_timepoints, n_features) concatenated across segments
    """
    all_features = []
    
    # Key format: Ahat0, Ahat1, ... or E0, E1, ...
    # layer 1 -> index 0, layer 2 -> index 1, etc.
    layer_idx = layer - 1
    key_name = f'{signal_type}{layer_idx}'
    
    for seg in segments:
        h5_path = os.path.join(feature_dir, f'{seg}_{signal_type}.h5')
        
        if os.path.exists(h5_path):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if key_name in f:
                        data = f[key_name][:]
                        
                        # Remove batch dimension: (1, time, H, W, C) -> (time, H, W, C)
                        if data.ndim == 5 and data.shape[0] == 1:
                            data = data[0]
                        
                        # Flatten spatial dimensions: (time, H, W, C) -> (time, features)
                        if data.ndim > 2:
                            n_time = data.shape[0]
                            data = data.reshape(n_time, -1)
                        
                        all_features.append(data)
                        print(f"    Loaded {seg} [{key_name}]: {data.shape}")
                    else:
                        print(f"    Key '{key_name}' not found")
                        
            except Exception as e:
                print(f"    Error loading {h5_path}: {e}")
        else:
            print(f"    Warning: {h5_path} not found")
    
    if len(all_features) > 0:
        return np.concatenate(all_features, axis=0)
    else:
        return None


def load_temporal_mean_efficient(feature_dir, signal_type, layer, segments):
    """
    Memory-efficient version: Compute mean per timepoint on-the-fly
    
    Returns:
    --------
    temporal_signal : np.array (n_total_timepoints,)
        Mean absolute activation per timepoint
    """
    all_temporal = []
    
    layer_idx = layer - 1
    key_name = f'{signal_type}{layer_idx}'
    
    for seg in segments:
        h5_path = os.path.join(feature_dir, f'{seg}_{signal_type}.h5')
        
        if os.path.exists(h5_path):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if key_name in f:
                        data = f[key_name][:]
                        
                        # Remove batch dimension
                        if data.ndim == 5 and data.shape[0] == 1:
                            data = data[0]
                        
                        # Compute mean |activation| per timepoint
                        n_time = data.shape[0]
                        temporal_mean = np.mean(np.abs(data.reshape(n_time, -1)), axis=1)
                        
                        all_temporal.append(temporal_mean)
                        print(f"    Loaded {seg} [{key_name}]: {n_time} timepoints")
                        
                        del data
                        
            except Exception as e:
                print(f"    Error loading {h5_path}: {e}")
        else:
            print(f"    Warning: {h5_path} not found")
    
    if len(all_temporal) > 0:
        return np.concatenate(all_temporal, axis=0)
    else:
        return None


def compute_sparsity_metrics(features, threshold_percentile=95):
    """
    Compute temporal sparsity metrics
    
    Parameters:
    -----------
    features : array
        Either (n_timepoints, n_features) or (n_timepoints,) for pre-computed temporal mean
    
    Returns:
    --------
    metrics : dict
        - sparsity: % of timepoints with values near zero
        - nonzero_ratio: % of timepoints above threshold
        - cv: coefficient of variation (std/mean)
        - autocorr_lag1: temporal autocorrelation at lag 1
    """
    # Handle both 1D (pre-computed temporal mean) and 2D (full features) input
    if features.ndim == 1:
        temporal_signal = np.abs(features)
    else:
        temporal_signal = np.mean(np.abs(features), axis=1)  # Mean activation per timepoint
    
    # 1. Sparsity: % of timepoints with values below median
    median_val = np.median(temporal_signal)
    sparsity = np.mean(temporal_signal < median_val * 0.5) * 100
    
    # 2. Non-zero ratio: % of timepoints above noise threshold
    noise_threshold = np.percentile(temporal_signal, 10)  # Bottom 10% as noise
    signal_threshold = np.percentile(temporal_signal, threshold_percentile)
    nonzero_ratio = np.mean(temporal_signal > signal_threshold) * 100
    
    # 3. Coefficient of Variation
    cv = np.std(temporal_signal) / (np.mean(temporal_signal) + 1e-10)
    
    # 4. Temporal Autocorrelation (lag-1)
    if len(temporal_signal) > 1:
        autocorr = np.corrcoef(temporal_signal[:-1], temporal_signal[1:])[0, 1]
    else:
        autocorr = np.nan
    
    # 5. Kurtosis (high = heavy tails = sparse spikes)
    kurtosis = stats.kurtosis(temporal_signal)
    
    # 6. Skewness (positive = right-skewed = rare large values)
    skewness = stats.skew(temporal_signal)
    
    return {
        'sparsity': sparsity,
        'nonzero_ratio': nonzero_ratio,
        'cv': cv,
        'autocorr_lag1': autocorr,
        'kurtosis': kurtosis,
        'skewness': skewness,
        'temporal_signal': temporal_signal
    }


def compute_power_spectrum(temporal_signal, fs=1.0):
    """
    Compute power spectrum of temporal signal
    """
    # Detrend
    detrended = signal.detrend(temporal_signal)
    
    # Compute power spectrum
    freqs, psd = signal.welch(detrended, fs=fs, nperseg=min(256, len(temporal_signal)//4))
    
    return freqs, psd


def analyze_activation_distribution(features):
    """
    Analyze the distribution of activation values
    """
    flat_features = features.flatten()
    
    # Remove exact zeros for log histogram
    nonzero = flat_features[flat_features != 0]
    
    return {
        'mean': np.mean(flat_features),
        'std': np.std(flat_features),
        'median': np.median(flat_features),
        'percent_zero': np.mean(flat_features == 0) * 100,
        'percent_near_zero': np.mean(np.abs(flat_features) < 0.01) * 100,
        'percent_active': np.mean(np.abs(flat_features) > np.percentile(np.abs(flat_features), 95)) * 100,
        'nonzero_values': nonzero
    }


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_temporal_comparison(pred_metrics, error_metrics, layer, save_path=None):
    """
    Plot temporal signal comparison for Prediction vs Error
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 1. Temporal signal overlay
    ax = axes[0, 0]
    t_pred = np.arange(len(pred_metrics['temporal_signal']))
    t_error = np.arange(len(error_metrics['temporal_signal']))
    
    ax.plot(t_pred[:500], pred_metrics['temporal_signal'][:500], 
            color='#3498DB', alpha=0.8, linewidth=1, label='Prediction')
    ax.plot(t_error[:500], error_metrics['temporal_signal'][:500], 
            color='#E74C3C', alpha=0.8, linewidth=1, label='Error')
    ax.set_xlabel('Time (frames)', fontsize=11)
    ax.set_ylabel('Mean |Activation|', fontsize=11)
    ax.set_title(f'Layer {layer}: Temporal Signal (first 500 frames)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram comparison
    ax = axes[0, 1]
    ax.hist(pred_metrics['temporal_signal'], bins=50, alpha=0.6, 
            color='#3498DB', label='Prediction', density=True)
    ax.hist(error_metrics['temporal_signal'], bins=50, alpha=0.6, 
            color='#E74C3C', label='Error', density=True)
    ax.set_xlabel('Mean |Activation|', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Activation Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Autocorrelation function
    ax = axes[1, 0]
    max_lag = 50
    
    pred_signal = pred_metrics['temporal_signal']
    error_signal = error_metrics['temporal_signal']
    
    pred_acf = [np.corrcoef(pred_signal[:-lag], pred_signal[lag:])[0,1] if lag > 0 else 1.0 
                for lag in range(max_lag)]
    error_acf = [np.corrcoef(error_signal[:-lag], error_signal[lag:])[0,1] if lag > 0 else 1.0 
                 for lag in range(max_lag)]
    
    ax.plot(range(max_lag), pred_acf, 'o-', color='#3498DB', label='Prediction', markersize=4)
    ax.plot(range(max_lag), error_acf, 's-', color='#E74C3C', label='Error', markersize=4)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (frames)', fontsize=11)
    ax.set_ylabel('Autocorrelation', fontsize=11)
    ax.set_title('Temporal Autocorrelation', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Metrics bar chart
    ax = axes[1, 1]
    metrics_names = ['CV', 'Autocorr\n(lag-1)', 'Kurtosis', 'Skewness']
    pred_vals = [pred_metrics['cv'], pred_metrics['autocorr_lag1'], 
                 pred_metrics['kurtosis']/10, pred_metrics['skewness']]  # Scale kurtosis
    error_vals = [error_metrics['cv'], error_metrics['autocorr_lag1'],
                  error_metrics['kurtosis']/10, error_metrics['skewness']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pred_vals, width, label='Prediction', color='#3498DB')
    bars2 = ax.bar(x + width/2, error_vals, width, label='Error', color='#E74C3C')
    
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Temporal Metrics Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    
    plt.suptitle(f'Temporal Pattern Analysis: Layer {layer}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_layer_comparison(all_pred_metrics, all_error_metrics, save_path=None):
    """
    Compare metrics across all layers
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    layers = list(range(1, N_LAYERS + 1))
    
    metrics_to_plot = [
        ('cv', 'Coefficient of Variation (CV)', 'Higher = More variable/sparse'),
        ('autocorr_lag1', 'Temporal Autocorrelation (lag-1)', 'Higher = More continuous'),
        ('kurtosis', 'Kurtosis', 'Higher = More peaked/sparse spikes'),
        ('skewness', 'Skewness', 'Higher = Right-skewed (rare large values)'),
        ('sparsity', 'Sparsity (%)', 'Higher = More sparse'),
        ('nonzero_ratio', 'High-activation Ratio (%)', 'Active timepoints above 95th percentile'),
    ]
    
    for idx, (metric_key, title, description) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        pred_vals = [all_pred_metrics[l][metric_key] for l in layers]
        error_vals = [all_error_metrics[l][metric_key] for l in layers]
        
        ax.plot(layers, pred_vals, 'o-', color='#3498DB', linewidth=2.5, 
                markersize=10, label='Prediction', markerfacecolor='white', markeredgewidth=2)
        ax.plot(layers, error_vals, 's-', color='#E74C3C', linewidth=2.5, 
                markersize=10, label='Error', markerfacecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('PredNet Layer', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}', fontsize=12, fontweight='bold')
        ax.set_xticks(layers)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add description as text
        ax.text(0.02, 0.98, description, transform=ax.transAxes, fontsize=8,
               verticalalignment='top', style='italic', alpha=0.7)
    
    plt.suptitle('Temporal Pattern Metrics Across Layers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_summary_figure(all_pred_metrics, all_error_metrics, save_path=None):
    """
    Create a summary figure highlighting the key finding:
    Prediction = Sparse/Event-driven, Error = Continuous/Dense
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    layers = list(range(1, N_LAYERS + 1))
    
    # Left: CV comparison (Sparsity indicator)
    ax = axes[0]
    pred_cv = [all_pred_metrics[l]['cv'] for l in layers]
    error_cv = [all_error_metrics[l]['cv'] for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pred_cv, width, label='Prediction', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, error_cv, width, label='Error', color='#E74C3C', edgecolor='black')
    
    ax.set_xlabel('PredNet Layer', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('Signal Variability\n(Higher CV = More Sparse/Bursty)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
    # Right: Autocorrelation comparison (Continuity indicator)
    ax = axes[1]
    pred_ac = [all_pred_metrics[l]['autocorr_lag1'] for l in layers]
    error_ac = [all_error_metrics[l]['autocorr_lag1'] for l in layers]
    
    bars1 = ax.bar(x - width/2, pred_ac, width, label='Prediction', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, error_ac, width, label='Error', color='#E74C3C', edgecolor='black')
    
    ax.set_xlabel('PredNet Layer', fontsize=12)
    ax.set_ylabel('Autocorrelation (lag-1)', fontsize=12)
    ax.set_title('Temporal Continuity\n(Higher = More Continuous Signal)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
    plt.suptitle('Prediction vs Error: Temporal Pattern Difference', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("="*70)
    print(" Temporal Pattern Analysis: Prediction vs Error")
    print(" Quantifying Sparsity vs Continuity")
    print("="*70)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Store metrics for all layers
    all_pred_metrics = {}
    all_error_metrics = {}
    
    for layer in range(1, N_LAYERS + 1):
        print(f"\n[Layer {layer}] Loading features (memory-efficient)...")
        
        # Load temporal mean (memory-efficient)
        pred_features = load_temporal_mean_efficient(FEATURE_DIR, 'Ahat', layer, SEGMENTS)
        error_features = load_temporal_mean_efficient(FEATURE_DIR, 'E', layer, SEGMENTS)
        
        if pred_features is None or error_features is None:
            print(f"    Skipping layer {layer} - features not found")
            continue
        
        print(f"    Prediction temporal: {pred_features.shape}")
        print(f"    Error temporal: {error_features.shape}")
        
        # Compute metrics
        print(f"    Computing temporal metrics...")
        pred_metrics = compute_sparsity_metrics(pred_features)
        error_metrics = compute_sparsity_metrics(error_features)
        
        all_pred_metrics[layer] = pred_metrics
        all_error_metrics[layer] = error_metrics
        
        # Print metrics
        print(f"\n    --- Prediction ---")
        print(f"    CV: {pred_metrics['cv']:.3f}")
        print(f"    Autocorr (lag-1): {pred_metrics['autocorr_lag1']:.3f}")
        print(f"    Kurtosis: {pred_metrics['kurtosis']:.3f}")
        print(f"    Skewness: {pred_metrics['skewness']:.3f}")
        
        print(f"\n    --- Error ---")
        print(f"    CV: {error_metrics['cv']:.3f}")
        print(f"    Autocorr (lag-1): {error_metrics['autocorr_lag1']:.3f}")
        print(f"    Kurtosis: {error_metrics['kurtosis']:.3f}")
        print(f"    Skewness: {error_metrics['skewness']:.3f}")
        
        # Plot per-layer comparison
        plot_temporal_comparison(
            pred_metrics, error_metrics, layer,
            save_path=os.path.join(SAVE_DIR, f'temporal_comparison_layer{layer}.png')
        )
    
    # Plot cross-layer comparison
    if len(all_pred_metrics) > 0:
        print("\n" + "="*70)
        print(" Creating summary plots...")
        
        plot_layer_comparison(
            all_pred_metrics, all_error_metrics,
            save_path=os.path.join(SAVE_DIR, 'temporal_metrics_all_layers.png')
        )
        
        plot_summary_figure(
            all_pred_metrics, all_error_metrics,
            save_path=os.path.join(SAVE_DIR, 'temporal_summary.png')
        )
        
        # Save numerical results
        results = {
            'prediction': {l: {k: v for k, v in m.items() if k != 'temporal_signal'} 
                          for l, m in all_pred_metrics.items()},
            'error': {l: {k: v for k, v in m.items() if k != 'temporal_signal'} 
                     for l, m in all_error_metrics.items()}
        }
        np.save(os.path.join(SAVE_DIR, 'temporal_metrics.npy'), results)
        
        # Print summary table
        print("\n" + "="*70)
        print(" SUMMARY TABLE")
        print("="*70)
        print(f"\n{'Layer':<8} {'Signal':<12} {'CV':<10} {'Autocorr':<12} {'Kurtosis':<12} {'Skewness':<10}")
        print("-"*64)
        for layer in sorted(all_pred_metrics.keys()):
            pm = all_pred_metrics[layer]
            em = all_error_metrics[layer]
            print(f"{layer:<8} {'Prediction':<12} {pm['cv']:<10.3f} {pm['autocorr_lag1']:<12.3f} {pm['kurtosis']:<12.3f} {pm['skewness']:<10.3f}")
            print(f"{'':<8} {'Error':<12} {em['cv']:<10.3f} {em['autocorr_lag1']:<12.3f} {em['kurtosis']:<12.3f} {em['skewness']:<10.3f}")
            print("-"*64)
    
    print("\n" + "="*70)
    print(" Done!")
    print("="*70)
    print(f"\nOutput files in: {SAVE_DIR}")
    print(f"  - temporal_comparison_layerX.png (per layer)")
    print(f"  - temporal_metrics_all_layers.png")
    print(f"  - temporal_summary.png")
    print(f"  - temporal_metrics.npy")