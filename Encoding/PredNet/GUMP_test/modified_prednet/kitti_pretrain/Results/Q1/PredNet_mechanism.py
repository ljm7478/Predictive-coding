"""
Model Mechanism Analysis: Why Prediction is Smooth, Error is Spiky

Hypothesis:
  - Prediction = ReLU(Conv(Recurrent_State)) ¡æ Temporally smooth
  - Error = ReLU(A - Ahat) ¡æ Frame-by-frame, no smoothing ¡æ Spiky

Quantitative Metrics:
  1. Autocorrelation decay: Smooth signal = slow decay
  2. Power spectrum: Smooth = low freq, Spiky = high freq
  3. Temporal derivative: Smooth = small changes, Spiky = large changes
  4. Zero-crossing rate: Smooth = few crossings, Spiky = many crossings
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import h5py
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4'

SAVE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4/Results/Q1/model_mechanism_analysis'

N_LAYERS = 4
SEGMENTS = ['seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_temporal_mean_efficient(feature_dir, signal_type, layer, segments):
    """Memory-efficient: compute mean|activation| per timepoint"""
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
                        
                        if data.ndim == 5 and data.shape[0] == 1:
                            data = data[0]
                        
                        n_time = data.shape[0]
                        temporal_mean = np.mean(np.abs(data.reshape(n_time, -1)), axis=1)
                        all_temporal.append(temporal_mean)
                        
                        del data
                        
            except Exception as e:
                print(f"    Error: {e}")
    
    if len(all_temporal) > 0:
        return np.concatenate(all_temporal, axis=0)
    return None


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_autocorrelation_curve(signal_1d, max_lag=100):
    """
    Compute autocorrelation at multiple lags
    Smooth signal ¡æ slow decay (high autocorr at long lags)
    Spiky signal ¡æ fast decay (low autocorr at long lags)
    """
    signal_centered = signal_1d - np.mean(signal_1d)
    autocorr = np.correlate(signal_centered, signal_centered, mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
    autocorr = autocorr / autocorr[0]  # Normalize
    
    return autocorr[:max_lag]


def compute_power_spectrum(signal_1d, fs=1.0):
    """
    Compute power spectrum
    Smooth signal ¡æ low frequency dominant
    Spiky signal ¡æ high frequency dominant
    """
    n = len(signal_1d)
    
    # Detrend and window
    signal_detrend = signal_1d - np.mean(signal_1d)
    window = np.hanning(n)
    signal_windowed = signal_detrend * window
    
    # FFT
    fft_vals = fft(signal_windowed)
    power = np.abs(fft_vals[:n//2])**2
    freqs = fftfreq(n, 1/fs)[:n//2]
    
    return freqs, power


def compute_temporal_derivative(signal_1d):
    """
    Compute frame-to-frame changes
    Smooth signal ¡æ small derivatives
    Spiky signal ¡æ large derivatives
    """
    derivative = np.diff(signal_1d)
    
    return {
        'mean_abs_derivative': np.mean(np.abs(derivative)),
        'std_derivative': np.std(derivative),
        'max_derivative': np.max(np.abs(derivative)),
        'derivative_signal': derivative
    }


def compute_zero_crossing_rate(signal_1d):
    """
    Count zero crossings (after mean subtraction)
    Smooth signal ¡æ few crossings
    Spiky signal ¡æ many crossings
    """
    signal_centered = signal_1d - np.mean(signal_1d)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(signal_centered))) > 0)
    zcr = zero_crossings / len(signal_1d)
    
    return zcr


def compute_spectral_centroid(freqs, power):
    """
    Weighted average frequency
    Smooth signal ¡æ low centroid
    Spiky signal ¡æ high centroid
    """
    centroid = np.sum(freqs * power) / (np.sum(power) + 1e-10)
    return centroid


def compute_low_high_freq_ratio(freqs, power, cutoff=0.1):
    """
    Ratio of low frequency to high frequency power
    Smooth signal ¡æ high ratio (more low freq)
    Spiky signal ¡æ low ratio (more high freq)
    """
    low_mask = freqs < cutoff
    high_mask = freqs >= cutoff
    
    low_power = np.sum(power[low_mask])
    high_power = np.sum(power[high_mask])
    
    ratio = low_power / (high_power + 1e-10)
    
    return ratio


def compute_all_metrics(signal_1d):
    """Compute all smoothness/spikiness metrics"""
    
    # 1. Autocorrelation
    autocorr = compute_autocorrelation_curve(signal_1d, max_lag=100)
    autocorr_lag10 = autocorr[10] if len(autocorr) > 10 else np.nan
    autocorr_lag50 = autocorr[50] if len(autocorr) > 50 else np.nan
    
    # 2. Power spectrum
    freqs, power = compute_power_spectrum(signal_1d)
    spectral_centroid = compute_spectral_centroid(freqs, power)
    low_high_ratio = compute_low_high_freq_ratio(freqs, power, cutoff=0.1)
    
    # 3. Temporal derivative
    deriv_metrics = compute_temporal_derivative(signal_1d)
    
    # 4. Zero crossing rate
    zcr = compute_zero_crossing_rate(signal_1d)
    
    # 5. Basic stats
    cv = np.std(signal_1d) / (np.mean(signal_1d) + 1e-10)
    
    return {
        'autocorr_lag10': autocorr_lag10,
        'autocorr_lag50': autocorr_lag50,
        'autocorr_curve': autocorr,
        'spectral_centroid': spectral_centroid,
        'low_high_freq_ratio': low_high_ratio,
        'freqs': freqs,
        'power': power,
        'mean_abs_derivative': deriv_metrics['mean_abs_derivative'],
        'std_derivative': deriv_metrics['std_derivative'],
        'zero_crossing_rate': zcr,
        'cv': cv
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_autocorrelation_comparison(pred_metrics, error_metrics, layer, save_path=None):
    """Plot autocorrelation decay curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = np.arange(len(pred_metrics['autocorr_curve']))
    
    ax.plot(lags, pred_metrics['autocorr_curve'], 
            color='#3498DB', linewidth=2.5, label='Prediction')
    ax.plot(lags, error_metrics['autocorr_curve'], 
            color='#E74C3C', linewidth=2.5, label='Error')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    
    # Annotate decay points
    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    
    ax.annotate(f"Pred lag-10: {pred_metrics['autocorr_lag10']:.3f}", 
                xy=(10, pred_metrics['autocorr_lag10']), fontsize=10,
                xytext=(15, pred_metrics['autocorr_lag10']+0.1),
                arrowprops=dict(arrowstyle='->', color='#3498DB'))
    ax.annotate(f"Error lag-10: {error_metrics['autocorr_lag10']:.3f}", 
                xy=(10, error_metrics['autocorr_lag10']), fontsize=10,
                xytext=(15, error_metrics['autocorr_lag10']-0.1),
                arrowprops=dict(arrowstyle='->', color='#E74C3C'))
    
    ax.set_xlabel('Lag (frames)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title(f'Layer {layer}: Autocorrelation Decay\n(Slow decay = Smooth, Fast decay = Spiky)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.2, 1.0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_power_spectrum_comparison(pred_metrics, error_metrics, layer, save_path=None):
    """Plot power spectrum comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Full spectrum (log scale)
    ax = axes[0]
    
    # Normalize power for comparison
    pred_power_norm = pred_metrics['power'] / np.max(pred_metrics['power'])
    error_power_norm = error_metrics['power'] / np.max(error_metrics['power'])
    
    ax.semilogy(pred_metrics['freqs'], pred_power_norm, 
                color='#3498DB', linewidth=1.5, alpha=0.8, label='Prediction')
    ax.semilogy(error_metrics['freqs'], error_power_norm, 
                color='#E74C3C', linewidth=1.5, alpha=0.8, label='Error')
    
    ax.axvline(x=0.1, color='black', linestyle='--', alpha=0.5, label='Low/High cutoff')
    
    ax.set_xlabel('Frequency', fontsize=12)
    ax.set_ylabel('Normalized Power (log)', fontsize=12)
    ax.set_title('Power Spectrum', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    
    # Right: Low vs High frequency comparison
    ax = axes[1]
    
    x = np.arange(2)
    width = 0.35
    
    pred_ratio = pred_metrics['low_high_freq_ratio']
    error_ratio = error_metrics['low_high_freq_ratio']
    
    # Normalize for visualization
    max_ratio = max(pred_ratio, error_ratio)
    
    bars1 = ax.bar(x - width/2, [pred_ratio, pred_metrics['spectral_centroid']*1000], 
                   width, label='Prediction', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, [error_ratio, error_metrics['spectral_centroid']*1000], 
                   width, label='Error', color='#E74C3C', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Low/High Freq Ratio\n(Higher=Smoother)', 
                       'Spectral Centroid (¡¿1000)\n(Lower=Smoother)'])
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Layer {layer}: Frequency Content Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_derivative_comparison(pred_metrics, error_metrics, layer, save_path=None):
    """Plot temporal derivative comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(3)
    width = 0.35
    
    pred_vals = [
        pred_metrics['mean_abs_derivative'],
        pred_metrics['std_derivative'],
        pred_metrics['zero_crossing_rate']
    ]
    error_vals = [
        error_metrics['mean_abs_derivative'],
        error_metrics['std_derivative'],
        error_metrics['zero_crossing_rate']
    ]
    
    # Normalize each metric by max for visualization
    for i in range(3):
        max_val = max(pred_vals[i], error_vals[i])
        if max_val > 0:
            pred_vals[i] /= max_val
            error_vals[i] /= max_val
    
    bars1 = ax.bar(x - width/2, pred_vals, width, label='Prediction', 
                   color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, error_vals, width, label='Error', 
                   color='#E74C3C', edgecolor='black')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean |Derivative|\n(Higher=Spikier)', 
                       'Std Derivative\n(Higher=Spikier)',
                       'Zero Crossing Rate\n(Higher=Spikier)'])
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title(f'Layer {layer}: Temporal Variability Metrics\n(All metrics: Higher = More Spiky/Detailed)', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_summary_all_layers(all_results, save_path=None):
    """Summary plot across all layers"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    layers = sorted(all_results.keys())
    x = np.arange(len(layers))
    width = 0.35
    
    # 1. Autocorrelation at lag-50
    ax = axes[0, 0]
    pred_vals = [all_results[l]['pred']['autocorr_lag50'] for l in layers]
    error_vals = [all_results[l]['error']['autocorr_lag50'] for l in layers]
    
    ax.bar(x - width/2, pred_vals, width, label='Prediction', color='#3498DB', edgecolor='black')
    ax.bar(x + width/2, error_vals, width, label='Error', color='#E74C3C', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_ylabel('Autocorrelation (lag-50)', fontsize=11)
    ax.set_title('Temporal Smoothness\n(Higher = Smoother)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Low/High frequency ratio
    ax = axes[0, 1]
    pred_vals = [all_results[l]['pred']['low_high_freq_ratio'] for l in layers]
    error_vals = [all_results[l]['error']['low_high_freq_ratio'] for l in layers]
    
    ax.bar(x - width/2, pred_vals, width, label='Prediction', color='#3498DB', edgecolor='black')
    ax.bar(x + width/2, error_vals, width, label='Error', color='#E74C3C', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_ylabel('Low/High Freq Ratio', fontsize=11)
    ax.set_title('Frequency Content\n(Higher = More Low-Freq Dominant)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Mean absolute derivative (normalized)
    ax = axes[1, 0]
    pred_vals = [all_results[l]['pred']['mean_abs_derivative'] for l in layers]
    error_vals = [all_results[l]['error']['mean_abs_derivative'] for l in layers]
    
    # Normalize by layer
    for i in range(len(layers)):
        max_val = max(pred_vals[i], error_vals[i])
        if max_val > 0:
            pred_vals[i] /= max_val
            error_vals[i] /= max_val
    
    ax.bar(x - width/2, pred_vals, width, label='Prediction', color='#3498DB', edgecolor='black')
    ax.bar(x + width/2, error_vals, width, label='Error', color='#E74C3C', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_ylabel('Normalized Mean |Derivative|', fontsize=11)
    ax.set_title('Frame-to-Frame Changes\n(Higher = More Variable)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Zero crossing rate
    ax = axes[1, 1]
    pred_vals = [all_results[l]['pred']['zero_crossing_rate'] for l in layers]
    error_vals = [all_results[l]['error']['zero_crossing_rate'] for l in layers]
    
    ax.bar(x - width/2, pred_vals, width, label='Prediction', color='#3498DB', edgecolor='black')
    ax.bar(x + width/2, error_vals, width, label='Error', color='#E74C3C', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.set_ylabel('Zero Crossing Rate', fontsize=11)
    ax.set_title('Signal Oscillation\n(Higher = More Oscillatory)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Model Mechanism Analysis: Prediction (Smooth) vs Error (Spiky)', 
                 fontsize=14, fontweight='bold')
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
    print(" Model Mechanism Analysis: Why Prediction=Smooth, Error=Spiky")
    print("="*70)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    all_results = {}
    
    for layer in range(1, N_LAYERS + 1):
        print(f"\n[Layer {layer}] Loading and analyzing...")
        
        pred_signal = load_temporal_mean_efficient(FEATURE_DIR, 'Ahat', layer, SEGMENTS)
        error_signal = load_temporal_mean_efficient(FEATURE_DIR, 'E', layer, SEGMENTS)
        
        if pred_signal is None or error_signal is None:
            print(f"    Skipping layer {layer}")
            continue
        
        print(f"    Prediction: {len(pred_signal)} timepoints")
        print(f"    Error: {len(error_signal)} timepoints")
        
        # Compute all metrics
        print("    Computing metrics...")
        pred_metrics = compute_all_metrics(pred_signal)
        error_metrics = compute_all_metrics(error_signal)
        
        # Print summary
        print(f"\n    === Layer {layer} Results ===")
        print(f"    Metric                  | Prediction | Error    | Winner")
        print("    " + "-"*60)
        print(f"    Autocorr (lag-50)       | {pred_metrics['autocorr_lag50']:.4f}     | {error_metrics['autocorr_lag50']:.4f}   | {'Pred' if pred_metrics['autocorr_lag50'] > error_metrics['autocorr_lag50'] else 'Error'}")
        print(f"    Low/High Freq Ratio     | {pred_metrics['low_high_freq_ratio']:.4f}     | {error_metrics['low_high_freq_ratio']:.4f}   | {'Pred' if pred_metrics['low_high_freq_ratio'] > error_metrics['low_high_freq_ratio'] else 'Error'}")
        print(f"    Mean |Derivative|       | {pred_metrics['mean_abs_derivative']:.6f}   | {error_metrics['mean_abs_derivative']:.6f} | {'Error' if error_metrics['mean_abs_derivative'] > pred_metrics['mean_abs_derivative'] else 'Pred'}")
        print(f"    Zero Crossing Rate      | {pred_metrics['zero_crossing_rate']:.4f}     | {error_metrics['zero_crossing_rate']:.4f}   | {'Error' if error_metrics['zero_crossing_rate'] > pred_metrics['zero_crossing_rate'] else 'Pred'}")
        
        # Store results
        all_results[layer] = {
            'pred': pred_metrics,
            'error': error_metrics
        }
        
        # Per-layer plots
        plot_autocorrelation_comparison(
            pred_metrics, error_metrics, layer,
            save_path=os.path.join(SAVE_DIR, f'autocorr_layer{layer}.png')
        )
        
        plot_power_spectrum_comparison(
            pred_metrics, error_metrics, layer,
            save_path=os.path.join(SAVE_DIR, f'power_spectrum_layer{layer}.png')
        )
        
        plot_derivative_comparison(
            pred_metrics, error_metrics, layer,
            save_path=os.path.join(SAVE_DIR, f'derivative_layer{layer}.png')
        )
    
    # Summary plot
    if len(all_results) > 0:
        print("\n" + "="*70)
        print(" Creating summary plots...")
        
        plot_summary_all_layers(
            all_results,
            save_path=os.path.join(SAVE_DIR, 'model_mechanism_summary.png')
        )
        
        # Save results
        np.save(os.path.join(SAVE_DIR, 'model_mechanism_results.npy'), all_results)
        
        # Final summary table
        print("\n" + "="*70)
        print(" FINAL SUMMARY: Model Mechanism Verification")
        print("="*70)
        print("\n Hypothesis: Prediction = Smooth (recurrent), Error = Spiky (frame-by-frame)")
        print("\n Evidence:")
        print(" +-------+--------------------+--------------------+---------+")
        print(" | Layer | Autocorr(50) P vs E| LowFreq Ratio P vs E| Support |")
        print(" +-------+--------------------+--------------------+---------+")
        
        for l in sorted(all_results.keys()):
            p_ac = all_results[l]['pred']['autocorr_lag50']
            e_ac = all_results[l]['error']['autocorr_lag50']
            p_lf = all_results[l]['pred']['low_high_freq_ratio']
            e_lf = all_results[l]['error']['low_high_freq_ratio']
            
            ac_support = "O" if p_ac > e_ac else "X"
            lf_support = "O" if p_lf > e_lf else "X"
            overall = "OO" if (p_ac > e_ac and p_lf > e_lf) else "??"
            
            print(f" | L{l}    | {p_ac:.3f} vs {e_ac:.3f} {ac_support}   | {p_lf:.2f} vs {e_lf:.2f} {lf_support}      |   {overall}   |")
        
        print(" +-------+--------------------+--------------------+---------+")
        print("\n O = Prediction smoother than Error (hypothesis supported)")
        print(" X = Error smoother than Prediction (hypothesis NOT supported)")
        
    print("\n" + "="*70)
    print(" Analysis complete!")
    print("="*70)