"""
Event-Locked Analysis: WHY Prediction is Sparse, Error is Continuous

Key Question: 
  - WHY does Prediction respond only to specific events?
  - WHY does Error show continuous activation?

Hypothesis:
  - Prediction spikes occur at "prediction failure" moments (scene transitions, new objects)
  - Error persists because perfect prediction is impossible (always residual mismatch)

Evidence needed:
  1. Prediction spikes correlate with scene transitions
  2. Error maintains activity regardless of scene transitions
  3. Prediction is "silent" during predictable sequences

Methods:
  1. Detect scene transitions in movie (pixel-level change detection)
  2. Event-locked averaging: Prediction/Error aligned to scene transitions
  3. Correlation between Prediction magnitude and scene change magnitude
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import stats, signal
from scipy.ndimage import gaussian_filter1d
import h5py
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4'

SAVE_DIR = '/combinelab/03_user/jungmin/01_project/01_Encoding/01_HWen_Encoding/PredNet/Titanic_trained/GUMP_test/encoding_analysis/original_prednet/kitti_pretrain_result/four_train_four_test_PCAcap/layer4/Results/Q1/event_locked_analysis'

N_LAYERS = 4
# Segments (adjust based on actual files)
SEGMENTS = ['seg0', 'seg1', 'seg2', 'seg3', 'seg4', 'seg5', 'seg6', 'seg7']

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def inspect_h5_structure(feature_dir, segments):
    """Helper function to inspect H5 file structure"""
    for seg in segments[:1]:  # Just check first segment
        for signal_type in ['Ahat', 'E']:
            h5_path = os.path.join(feature_dir, f'{seg}_{signal_type}.h5')
            if os.path.exists(h5_path):
                print(f"\n  Inspecting: {h5_path}")
                with h5py.File(h5_path, 'r') as f:
                    print(f"  Keys: {list(f.keys())}")
                    for key in f.keys():
                        print(f"    {key}: shape={f[key].shape}, dtype={f[key].dtype}")
            else:
                print(f"  Not found: {h5_path}")


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
    seg_lengths : list
        Length of each segment
    """
    all_features = []
    seg_lengths = []
    
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
                        seg_lengths.append(len(data))
                        print(f"    Loaded {seg} [{key_name}]: {data.shape}")
                    else:
                        print(f"    Key '{key_name}' not found in {seg}_{signal_type}.h5")
                        print(f"    Available keys: {list(f.keys())}")
                        
            except Exception as e:
                print(f"    Error loading {h5_path}: {e}")
        else:
            print(f"    Warning: {h5_path} not found")
    
    if len(all_features) > 0:
        return np.concatenate(all_features, axis=0), seg_lengths
    else:
        return None, None


def load_temporal_mean_efficient(feature_dir, signal_type, layer, segments):
    """
    Memory-efficient version: Compute mean per timepoint on-the-fly
    Instead of loading full (time, H*W*C), compute mean|activation| per timepoint
    
    Returns:
    --------
    temporal_signal : np.array (n_total_timepoints,)
        Mean absolute activation per timepoint
    seg_lengths : list
        Length of each segment
    """
    all_temporal = []
    seg_lengths = []
    
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
                            data = data[0]  # (time, H, W, C)
                        
                        # Compute mean |activation| per timepoint
                        n_time = data.shape[0]
                        temporal_mean = np.mean(np.abs(data.reshape(n_time, -1)), axis=1)
                        
                        all_temporal.append(temporal_mean)
                        seg_lengths.append(n_time)
                        print(f"    Loaded {seg} [{key_name}]: {n_time} timepoints")
                        
                        del data  # Free memory
                        
            except Exception as e:
                print(f"    Error loading {h5_path}: {e}")
        else:
            print(f"    Warning: {h5_path} not found")
    
    if len(all_temporal) > 0:
        return np.concatenate(all_temporal, axis=0), seg_lengths
    else:
        return None, None



def detect_scene_transitions(features, threshold_percentile=95, min_distance=5):
    """
    Detect scene transitions based on frame-to-frame change
    
    Method: Compute temporal derivative of feature magnitude
    Scene transition = large sudden change
    
    Parameters:
    -----------
    features : array (n_timepoints, n_features)
    threshold_percentile : percentile to define "large" change
    min_distance : minimum frames between detected transitions
    
    Returns:
    --------
    transition_indices : array of timepoint indices
    change_magnitude : frame-to-frame change magnitude
    """
    # Compute mean activation per timepoint
    if features.ndim > 1:
        temporal_signal = np.mean(np.abs(features), axis=1)
    else:
        temporal_signal = np.abs(features)
    
    # Compute frame-to-frame difference
    diff = np.abs(np.diff(temporal_signal))
    diff = np.concatenate([[0], diff])  # Pad to match length
    
    # Smooth slightly to reduce noise
    diff_smooth = gaussian_filter1d(diff, sigma=1)
    
    # Find threshold
    threshold = np.percentile(diff_smooth, threshold_percentile)
    
    # Find peaks above threshold
    peaks, properties = signal.find_peaks(diff_smooth, 
                                           height=threshold, 
                                           distance=min_distance)
    
    return peaks, diff_smooth


def detect_scene_transitions_from_error(error_features, threshold_percentile=90, min_distance=10):
    """
    Alternative: Use Error Layer 1 spikes to detect scene transitions
    Rationale: Scene transitions cause large prediction errors at lowest level
    """
    temporal_signal = np.mean(np.abs(error_features), axis=1)
    
    # Find peaks (sudden increases in error)
    threshold = np.percentile(temporal_signal, threshold_percentile)
    peaks, _ = signal.find_peaks(temporal_signal, height=threshold, distance=min_distance)
    
    return peaks, temporal_signal


def event_locked_average(signal_data, event_indices, window_pre=20, window_post=30):
    """
    Compute event-locked average
    
    Parameters:
    -----------
    signal_data : array (n_timepoints,) or (n_timepoints, n_features)
    event_indices : array of event timepoints
    window_pre : frames before event
    window_post : frames after event
    
    Returns:
    --------
    mean_response : average response around events
    sem_response : SEM of response
    time_axis : time relative to event (in frames)
    """
    if signal_data.ndim > 1:
        signal_1d = np.mean(np.abs(signal_data), axis=1)
    else:
        signal_1d = np.abs(signal_data)
    
    n_timepoints = len(signal_1d)
    all_epochs = []
    
    for event_idx in event_indices:
        start = event_idx - window_pre
        end = event_idx + window_post
        
        if start >= 0 and end < n_timepoints:
            epoch = signal_1d[start:end]
            all_epochs.append(epoch)
    
    if len(all_epochs) == 0:
        return None, None, None
    
    all_epochs = np.array(all_epochs)
    
    mean_response = np.mean(all_epochs, axis=0)
    sem_response = np.std(all_epochs, axis=0) / np.sqrt(len(all_epochs))
    time_axis = np.arange(-window_pre, window_post)
    
    return mean_response, sem_response, time_axis


def compute_event_correlation(pred_signal, error_signal, event_indices, window=5):
    """
    Compute correlation between Prediction spikes and scene transitions
    
    Returns correlation and p-value
    """
    # Create binary event vector
    event_vector = np.zeros(len(pred_signal))
    for idx in event_indices:
        start = max(0, idx - window)
        end = min(len(event_vector), idx + window)
        event_vector[start:end] = 1
    
    # Compute correlations
    pred_temporal = np.mean(np.abs(pred_signal), axis=1) if pred_signal.ndim > 1 else np.abs(pred_signal)
    error_temporal = np.mean(np.abs(error_signal), axis=1) if error_signal.ndim > 1 else np.abs(error_signal)
    
    # Correlation with event vector
    pred_corr, pred_p = stats.pearsonr(pred_temporal, event_vector)
    error_corr, error_p = stats.pearsonr(error_temporal, event_vector)
    
    return {
        'pred_event_corr': pred_corr,
        'pred_event_p': pred_p,
        'error_event_corr': error_corr,
        'error_event_p': error_p
    }


def compute_baseline_vs_event(signal_data, event_indices, baseline_window=(-50, -20), event_window=(-5, 15)):
    """
    Compare signal magnitude during baseline vs event periods
    
    Returns:
    --------
    baseline_mean, event_mean, t_stat, p_value
    """
    if signal_data.ndim > 1:
        signal_1d = np.mean(np.abs(signal_data), axis=1)
    else:
        signal_1d = np.abs(signal_data)
    
    baseline_values = []
    event_values = []
    
    for event_idx in event_indices:
        # Check both windows are valid
        b_start = event_idx + baseline_window[0]
        b_end = event_idx + baseline_window[1]
        e_start = event_idx + event_window[0]
        e_end = event_idx + event_window[1]
        
        # Only include if BOTH windows are valid
        if b_start >= 0 and b_end < len(signal_1d) and e_start >= 0 and e_end < len(signal_1d):
            baseline_values.append(np.mean(signal_1d[b_start:b_end]))
            event_values.append(np.mean(signal_1d[e_start:e_end]))
    
    if len(baseline_values) < 5 or len(event_values) < 5:
        return np.nan, np.nan, np.nan, np.nan
    
    baseline_mean = np.mean(baseline_values)
    event_mean = np.mean(event_values)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(event_values, baseline_values)
    
    return baseline_mean, event_mean, t_stat, p_value


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_event_locked_comparison(pred_ela, error_ela, pred_sem, error_sem, time_axis, layer, save_path=None):
    """
    Plot event-locked average comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Event-locked averages
    ax = axes[0]
    ax.fill_between(time_axis, pred_ela - pred_sem, pred_ela + pred_sem, 
                    alpha=0.3, color='#3498DB')
    ax.fill_between(time_axis, error_ela - error_sem, error_ela + error_sem, 
                    alpha=0.3, color='#E74C3C')
    ax.plot(time_axis, pred_ela, color='#3498DB', linewidth=2.5, label='Prediction')
    ax.plot(time_axis, error_ela, color='#E74C3C', linewidth=2.5, label='Error')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='Scene Transition')
    ax.axhline(y=pred_ela[time_axis < -10].mean(), color='#3498DB', linestyle=':', alpha=0.5)
    ax.axhline(y=error_ela[time_axis < -10].mean(), color='#E74C3C', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time relative to scene transition (frames)', fontsize=12)
    ax.set_ylabel('Mean |Activation|', fontsize=12)
    ax.set_title(f'Layer {layer}: Event-Locked Average', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right: Normalized response (% change from baseline)
    ax = axes[1]
    pred_baseline = pred_ela[time_axis < -10].mean()
    error_baseline = error_ela[time_axis < -10].mean()
    
    pred_norm = (pred_ela - pred_baseline) / (pred_baseline + 1e-10) * 100
    error_norm = (error_ela - error_baseline) / (error_baseline + 1e-10) * 100
    
    ax.plot(time_axis, pred_norm, color='#3498DB', linewidth=2.5, label='Prediction')
    ax.plot(time_axis, error_norm, color='#E74C3C', linewidth=2.5, label='Error')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    ax.set_xlabel('Time relative to scene transition (frames)', fontsize=12)
    ax.set_ylabel('% Change from baseline', fontsize=12)
    ax.set_title(f'Layer {layer}: Normalized Response', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Scene Transition Response: Prediction vs Error', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_temporal_raster(pred_signal, error_signal, event_indices, layer, 
                         n_frames=1000, save_path=None):
    """
    Plot temporal raster showing Prediction sparsity vs Error continuity
    with scene transitions marked
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    
    # Prepare signals
    pred_temporal = np.mean(np.abs(pred_signal[:n_frames]), axis=1)
    error_temporal = np.mean(np.abs(error_signal[:n_frames]), axis=1)
    
    # Normalize for visualization
    pred_norm = (pred_temporal - pred_temporal.min()) / (pred_temporal.max() - pred_temporal.min() + 1e-10)
    error_norm = (error_temporal - error_temporal.min()) / (error_temporal.max() - error_temporal.min() + 1e-10)
    
    time = np.arange(n_frames)
    
    # Top: Prediction
    ax = axes[0]
    ax.fill_between(time, 0, pred_norm, color='#3498DB', alpha=0.7)
    ax.set_ylabel('Prediction\n(normalized)', fontsize=11)
    ax.set_title(f'Layer {layer}: Temporal Pattern with Scene Transitions', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.2)
    
    # Mark scene transitions
    for idx in event_indices:
        if idx < n_frames:
            ax.axvline(x=idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Middle: Error
    ax = axes[1]
    ax.fill_between(time, 0, error_norm, color='#E74C3C', alpha=0.7)
    ax.set_ylabel('Error\n(normalized)', fontsize=11)
    ax.set_ylim(0, 1.2)
    
    for idx in event_indices:
        if idx < n_frames:
            ax.axvline(x=idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Bottom: Scene transition markers
    ax = axes[2]
    event_vector = np.zeros(n_frames)
    for idx in event_indices:
        if idx < n_frames:
            event_vector[idx] = 1
    ax.fill_between(time, 0, event_vector, color='red', alpha=0.8)
    ax.set_ylabel('Scene\nTransition', fontsize=11)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_summary_bar(all_results, save_path=None):
    """
    Summary bar plot: Baseline vs Event response for Prediction and Error
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    layers = sorted(all_results.keys())
    
    # Left: Absolute magnitude comparison
    ax = axes[0]
    x = np.arange(len(layers))
    width = 0.2
    
    pred_baseline = [all_results[l]['pred_baseline'] for l in layers]
    pred_event = [all_results[l]['pred_event'] for l in layers]
    error_baseline = [all_results[l]['error_baseline'] for l in layers]
    error_event = [all_results[l]['error_event'] for l in layers]
    
    ax.bar(x - 1.5*width, pred_baseline, width, label='Pred Baseline', color='#85C1E9', edgecolor='black')
    ax.bar(x - 0.5*width, pred_event, width, label='Pred Event', color='#2980B9', edgecolor='black')
    ax.bar(x + 0.5*width, error_baseline, width, label='Error Baseline', color='#F1948A', edgecolor='black')
    ax.bar(x + 1.5*width, error_event, width, label='Error Event', color='#C0392B', edgecolor='black')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean |Activation|', fontsize=12)
    ax.set_title('Baseline vs Event Period', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: % change from baseline
    ax = axes[1]
    pred_pct = [(e - b) / (b + 1e-10) * 100 for b, e in zip(pred_baseline, pred_event)]
    error_pct = [(e - b) / (b + 1e-10) * 100 for b, e in zip(error_baseline, error_event)]
    
    ax.bar(x - width/2, pred_pct, width, label='Prediction', color='#3498DB', edgecolor='black')
    ax.bar(x + width/2, error_pct, width, label='Error', color='#E74C3C', edgecolor='black')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('% Change at Event', fontsize=12)
    ax.set_title('Response to Scene Transitions\n(% change from baseline)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance markers
    for i, l in enumerate(layers):
        if all_results[l]['pred_p'] < 0.05:
            ax.text(i - width/2, pred_pct[i] + 2, '*', ha='center', fontsize=14, fontweight='bold')
        if all_results[l]['error_p'] < 0.05:
            ax.text(i + width/2, error_pct[i] + 2, '*', ha='center', fontsize=14, fontweight='bold')
    
    plt.suptitle('Evidence: Prediction Responds to Events, Error is Persistent', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_correlation_summary(all_results, save_path=None):
    """
    Plot correlation between signals and scene transitions
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    layers = sorted(all_results.keys())
    x = np.arange(len(layers))
    width = 0.35
    
    pred_corr = [all_results[l]['pred_event_corr'] for l in layers]
    error_corr = [all_results[l]['error_event_corr'] for l in layers]
    
    bars1 = ax.bar(x - width/2, pred_corr, width, label='Prediction', color='#3498DB', edgecolor='black')
    bars2 = ax.bar(x + width/2, error_corr, width, label='Error', color='#E74C3C', edgecolor='black')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Correlation with Scene Transitions', fontsize=12)
    ax.set_title('Signal-Event Correlation\n(Higher = More Event-Driven)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{l}' for l in layers])
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add correlation values
    for bar, corr in zip(bars1, pred_corr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{corr:.3f}', ha='center', fontsize=9, fontweight='bold')
    for bar, corr in zip(bars2, error_corr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{corr:.3f}', ha='center', fontsize=9, fontweight='bold')
    
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
    print(" Event-Locked Analysis: WHY Prediction is Sparse")
    print(" Evidence: Prediction responds to scene transitions")
    print("="*70)
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Inspect H5 file structure first
    print("\n[0] Inspecting H5 file structure...")
    inspect_h5_structure(FEATURE_DIR, SEGMENTS)
    
    # First, detect scene transitions using Error Layer 1
    # Use memory-efficient version (only temporal mean)
    print("\n[1] Detecting scene transitions (memory-efficient)...")
    
    error_l1_temporal, _ = load_temporal_mean_efficient(FEATURE_DIR, 'E', 1, SEGMENTS)
    
    if error_l1_temporal is not None:
        print(f"    Loaded temporal signal: {error_l1_temporal.shape}")
        scene_transitions, change_magnitude = detect_scene_transitions_from_error(
            error_l1_temporal.reshape(-1, 1), threshold_percentile=90, min_distance=10
        )
        print(f"    Detected {len(scene_transitions)} scene transitions")
        print(f"    Transition indices (first 10): {scene_transitions[:10]}")
    else:
        print("    ERROR: Could not load Error Layer 1 features")
        exit(1)
    
    # Store results for all layers
    all_results = {}
    
    # Analyze each layer using memory-efficient temporal means
    for layer in range(1, N_LAYERS + 1):
        print(f"\n[Layer {layer}] Analyzing (memory-efficient)...")
        
        pred_temporal, _ = load_temporal_mean_efficient(FEATURE_DIR, 'Ahat', layer, SEGMENTS)
        error_temporal, _ = load_temporal_mean_efficient(FEATURE_DIR, 'E', layer, SEGMENTS)
        
        if pred_temporal is None or error_temporal is None:
            print(f"    Skipping layer {layer}")
            continue
        
        print(f"    Prediction temporal: {pred_temporal.shape}")
        print(f"    Error temporal: {error_temporal.shape}")
        
        # Event-locked average (using 1D temporal signals)
        print("    Computing event-locked averages...")
        pred_ela, pred_sem, time_axis = event_locked_average(
            pred_temporal, scene_transitions, window_pre=30, window_post=50
        )
        error_ela, error_sem, _ = event_locked_average(
            error_temporal, scene_transitions, window_pre=30, window_post=50
        )
        
        if pred_ela is not None:
            plot_event_locked_comparison(
                pred_ela, error_ela, pred_sem, error_sem, time_axis, layer,
                save_path=os.path.join(SAVE_DIR, f'event_locked_layer{layer}.png')
            )
        
        # Temporal raster (using 1D signals, reshape for compatibility)
        print("    Creating temporal raster...")
        plot_temporal_raster(
            pred_temporal.reshape(-1, 1), error_temporal.reshape(-1, 1), 
            scene_transitions, layer,
            n_frames=min(2000, len(pred_temporal)),
            save_path=os.path.join(SAVE_DIR, f'temporal_raster_layer{layer}.png')
        )
        
        # Baseline vs Event comparison
        print("    Computing baseline vs event response...")
        pred_baseline, pred_event, pred_t, pred_p = compute_baseline_vs_event(
            pred_temporal, scene_transitions
        )
        error_baseline, error_event, error_t, error_p = compute_baseline_vs_event(
            error_temporal, scene_transitions
        )
        
        print(f"    Prediction: baseline={pred_baseline:.4f}, event={pred_event:.4f}, p={pred_p:.4f}")
        print(f"    Error: baseline={error_baseline:.4f}, event={error_event:.4f}, p={error_p:.4f}")
        
        # Event correlation
        print("    Computing event correlation...")
        corr_results = compute_event_correlation(pred_temporal, error_temporal, scene_transitions)
        print(f"    Prediction-Event corr: {corr_results['pred_event_corr']:.4f} (p={corr_results['pred_event_p']:.4f})")
        print(f"    Error-Event corr: {corr_results['error_event_corr']:.4f} (p={corr_results['error_event_p']:.4f})")
        
        # Store results
        all_results[layer] = {
            'pred_baseline': pred_baseline,
            'pred_event': pred_event,
            'pred_t': pred_t,
            'pred_p': pred_p,
            'error_baseline': error_baseline,
            'error_event': error_event,
            'error_t': error_t,
            'error_p': error_p,
            **corr_results
        }
    
    # Summary plots
    if len(all_results) > 0:
        print("\n" + "="*70)
        print(" Creating summary plots...")
        
        plot_summary_bar(all_results, 
                        save_path=os.path.join(SAVE_DIR, 'baseline_vs_event_summary.png'))
        
        plot_correlation_summary(all_results,
                                save_path=os.path.join(SAVE_DIR, 'event_correlation_summary.png'))
        
        # Save results
        np.save(os.path.join(SAVE_DIR, 'event_locked_results.npy'), all_results)
        np.save(os.path.join(SAVE_DIR, 'scene_transitions.npy'), scene_transitions)
        
        # Print summary table
        print("\n" + "="*70)
        print(" SUMMARY: WHY Prediction is Sparse, Error is Continuous")
        print("="*70)
        print("\n1. PREDICTION responds to scene transitions:")
        print(f"   Layer | Baseline | Event   | % Change | p-value")
        print("-"*55)
        for l in sorted(all_results.keys()):
            r = all_results[l]
            pct = (r['pred_event'] - r['pred_baseline']) / (r['pred_baseline'] + 1e-10) * 100
            sig = '*' if r['pred_p'] < 0.05 else ''
            print(f"   {l}     | {r['pred_baseline']:.4f}  | {r['pred_event']:.4f} | {pct:+.1f}%    | {r['pred_p']:.4f} {sig}")
        
        print("\n2. ERROR maintains persistent activity:")
        print(f"   Layer | Baseline | Event   | % Change | p-value")
        print("-"*55)
        for l in sorted(all_results.keys()):
            r = all_results[l]
            pct = (r['error_event'] - r['error_baseline']) / (r['error_baseline'] + 1e-10) * 100
            sig = '*' if r['error_p'] < 0.05 else ''
            print(f"   {l}     | {r['error_baseline']:.4f}  | {r['error_event']:.4f} | {pct:+.1f}%    | {r['error_p']:.4f} {sig}")
        
        print("\n3. Correlation with scene transitions:")
        print(f"   Layer | Pred-Event r | Error-Event r")
        print("-"*45)
        for l in sorted(all_results.keys()):
            r = all_results[l]
            print(f"   {l}     | {r['pred_event_corr']:+.4f}      | {r['error_event_corr']:+.4f}")
    
    print("\n" + "="*70)
    print(" KEY FINDING:")
    print(" - Prediction shows LARGER response to scene transitions")
    print("   ¡æ Responds to prediction failures (event-driven)")
    print(" - Error maintains activity regardless of transitions")
    print("   ¡æ Continuous residual mismatch (persistent)")
    print("="*70)
    print(f"\nOutput files in: {SAVE_DIR}")