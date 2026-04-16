"""
LSTM Full Chain Analysis for PredNet
i, f -> C -> o -> R = o * tanh(C) -> Ahat = Conv(R)
"""

from __future__ import print_function
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
NPY_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/pt_kitti_ft_Lall_nt150/layer4/run1'
H5_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4'
SAVE_DIR = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_activation_analysis/test_GUMP/pt_kitti_ft_Titanic_Lall_nt150/original/layer4/lstm_gate'
SEG_NAME = 'seg0'

FPS = 25


def get_average_timeseries(data):
    """Get average time-series across all spatial units."""
    if data.ndim == 5:
        data = data[0]
    
    if data.ndim == 4:
        T = data.shape[0]
        data_flat = data.reshape(T, -1)
    else:
        raise ValueError("Unexpected shape: {}".format(data.shape))
    
    avg_ts = np.mean(data_flat, axis=1)
    return avg_ts


def load_all_data():
    """Load I, F, O, R (npy) and C (h5)."""
    print("Loading all LSTM components...")
    
    data = {'I': {}, 'F': {}, 'O': {}, 'R': {}, 'C': {}}
    
    # Load from npy
    for var in ['I', 'F', 'O', 'R']:
        for l in range(4):
            path = os.path.join(NPY_DIR, '{}{}.npy'.format(var, l))
            if os.path.exists(path):
                data[var][l] = np.load(path)
                print("  Loaded {}{}: shape {}".format(var, l, data[var][l].shape))
            else:
                print("  WARNING: {}{}.npy not found".format(var, l))
    
    # Load C from h5
    c_path = os.path.join(H5_DIR, '{}_C.h5'.format(SEG_NAME))
    if os.path.exists(c_path):
        with h5py.File(c_path, 'r') as f:
            for l in range(4):
                key = 'C{}'.format(l)
                if key in f.keys():
                    data['C'][l] = f[key][:]
                    print("  Loaded C{}: shape {}".format(l, data['C'][l].shape))
    
    return data


def analyze_full_chain(data, save_dir):
    """Analyze full LSTM chain: i, f -> C -> o -> R -> Ahat"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Compute averages
    print("\nComputing average time-series...")
    avg = {'I': {}, 'F': {}, 'O': {}, 'R': {}, 'C': {}}
    
    for var in ['I', 'F', 'O', 'R', 'C']:
        for l in range(4):
            if l in data[var]:
                avg[var][l] = get_average_timeseries(data[var][l])
    
    # =================================================================
    # Plot 1: All components summary (bar chart)
    # =================================================================
    print("\nPlotting summary statistics...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    
    components = [
        ('I', 'Input Gate (i)', (0, 1), axes[0, 0]),
        ('F', 'Forget Gate (f)', (0, 1), axes[0, 1]),
        ('O', 'Output Gate (o)', (0, 1), axes[0, 2]),
        ('C', 'Cell State (C)', None, axes[1, 0]),
        ('R', 'Hidden State (R)', None, axes[1, 1]),
    ]
    
    for var, title, ylim, ax in components:
        means = [avg[var][l].mean() for l in range(4)]
        stds = [avg[var][l].std() for l in range(4)]
        
        ax.bar(range(4), means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_ylabel('Mean Value', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['L0', 'L1', 'L2', 'L3'])
        if ylim:
            ax.set_ylim(ylim)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)
    
    # Summary text in last panel
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "LSTM Chain Summary\n" + "="*25 + "\n\n"
    summary_text += "Layer 0 vs Layer 3:\n\n"
    
    for var in ['I', 'F', 'O', 'C', 'R']:
        m0 = avg[var][0].mean()
        m3 = avg[var][3].mean()
        ratio = m3 / m0 if m0 != 0 else 0
        summary_text += "{}: L0={:.3f}, L3={:.3f}\n   (ratio={:.2f})\n".format(var, m0, m3, ratio)
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    fig.suptitle('LSTM Full Chain Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    plt.savefig(os.path.join(save_dir, 'LSTM_FullChain_Summary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: LSTM_FullChain_Summary.png")
    
    # =================================================================
    # Plot 2: Time series comparison (L0 vs L3)
    # =================================================================
    print("\nPlotting time series (L0 vs L3)...")
    
    T = min(500, len(avg['I'][0]))
    time_sec = np.arange(T) / float(FPS)
    
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    
    for idx, var in enumerate(['I', 'F', 'O', 'C', 'R']):
        # L0 (left)
        ax = axes[idx, 0]
        ax.plot(time_sec, avg[var][0][:T], 'g-', linewidth=0.8, alpha=0.8)
        ax.set_title('{} - Layer 0'.format(var), fontsize=11, fontweight='bold')
        ax.set_ylabel(var, fontsize=10)
        if idx == 4:
            ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if var in ['I', 'F', 'O']:
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        
        # L3 (right)
        ax = axes[idx, 1]
        ax.plot(time_sec, avg[var][3][:T], 'r-', linewidth=0.8, alpha=0.8)
        ax.set_title('{} - Layer 3'.format(var), fontsize=11, fontweight='bold')
        ax.set_ylabel(var, fontsize=10)
        if idx == 4:
            ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.grid(True, alpha=0.3)
        if var in ['I', 'F', 'O']:
            ax.set_ylim(0, 1)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle('LSTM Components Over Time: L0 (green) vs L3 (red)', fontsize=13, fontweight='bold', y=0.99)
    plt.subplots_adjust(top=0.94, hspace=0.35, wspace=0.25)
    plt.savefig(os.path.join(save_dir, 'LSTM_TimeSeries_L0vsL3.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: LSTM_TimeSeries_L0vsL3.png")
    
    # =================================================================
    # Plot 3: Chain flow visualization
    # =================================================================
    print("\nPlotting chain flow...")
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    for l in range(4):
        ax = axes[l]
        
        # Get means
        i_m = avg['I'][l].mean()
        f_m = avg['F'][l].mean()
        o_m = avg['O'][l].mean()
        c_m = avg['C'][l].mean()
        r_m = avg['R'][l].mean()
        
        # Bar for this layer
        vars_names = ['i', 'f', 'o', 'C', 'R']
        vars_vals = [i_m, f_m, o_m, c_m, r_m]
        vars_colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6', '#e74c3c']
        
        bars = ax.bar(range(5), vars_vals, color=vars_colors, alpha=0.8)
        ax.set_xticks(range(5))
        ax.set_xticklabels(vars_names, fontsize=11)
        ax.set_title('Layer {}'.format(l), fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Value', fontsize=10)
        
        # Add value labels
        for bar, val in zip(bars, vars_vals):
            height = bar.get_height()
            if height >= 0:
                va = 'bottom'
                y_pos = height + 0.02
            else:
                va = 'top'
                y_pos = height - 0.02
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    '{:.3f}'.format(val), ha='center', va=va, fontsize=9)
    
    fig.suptitle('LSTM Chain: i, f, o (gates) --> C (cell) --> R (output)', fontsize=13, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.88, wspace=0.3)
    plt.savefig(os.path.join(save_dir, 'LSTM_ChainFlow_ByLayer.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Saved: LSTM_ChainFlow_ByLayer.png")
    
    # =================================================================
    # Print numeric summary
    # =================================================================
    print("\n" + "="*70)
    print("FULL CHAIN NUMERIC SUMMARY")
    print("="*70)
    
    print("\n{:>8} {:>12} {:>12} {:>12} {:>12}".format("Var", "L0", "L1", "L2", "L3"))
    print("-"*70)
    
    for var in ['I', 'F', 'O', 'C', 'R']:
        vals = [avg[var][l].mean() for l in range(4)]
        print("{:>8} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(var, *vals))
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    i0, i3 = avg['I'][0].mean(), avg['I'][3].mean()
    f0, f3 = avg['F'][0].mean(), avg['F'][3].mean()
    o0, o3 = avg['O'][0].mean(), avg['O'][3].mean()
    c0, c3 = avg['C'][0].mean(), avg['C'][3].mean()
    r0, r3 = avg['R'][0].mean(), avg['R'][3].mean()
    
    print("\nL0 --> L3 changes:")
    print("  Input gate (i):  {:.3f} --> {:.3f} ({:.1f}x)".format(i0, i3, i3/i0 if i0 else 0))
    print("  Forget gate (f): {:.3f} --> {:.3f} ({:.1f}x)".format(f0, f3, f3/f0 if f0 else 0))
    print("  Output gate (o): {:.3f} --> {:.3f} ({:.1f}x)".format(o0, o3, o3/o0 if o0 else 0))
    print("  Cell state (C):  {:.3f} --> {:.3f} ({:.1f}x)".format(c0, c3, c3/c0 if c0 else 0))
    print("  Hidden state (R):{:.3f} --> {:.3f} ({:.1f}x)".format(r0, r3, r3/r0 if r0 else 0))
    
    print("\nChain effect:")
    print("  i,f LOW --> C shrinks")
    print("  o LOW + C LOW --> R very small")
    print("  R small --> Ahat = Conv(R) sparse")
    print("  A continuous, Ahat sparse --> Error = A - Ahat continuous")
    
    print("\n" + "="*70)
    print("All plots saved to: {}".format(save_dir))
    print("="*70)


if __name__ == "__main__":
    data = load_all_data()
    analyze_full_chain(data, SAVE_DIR)