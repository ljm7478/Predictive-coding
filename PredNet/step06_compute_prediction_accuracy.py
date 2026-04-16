"""
=============================================================================
PredNet Prediction Accuracy: MSE & SSIM
=============================================================================
Computes frame-level MSE and SSIM between predicted frames (Ahat) and
actual input frames (A), reconstructed from Ahat and E.

In PredNet, E0 = [ReLU(A - Ahat), ReLU(Ahat - A)]  (2x channels, last axis)
Therefore:  A = Ahat + E0[..., :n_ch] - E0[..., n_ch:]

Data format: channels-last (batch, time, H, W, C)

Usage:
    python prednet_prediction_accuracy.py --diagnose
    python prednet_prediction_accuracy.py
    python prednet_prediction_accuracy.py --no_plot

Requirements:
    pip install h5py numpy scikit-image pandas matplotlib
=============================================================================
"""

import os
import numpy as np
import h5py
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# [*] CONFIGURATION
# =============================================================================

ORIGINAL_DIR = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4"
MODIFIED_DIR = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/CNN_like/h5/pt_kitti_ft_Lall_nt150/layer4"
OUTPUT_DIR = "/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_model_accuracy"

FILE_PATTERN = "seg{run_idx}"  # seg0, seg1, ...

AHAT_KEY = "Ahat0"
E_KEY = "E0"

N_INPUT_CH = 3
NUM_RUNS = 8
SKIP_FIRST_N = 1

# --- Plot style ---
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

COLORS = {
    'original': '#2166AC',
    'modified': '#B2182B',
}


# =============================================================================
# DATA LOADING & RECONSTRUCTION
# =============================================================================

def load_h5_array(h5_path, key):
    """
    Load array from h5 file.
    Your data: (batch, time, H, W, C) -> squeeze to (time, H, W, C)
    """
    with h5py.File(h5_path, 'r') as f:
        if key not in f:
            available = list(f.keys())
            raise KeyError(f"Key '{key}' not found in {h5_path}. Available: {available}")
        data = f[key][:]

    # (batch, time, H, W, C) -> (time, H, W, C)
    if data.ndim == 5:
        data = data[0]

    return data


def reconstruct_actual(ahat, error, n_ch=3):
    """
    Reconstruct actual input A from Ahat and E.

    Data is channels-last: (time, H, W, C)
    E0 last axis = [ReLU(A - Ahat), ReLU(Ahat - A)] = 2*n_ch channels

    A = Ahat + E_pos - E_neg
    """
    assert error.shape[-1] == 2 * n_ch, \
        f"E0 last dim should be {2*n_ch}, got {error.shape[-1]}. Check N_INPUT_CH."
    assert ahat.shape[-1] == n_ch, \
        f"Ahat0 last dim should be {n_ch}, got {ahat.shape[-1]}."

    e_pos = error[..., :n_ch]    # ReLU(A - Ahat)
    e_neg = error[..., n_ch:]    # ReLU(Ahat - A)

    actual = ahat + e_pos - e_neg
    return actual


def diagnose_h5(directory, max_files=3):
    """Print h5 file structure for debugging."""
    print("\n" + "=" * 60)
    print("  H5 File Structure Diagnosis")
    print("=" * 60)

    h5_files = sorted(Path(directory).glob("**/*.h5"))[:max_files]
    if not h5_files:
        print(f"  [X] No .h5 files found in: {directory}")
        return

    for fpath in h5_files:
        print(f"\n[FILE] {fpath.name}")
        with h5py.File(fpath, 'r') as f:
            print(f"   Keys: {list(f.keys())}")
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    d = f[key]
                    print(f"   {key}: shape={d.shape}, dtype={d.dtype}, "
                          f"range=[{d[:].min():.4f}, {d[:].max():.4f}]")
                elif isinstance(f[key], h5py.Group):
                    print(f"   {key}/ (Group): {list(f[key].keys())}")

    print(f"\n  [INFO] Data format: channels-last (batch, time, H, W, C)")
    print(f"    Ahat0 last dim: {N_INPUT_CH}")
    print(f"    E0 last dim:    {N_INPUT_CH * 2}")
    print(f"    Reconstruction: A = Ahat0 + E0[..., :{N_INPUT_CH}] - E0[..., {N_INPUT_CH}:]")


# =============================================================================
# METRICS
# =============================================================================

def compute_mse(pred, actual):
    """
    Per-frame MSE.
    Input: (time, H, W, C)
    Returns: (mean, per_frame_array)
    """
    assert pred.shape == actual.shape
    n = pred.shape[0]
    pf = np.zeros(n)
    for t in range(n):
        pf[t] = np.mean((pred[t] - actual[t]) ** 2)
    return np.mean(pf), pf


def compute_ssim(pred, actual, data_range=None):
    """
    Per-frame SSIM.
    Input: (time, H, W, C) -- already channels-last, ready for skimage
    Returns: (mean, per_frame_array)
    """
    assert pred.shape == actual.shape
    n = pred.shape[0]
    n_ch = pred.shape[-1]
    pf = np.zeros(n)

    if data_range is None:
        if actual.max() <= 1.0:
            data_range = 1.0
        elif actual.max() <= 255.0:
            data_range = 255.0
        else:
            data_range = max(actual.max() - actual.min(), 1e-8)

    for t in range(n):
        if n_ch == 1:
            pf[t] = ssim(actual[t, :, :, 0], pred[t, :, :, 0],
                          data_range=data_range)
        elif n_ch == 3:
            pf[t] = ssim(actual[t], pred[t],
                          data_range=data_range, channel_axis=-1)
        else:
            ch_vals = []
            for c in range(n_ch):
                ch_vals.append(ssim(actual[t, :, :, c], pred[t, :, :, c],
                                    data_range=data_range))
            pf[t] = np.mean(ch_vals)

    return np.mean(pf), pf


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model_dir, model_name, ahat_suffix="_Ahat.h5",
                   e_suffix="_E.h5"):
    """
    Evaluate a single model across all runs.
    Reconstructs actual frames from Ahat + E, then computes MSE & SSIM.
    """
    all_mse, all_ssim = [], []
    all_pf_mse, all_pf_ssim = [], []

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"  Directory:  {model_dir}")
    print(f"  Reconstruct A from Ahat0 + E0")
    print(f"{'='*60}")

    model_path = Path(model_dir)

    for run_idx in range(NUM_RUNS):
        run_id = FILE_PATTERN.format(run_idx=run_idx)

        ahat_file = model_path / f"{run_id}{ahat_suffix}"
        e_file = model_path / f"{run_id}{e_suffix}"
        combined_file = model_path / f"{run_id}.h5"

        try:
            if ahat_file.exists() and e_file.exists():
                ahat = load_h5_array(ahat_file, AHAT_KEY)
                error = load_h5_array(e_file, E_KEY)
            elif combined_file.exists():
                ahat = load_h5_array(combined_file, AHAT_KEY)
                error = load_h5_array(combined_file, E_KEY)
            elif ahat_file.exists():
                ahat = load_h5_array(ahat_file, AHAT_KEY)
                try:
                    error = load_h5_array(ahat_file, E_KEY)
                except KeyError:
                    print(f"  [!] Run {run_idx}: E0 not found. Skipping.")
                    continue
            else:
                print(f"  [!] Run {run_idx}: File not found. Skipping.")
                print(f"     Tried: {ahat_file}")
                continue

            # Data is now (time, H, W, C) after squeeze
            actual = reconstruct_actual(ahat, error, n_ch=N_INPUT_CH)

            # Skip warmup frames
            ahat = ahat[SKIP_FIRST_N:]
            actual = actual[SKIP_FIRST_N:]

            min_frames = min(len(ahat), len(actual))
            ahat = ahat[:min_frames]
            actual = actual[:min_frames]

            # Already (time, H, W, C) -- no transpose needed
            mse_val, pf_mse = compute_mse(ahat, actual)
            ssim_val, pf_ssim = compute_ssim(ahat, actual)

            all_mse.append(mse_val)
            all_ssim.append(ssim_val)
            all_pf_mse.extend(pf_mse)
            all_pf_ssim.extend(pf_ssim)

            print(f"  Run {run_idx}: MSE={mse_val:.4f}, SSIM={ssim_val:.4f} "
                  f"({min_frames} frames, shape={ahat.shape})")

        except Exception as e:
            print(f"  [X] Run {run_idx}: Error - {e}")
            continue

    if not all_mse:
        print(f"  [X] No valid runs found for {model_name}")
        return None

    return {
        'model': model_name,
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'per_frame_mse': np.array(all_pf_mse),
        'per_frame_ssim': np.array(all_pf_ssim),
        'n_runs': len(all_mse),
        'n_frames_total': len(all_pf_mse),
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def print_paper_table(results_list):
    """Print results in paper-ready table format."""
    print("\n" + "=" * 60)
    print("  Table: Prediction Accuracy (MSE [lower=better] / SSIM [higher=better])")
    print("=" * 60)
    print(f"{'Model':<30} {'MSE':>10} {'SSIM':>10}")
    print("-" * 60)
    for r in results_list:
        if r is None:
            continue
        print(f"{r['model']:<30} {r['mse_mean']:>10.4f} {r['ssim_mean']:>10.3f}")
    print("-" * 60)
    print()
    print("  (With standard deviation across runs)")
    print(f"{'Model':<30} {'MSE':>20} {'SSIM':>20}")
    print("-" * 70)
    for r in results_list:
        if r is None:
            continue
        print(f"{r['model']:<30} {r['mse_mean']:.4f} +/- {r['mse_std']:.4f}"
              f"     {r['ssim_mean']:.3f} +/- {r['ssim_std']:.3f}")
    print("-" * 70)


def save_csv_and_latex(results_list, out_dir):
    """Save CSV summary and LaTeX table."""
    rows = []
    for r in results_list:
        if r is None:
            continue
        rows.append({
            'Model': r['model'],
            'MSE': r['mse_mean'], 'MSE_std': r['mse_std'],
            'SSIM': r['ssim_mean'], 'SSIM_std': r['ssim_std'],
            'N_runs': r['n_runs'], 'N_frames': r['n_frames_total'],
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "prediction_accuracy.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] CSV saved: {csv_path}")

    latex_path = os.path.join(out_dir, "prediction_accuracy_latex.txt")
    with open(latex_path, 'w') as f:
        f.write("% LaTeX table for paper\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Evaluation of next-frame predictions on [Dataset Name].}\n")
        f.write("\\label{tab:prediction_accuracy}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write(" & MSE $\\downarrow$ & SSIM $\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        for r in results_list:
            if r is None:
                continue
            f.write(f"PredNet {r['model']:<20} & "
                    f"\\textbf{{{r['mse_mean']:.4f}}} & "
                    f"\\textbf{{{r['ssim_mean']:.3f}}} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    print(f"[OK] LaTeX saved: {latex_path}")

    return df


def save_per_frame(results_list, out_dir):
    """Save per-frame npy files."""
    for r in results_list:
        if r is None:
            continue
        tag = r['model'].replace(' ', '_').replace('/', '_')
        np.save(os.path.join(out_dir, f"per_frame_mse_{tag}.npy"), r['per_frame_mse'])
        np.save(os.path.join(out_dir, f"per_frame_ssim_{tag}.npy"), r['per_frame_ssim'])
    print(f"[OK] Per-frame npy saved: {out_dir}")


# =============================================================================
# PLOTS
# =============================================================================

def plot_bar(results_list, out_dir):
    """Bar plot: MSE & SSIM comparison."""
    valid = [r for r in results_list if r is not None]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    models = [r['model'] for r in valid]
    n = len(models)
    x = np.arange(n)
    w = 0.5
    colors = [COLORS['original'], COLORS['modified']][:n]

    ax = axes[0]
    mse_vals = [r['mse_mean'] for r in valid]
    mse_stds = [r['mse_std'] for r in valid]
    bars = ax.bar(x, mse_vals, w, yerr=mse_stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 1.0})
    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel('MSE (lower=better)')
    ax.set_title('Mean Squared Error', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    ssim_vals = [r['ssim_mean'] for r in valid]
    ssim_stds = [r['ssim_std'] for r in valid]
    bars = ax.bar(x, ssim_vals, w, yerr=ssim_stds, capsize=4,
                  color=colors, edgecolor='black', linewidth=0.8,
                  error_kw={'linewidth': 1.0})
    for bar, val in zip(bars, ssim_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_ylabel('SSIM (higher=better)')
    ax.set_title('Structural Similarity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0, top=1.0)

    plt.tight_layout()
    p = os.path.join(out_dir, "prediction_accuracy_barplot")
    plt.savefig(p + ".png")
    plt.savefig(p + ".pdf")
    print(f"[OK] Bar plot saved: {p}.png")
    plt.close()


def plot_timeseries(results_list, out_dir):
    """Per-frame MSE/SSIM time-series."""
    valid = [r for r in results_list if r is not None]
    if not valid:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    color_list = [COLORS['original'], COLORS['modified']]

    for i, r in enumerate(valid):
        c = color_list[i % 2]
        label = r['model']
        n_frames = len(r['per_frame_mse'])
        x = np.arange(n_frames)
        window = min(50, n_frames // 10) if n_frames > 100 else 1

        if window > 1:
            mse_s = pd.Series(r['per_frame_mse']).rolling(window, center=True).mean()
            ssim_s = pd.Series(r['per_frame_ssim']).rolling(window, center=True).mean()
            axes[0].plot(x, mse_s, color=c, label=label, linewidth=1.2)
            axes[1].plot(x, ssim_s, color=c, label=label, linewidth=1.2)

            mse_q25 = pd.Series(r['per_frame_mse']).rolling(window, center=True).quantile(0.25)
            mse_q75 = pd.Series(r['per_frame_mse']).rolling(window, center=True).quantile(0.75)
            ssim_q25 = pd.Series(r['per_frame_ssim']).rolling(window, center=True).quantile(0.25)
            ssim_q75 = pd.Series(r['per_frame_ssim']).rolling(window, center=True).quantile(0.75)
            axes[0].fill_between(x, mse_q25, mse_q75, color=c, alpha=0.15)
            axes[1].fill_between(x, ssim_q25, ssim_q75, color=c, alpha=0.15)
        else:
            axes[0].plot(x, r['per_frame_mse'], color=c, label=label, linewidth=0.8)
            axes[1].plot(x, r['per_frame_ssim'], color=c, label=label, linewidth=0.8)

    axes[0].set_ylabel('MSE (lower=better)')
    axes[0].set_title('Per-frame MSE', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].set_ylabel('SSIM (higher=better)')
    axes[1].set_xlabel('Frame')
    axes[1].set_title('Per-frame SSIM', fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    p = os.path.join(out_dir, "prediction_accuracy_timeseries")
    plt.savefig(p + ".png")
    plt.savefig(p + ".pdf")
    print(f"[OK] Time-series saved: {p}.png")
    plt.close()


def plot_violin(results_list, out_dir):
    """Violin plot: distribution of per-frame metrics."""
    valid = [r for r in results_list if r is not None]
    if not valid:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    color_list = [COLORS['original'], COLORS['modified']]

    mse_data = [r['per_frame_mse'] for r in valid]
    parts = axes[0].violinplot(mse_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_list[i % 2])
        pc.set_alpha(0.6)
    axes[0].set_ylabel('MSE (lower=better)')
    axes[0].set_title('MSE Distribution', fontweight='bold')
    axes[0].set_xticks(range(1, len(valid) + 1))
    axes[0].set_xticklabels([r['model'] for r in valid], rotation=15, ha='right')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    ssim_data = [r['per_frame_ssim'] for r in valid]
    parts = axes[1].violinplot(ssim_data, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_list[i % 2])
        pc.set_alpha(0.6)
    axes[1].set_ylabel('SSIM (higher=better)')
    axes[1].set_title('SSIM Distribution', fontweight='bold')
    axes[1].set_xticks(range(1, len(valid) + 1))
    axes[1].set_xticklabels([r['model'] for r in valid], rotation=15, ha='right')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.tight_layout()
    p = os.path.join(out_dir, "prediction_accuracy_distribution")
    plt.savefig(p + ".png")
    plt.savefig(p + ".pdf")
    print(f"[OK] Violin plot saved: {p}.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='PredNet Prediction Accuracy: MSE & SSIM')
    parser.add_argument('--diagnose', action='store_true',
                        help='Diagnose h5 file structure (run this first!)')
    parser.add_argument('--original_dir', type=str, default=ORIGINAL_DIR)
    parser.add_argument('--modified_dir', type=str, default=MODIFIED_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--ahat_key', type=str, default=AHAT_KEY)
    parser.add_argument('--e_key', type=str, default=E_KEY)
    parser.add_argument('--n_input_ch', type=int, default=N_INPUT_CH)
    parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    AHAT_KEY = args.ahat_key
    E_KEY = args.e_key
    N_INPUT_CH = args.n_input_ch

    # --- Diagnose mode ---
    if args.diagnose:
        print("Running diagnosis on both directories...")
        if os.path.exists(args.original_dir):
            diagnose_h5(args.original_dir)
        else:
            print(f"  [X] Original dir not found: {args.original_dir}")
        if os.path.exists(args.modified_dir):
            diagnose_h5(args.modified_dir)
        else:
            print(f"  [X] Modified dir not found: {args.modified_dir}")
        exit(0)

    # --- Evaluate ---
    results = []

    r_orig = evaluate_model(
        model_dir=args.original_dir,
        model_name="Original (Error-prop.)",
    )
    results.append(r_orig)

    r_mod = evaluate_model(
        model_dir=args.modified_dir,
        model_name="Modified (Rep.-prop.)",
    )
    results.append(r_mod)

    # --- Print & Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    print_paper_table(results)
    save_csv_and_latex(results, args.output_dir)
    save_per_frame(results, args.output_dir)

    # --- Plot ---
    if not args.no_plot:
        print("\nGenerating figures...")
        plot_bar(results, args.output_dir)
        plot_timeseries(results, args.output_dir)
        plot_violin(results, args.output_dir)

    print("\n[DONE] All evaluations and figures complete!")