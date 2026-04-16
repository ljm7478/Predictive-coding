"""
LSTM Gate Analysis for PredNet
- Hypothesis: Higher layers have more closed input gates, making Prediction sparse
"""

from __future__ import print_function
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. Load Data
# =============================================================================

# Path settings
npy_dir = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/npy/pt_kitti_ft_Lall_nt150/layer4/run1'
h5_file = '/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/01_gump/result/prednet_original/h5/pt_kitti_ft_Lall_nt150/layer4/seg0_C.h5'

# Load I, F (npy)
I = {}
F = {}
for l in range(4):
    I[l] = np.load(os.path.join(npy_dir, 'I{}.npy'.format(l)))
    F[l] = np.load(os.path.join(npy_dir, 'F{}.npy'.format(l)))
    print("Layer {}: I shape = {}, F shape = {}".format(l, I[l].shape, F[l].shape))

# =============================================================================
# 2. Basic Statistics
# =============================================================================

print("\n" + "="*60)
print("INPUT GATE (i) - How much new info is accepted")
print("i ~ 1: gate open (accept new info)")
print("i ~ 0: gate closed (ignore new info)")
print("="*60)

for l in range(4):
    ig = I[l]
    # shape could be (1, T, C, H, W) or (T, C, H, W)
    if ig.ndim == 5:
        ig = ig[0]  # (T, C, H, W)
    
    # temporal mean (spatial + channel average)
    ig_temporal = ig.mean(axis=(1, 2, 3))  # (T,)
    
    print("Layer {}: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
        l, ig_temporal.mean(), ig_temporal.std(), ig_temporal.min(), ig_temporal.max()))

print("\n" + "="*60)
print("FORGET GATE (f) - How much old state is kept")
print("f ~ 1: keep old state")
print("f ~ 0: delete old state")
print("="*60)

for l in range(4):
    fg = F[l]
    if fg.ndim == 5:
        fg = fg[0]
    
    fg_temporal = fg.mean(axis=(1, 2, 3))
    
    print("Layer {}: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
        l, fg_temporal.mean(), fg_temporal.std(), fg_temporal.min(), fg_temporal.max()))

# =============================================================================
# 3. Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Panel A: Input Gate Mean by Layer ---
ax = axes[0, 0]
layer_means_i = []
layer_stds_i = []
for l in range(4):
    ig = I[l][0] if I[l].ndim == 5 else I[l]
    ig_temporal = ig.mean(axis=(1, 2, 3))
    layer_means_i.append(ig_temporal.mean())
    layer_stds_i.append(ig_temporal.std())

ax.bar(range(4), layer_means_i, yerr=layer_stds_i, capsize=5, 
       color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], alpha=0.8)
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Input Gate Mean', fontsize=12)
ax.set_title('A. Input Gate by Layer\n(Lower = blocks new info)', fontsize=14)
ax.set_xticks(range(4))
ax.set_xticklabels(['L0', 'L1', 'L2', 'L3'])
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='neutral (0.5)')
ax.legend()

# --- Panel B: Forget Gate Mean by Layer ---
ax = axes[0, 1]
layer_means_f = []
layer_stds_f = []
for l in range(4):
    fg = F[l][0] if F[l].ndim == 5 else F[l]
    fg_temporal = fg.mean(axis=(1, 2, 3))
    layer_means_f.append(fg_temporal.mean())
    layer_stds_f.append(fg_temporal.std())

ax.bar(range(4), layer_means_f, yerr=layer_stds_f, capsize=5,
       color=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c'], alpha=0.8)
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Forget Gate Mean', fontsize=12)
ax.set_title('B. Forget Gate by Layer\n(Higher = keeps old state)', fontsize=14)
ax.set_xticks(range(4))
ax.set_xticklabels(['L0', 'L1', 'L2', 'L3'])
ax.set_ylim(0, 1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

# --- Panel C: Input Gate Over Time (L0 vs L3) ---
ax = axes[1, 0]
ig0 = I[0][0] if I[0].ndim == 5 else I[0]
ig3 = I[3][0] if I[3].ndim == 5 else I[3]
ig0_temporal = ig0.mean(axis=(1, 2, 3))
ig3_temporal = ig3.mean(axis=(1, 2, 3))

T = min(len(ig0_temporal), 500)  # first 500 frames only
ax.plot(range(T), ig0_temporal[:T], label='Layer 0', color='#2ecc71', alpha=0.8)
ax.plot(range(T), ig3_temporal[:T], label='Layer 3', color='#e74c3c', alpha=0.8)
ax.set_xlabel('Time (frames)', fontsize=12)
ax.set_ylabel('Input Gate', fontsize=12)
ax.set_title('C. Input Gate Over Time (L0 vs L3)', fontsize=14)
ax.legend()
ax.set_ylim(0, 1)

# --- Panel D: Input Gate Distribution (L0 vs L3) ---
ax = axes[1, 1]
ax.hist(ig0_temporal, bins=50, alpha=0.6, label='Layer 0', color='#2ecc71', normed=True)
ax.hist(ig3_temporal, bins=50, alpha=0.6, label='Layer 3', color='#e74c3c', normed=True)
ax.set_xlabel('Input Gate Value', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('D. Input Gate Distribution', fontsize=14)
ax.legend()
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/combinelab/03_user/jungmin/01_project/02_PredNet/jmPNET/data/99_activation_analysis/test_GUMP/pt_kitti_ft_Titanic_Lall_nt150/original/layer4/lstm_gate/lstm_gate_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*60)
print("Figure saved: lstm_gate_analysis.png")
print("="*60)

# =============================================================================
# 4. Hypothesis Test Summary
# =============================================================================

print("\n" + "="*60)
print("HYPOTHESIS TEST SUMMARY")
print("="*60)

print("\n[Hypothesis 1] Higher layers have lower Input Gate")
print("  L0 input gate: {:.4f}".format(layer_means_i[0]))
print("  L3 input gate: {:.4f}".format(layer_means_i[3]))
if layer_means_i[0] > layer_means_i[3]:
    print("  >> SUPPORTED: L0 > L3 (higher layer more closed)")
else:
    print("  >> REJECTED: L0 <= L3")

print("\n[Hypothesis 2] Higher layers have higher Forget Gate")
print("  L0 forget gate: {:.4f}".format(layer_means_f[0]))
print("  L3 forget gate: {:.4f}".format(layer_means_f[3]))
if layer_means_f[3] > layer_means_f[0]:
    print("  >> SUPPORTED: L3 > L0 (higher layer keeps more)")
else:
    print("  >> REJECTED: L3 <= L0")

print("\n[Interpretation]")
print("If Input gate is LOW and Forget gate is HIGH:")
print("  -> Block new info + Keep old state")
print("  -> LSTM output (R) becomes stable/smooth")
print("  -> Ahat = Conv(R) becomes sparse")
print("  -> But Error = A - Ahat stays continuous (since A keeps changing)")