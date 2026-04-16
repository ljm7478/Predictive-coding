# LORO_PCA99_cap1500 — Parallel Pipeline with Higher PCA Cap

## Why this folder exists

The original `LORO/` pipeline uses `PCA_VARIANCE_THRESHOLD = 0.90` and `MAX_PCA_COMPONENTS = 500`, which was found to lose substantial variance for several PredNet feature layers. Specifically, running `encoding/PCA_variance_check.m` on the log10-transformed features yielded:

| Layer | n_raw_features | Ahat @ 500 PCs | Ahat 90% needs | E @ 500 PCs | E 90% needs |
|-------|----------------|----------------|----------------|-------------|-------------|
| L1    | 61,440         | **95.6%** ✓   | 173            | **65.0%** ✗ | 4166        |
| L2    | 245,760        | **43.2%** ✗✗  | 5411           | **41.4%** ✗✗| 7649        |
| L3    | 122,880        | **67.0%** ✗   | 1649           | **48.2%** ✗ | 5292        |
| L4    | 61,440         | **96.6%** ✓   | 372            | **58.9%** ✗ | 2719        |

Two patterns: (a) Ahat L2 is severely under-capped at only 43% variance retained, and (b) all E layers are systematically under-captured (E is a residual / high-frequency signal and is intrinsically higher-dimensional than the smooth Ahat prediction).

## What this pipeline changes

Exactly two preprocessing parameters change relative to `LORO/`:

```matlab
PCA_VARIANCE_THRESHOLD = 0.99;   % was 0.90
MAX_PCA_COMPONENTS     = 1500;   % was 500
```

And the output directory:

```matlab
SAVE_ROOT = '.../LORO_PCA99cap1500_log10/layer4/<feature_type>'   % was LORO_PCAcap_log10
```

The PCA computation logic in `step0_LORO_feature_processing_log10.m` already supports both parameters via its `compute_pca(data, variance_threshold, max_components)` helper, so the change is purely a parameter update — no algorithmic change.

## Why 99% threshold and 1500 cap

- **99% variance**: matches the de facto convention in CNN-encoding papers (Wen 2018, Güçlü & van Gerven 2015). The reasoning is that low-variance PCs may still carry signal relevant to brain encoding, and ridge regularization will downweight noise PCs anyway.
- **Cap at 1500**: keeps the PC-to-train-TR ratio safely above 2:1 in the worst LORO fold. The 8 Forrest Gump runs have TR counts (451, 441, 438, 488, 462, 439, 542, 338) summing to **3599 TRs total**. The LORO worst case (leaving out run7 = 542 TRs) yields a minimum training set of **3057 TRs**, so 3057 / 1500 = 2.04 — well within ridge regression's safe operating regime. Going higher would risk ill-conditioning of the closed-form ridge solution.
- **min(99%_count, 1500)**: layers whose 99% variance is reached well below 1500 (e.g., Ahat L4 needs only 623 PCs) keep their natural lower count. Layers that would otherwise blow past 1500 (e.g., L2 needs ~12,000 PCs for 99%) get hard-capped at 1500 for tractability.

Expected variance retention with the new cap:

| Layer | Old (cap=500) | New (cap=1500 @ 99%) |
|-------|---------------|----------------------|
| Ahat L1 | 95.6% | 96–97% (1500 not needed; 99% reached at 1807, capped at 1500) |
| Ahat L2 | **43.2%** | ~70–75% (still capped, but +30% recovered) |
| Ahat L3 | 67.0% | ~92–95% |
| Ahat L4 | 96.6% | 99% (623 PCs, well under cap) |
| E L1 | 65.0% | ~80% (still capped) |
| E L2 | 41.4% | ~65% (still capped) |
| E L3 | 48.2% | ~70% (still capped) |
| E L4 | 58.9% | ~80% (still capped) |

E layers will still be partially under-captured because their intrinsic dimensionality is too high for any practical PC count. The proper long-term fix for E is **banded ridge regression with raw features** (himalaya library), where the regularization handles dimensionality directly. That would be a separate refactor.

## What stays identical

- LORO 8-fold cross-validation (leave-one-run-out)
- log10 transform on raw features (`log10(x + 0.01)`)
- HRF convolution + downsample to TR (Wen et al. 2018 convention)
- Train-fold demeaning of regressors and fMRI
- Ridge regression with cross-validated lambda (`[0.1, 0.3, 0.5, 0.7, 0.9]`, inner 3-fold CV)
- All 15 Forrest Gump subjects
- Variance partitioning method (3 ridge models, sign-preserving R², preference index with clipped uniques)
- 2D bivariate preference figures (Popham 2021 style)

## Plain ridge vs banded ridge

`step01_run_encoding.py` and the Ahat-only / E-only models in `step04_variance_partitioning.py` stay as **plain ridge regression**. There is only one feature group in those fits — banded ridge would not apply.

The **joint model** in `step04` (which concatenates [Ahat | E]) currently uses a single shared lambda for the concatenated feature space. Upgrading it to **banded ridge** with separate lambdas per feature group (Nunez-Elizalde et al. 2019; `himalaya` library) would be the rigorous Gallant-lab convention for joint encoding models with heterogeneous feature spaces. This is left as a future upgrade and is not implemented here.

## Run order

1. **MATLAB**: `step0_LORO_feature_processing_log10.m`  (twice — once with `FEATURE_TYPE='Ahat'`, once with `'E'`)  → produces regressors h5 files
2. **Python**: `step01_run_encoding.py`  (once with `FEATURE_TYPE="Ahat"`, once with `"E"`) → ridge encoding correlation maps
3. **Python**: `step04_variance_partitioning.py` → variance partitioning maps
4. **Python**: `step05_plot_2d_preference.py [--group]` → 2D bivariate figures

Outputs land under:
```
.../kitti_pretrain_result/LORO_PCA99cap1500_log10/layer4/
    Ahat/<subject>/...
    E/<subject>/...
    variance_partitioning_LORO/<subject>/...
    variance_partitioning_LORO/figures_2d_preference/
```
which is parallel to and does not overwrite the existing `LORO_PCAcap_log10/` outputs.
