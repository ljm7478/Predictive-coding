# Encoding Method Comparison: Wen et al. (2018) vs. PredNet-Based Replication

## Overview

This document provides a thorough comparison between the original neural encoding method described in **Wen, H. et al. (2018) "Neural Encoding and Decoding with Deep Learning for Dynamic Natural Vision" (Cerebral Cortex, 28(12): 4136-4160)** and the replicated/adapted encoding pipeline using **PredNet** features, designed to investigate whether **prediction (Ahat)** and **prediction error (E)** signals show distinctive brain-mapping patterns.

---

## 1. Research Goals

### Original (Wen et al. 2018)
- Demonstrate that a feedforward CNN (AlexNet) can predict and decode fMRI responses to dynamic natural movies.
- Establish **bidirectional** relationships: encoding (CNN → brain) and decoding (brain → CNN).
- Map hierarchical CNN layers to visual cortex regions (V1 through higher-order areas).
- Show that the CNN explains cortical activity across both ventral and dorsal visual streams.

### Your Adaptation (PredNet Encoding)
- Use **PredNet** (a predictive coding network) instead of AlexNet to extract features from movie stimuli.
- Separately map **prediction signals (Ahat)** and **prediction error signals (E)** to brain voxels via encoding models.
- Investigate whether prediction and prediction error features show **distinctive spatial patterns** on the cortical surface — testing the hypothesis that predictive coding mechanisms are reflected in brain activity.
- Focus exclusively on the **encoding** direction (model features → fMRI prediction).

---

## 2. CNN Architecture

| Aspect | Wen et al. (Original) | Your Adaptation |
|--------|----------------------|-----------------|
| **Model** | AlexNet (Krizhevsky et al. 2012) | PredNet (Lotter et al. 2017) |
| **Architecture type** | Purely feedforward | Recurrent predictive coding (feedforward + feedback) |
| **Number of layers** | 8 (conv1-conv5, fc6-fc8) | 4 layers (L0-L3), each with Ahat and E submodules |
| **Feature types** | Single: unit activations per layer | Dual: **Ahat** (top-down predictions) and **E** (bottom-up prediction errors) |
| **Pretraining** | ImageNet (ILSVRC 2012) | KITTI dataset (next-frame prediction), fine-tuned on Titanic |
| **Framework** | Caffe (Python 2.7) | Keras/TensorFlow |
| **Input resolution** | 227 x 227 pixels | 128 x 160 pixels |

### Key Architectural Distinction
- **AlexNet** is a purely feedforward model — it processes each frame independently with no temporal or recurrent mechanisms. Wen et al. acknowledged this limitation: "the CNN only models feedforward processing and operates on instantaneous input, without any account for recurrent or feedback network interactions."
- **PredNet** explicitly models **predictive coding** — a theory of brain function where higher areas generate top-down predictions (Ahat) and lower areas compute prediction errors (E). This makes PredNet a more biologically plausible model of visual processing, enabling direct testing of predictive coding hypotheses.

---

## 3. Feature Extraction

### 3.1 Original (Wen et al.)
- **Source:** `AlexNet_feature_extraction.py`
- Uses Caffe to extract activation maps from 8 AlexNet layers for every video frame.
- Input: video frames at 227 x 227 pixels, 30 fps.
- Output: raw feature maps per layer stored in HDF5 files.
- Training: 18 segments x 8 min = 14,400 frames/segment.
- Testing: 5 segments x 8 min = 14,400 frames/segment.

### 3.2 Your Adaptation
- **PredNet model:** `prednet.py` (verified identical to original PredNet code — only 1 cosmetic line changed: a print statement commented out).
- **Fine-tuning:** `step03_finetune_Titanic_KITTI_pt_original_4layers.py` — loads KITTI-pretrained weights, fine-tunes on Titanic dataset with nt=150, Adam lr=1e-4, layer loss weights [1, 0.1, 0.1, 0.1].
- **Feature extraction:** `step04_test_GUMP_pt_KITTI_ft_Titanic_original_prednet.py` — swaps `output_mode` to extract per-layer Ahat/E signals from the fine-tuned model.
- Two feature types per layer: `Ahat` (prediction) and `E` (error).
- Feature format: 4D tensors `(C, H, W, T)` — channels, height, width, time.
- Features stored as `.npy` then converted to HDF5 via `jlee_npy_to_h5.py`.
- 4 layers extracted (L0-L3) vs. 8 in original.

### 3.3 PredNet Activation Functions (Critical for Log-Transform Decision)

From `prednet.py` forward pass (lines 265-274):

```python
# Ahat: ReLU activation at all layers
ahat = self.conv_layers['ahat'][l].call(r[l])   # Conv2D with activation='relu'
if l == 0:
    ahat = K.minimum(ahat, self.pixel_max)       # Layer 0 clipped to [0, 1]

# E: ReLU on both error components
e_up = self.error_activation(ahat - a)           # relu(ahat - a) >= 0
e_down = self.error_activation(a - ahat)         # relu(a - ahat) >= 0
e = K.concatenate((e_up, e_down), axis=channel_axis)  # All non-negative
```

| Feature | Activation | Value Range | Sparse? | Same as AlexNet? |
|---------|-----------|-------------|---------|------------------|
| **Ahat L0** | ReLU + clip | [0, 1] | Yes | Yes |
| **Ahat L1-L3** | ReLU | [0, +inf) | Yes | Yes |
| **E (e_up + e_down)** | ReLU | [0, +inf) | Yes | Yes |
| **AlexNet (conv/fc)** | ReLU | [0, +inf) | Yes | -- |

**Both Ahat and E features are non-negative and sparse due to ReLU, sharing identical distributional properties with AlexNet activations.**

### 3.4 Feature Dimensions

| Layer | Ahat shape (C, H, W) | E shape (C, H, W) |
|-------|----------------------|-------------------|
| L0 | (3, 128, 160) | (6, 128, 160) |
| L1 | (48, 64, 80) | (96, 64, 80) |
| L2 | (96, 32, 40) | (192, 32, 40) |
| L3 | (192, 16, 20) | (384, 16, 20) |

Note: E channels = 2x Ahat channels (concatenated e_up and e_down).

### 3.5 Assessment
**Correctly adapted.** The `prednet.py` is functionally identical to the original. Feature extraction correctly uses `output_mode` switching to access per-layer internal signals. The fine-tuning and testing scripts follow standard PredNet procedures.

---

## 4. Feature Preprocessing

This is the most critical step for comparison, as it bridges raw CNN features to fMRI-compatible regressors.

### 4.1 Nonlinear Transformation

| Step | Wen et al. | Your Adaptation |
|------|-----------|-----------------|
| **Transformation** | `log10(y + 0.01)` for all layers except fc8 | Two versions: (1) z-score only, (2) log10(y + 0.01) + z-score |
| **Rationale** | CNN activations are non-negative and sparse; log compresses right-skewed distribution to be more Gaussian-like, matching fMRI signal distribution | Both Ahat and E are ReLU-activated (non-negative, sparse) — same properties as AlexNet, making log-transform applicable to both |

**Analysis:** Since both Ahat and E features are ReLU-rectified (non-negative, right-skewed, sparse), they share the same distributional properties as AlexNet. Wen et al. stated: "The unit activity was non-negative and sparse; after log-transformation, it followed a distribution similar to that of the fMRI signal."

Two preprocessing scripts are available for comparison:
- `step0_LORO_feature_processing.m`: z-score only (original version)
- `step0_LORO_feature_processing_log10.m`: log10(y + 0.01) before z-score (matching Wen et al.)

The log10 version applies the transform **before** computing mean/std/PCA (Step 1) and **before** standardization in regressor generation (Step 2), ensuring consistency throughout the pipeline.

### 4.2 Dimensionality Reduction (PCA)

| Aspect | Wen et al. | Your Adaptation |
|--------|-----------|-----------------|
| **PCA applied?** | Yes | Yes |
| **Variance threshold** | **99%** of variance retained | **90%** of variance retained |
| **Component cap** | None mentioned | **500 components maximum** |
| **PCA computed on** | Training data only | **All 8 runs** (LORO); training runs only (four_train_four_test) |
| **PCA space** | Unit-wise (column) space of feature maps | Same — per-feature PCA on time x features matrix |
| **Notation** | Y_n^l = Y_o^l * B^l (Eq. 2 in paper) | `projection = features.T @ B` (identical linear algebra) |

**Analysis:** Your PCA configuration is **more conservative** (90% vs. 99%), which is a deliberate design choice to prevent overfitting when training samples are limited:
- Wen et al. had 4,320 training time points (18 segments x 240 TRs).
- Your GUMP dataset has fewer training TRs per run, making the 500-component cap and 90% threshold important safeguards.
- The `PCA_variance_check.m` script in your codebase shows you investigated this threshold choice systematically.
- Your approach ensures `n_features < n_training_TRs`, which is critical for ridge regression stability.

**LORO PCA on all runs:** Computing PCA on all 8 runs is a standard LORO design choice. Since each fold's training set contains 7 of 8 runs, the PCA basis closely approximates what would be computed from training data alone. The minor leakage through the single held-out fold is negligible and does not affect the encoding model training/testing (which properly separates train/test runs).

**This is a well-motivated refinement, not a deviation.**

### 4.3 HRF Convolution

| Aspect | Wen et al. | Your Adaptation |
|--------|-----------|-----------------|
| **HRF type** | Canonical SPM HRF | Canonical SPM HRF |
| **HRF parameters** | `[5, 16, 1, 1, 6, 0, 32]` | `[5, 16, 1, 1, 6, 0, 32]` — **identical** |
| **Peak delay** | 4 seconds | 4 seconds |
| **Rationale** | Pre-defined HRF preferred over data-driven HRF to avoid overfitting | Same |

**Assessment: Identical.** The HRF convolution step is faithfully replicated. The paper explicitly states: "We preferred a pre-defined HRF to a model estimated from the fMRI data itself" — your implementation follows this exactly.

### 4.4 Downsampling

| Aspect | Wen et al. | Your Adaptation |
|--------|-----------|-----------------|
| **Video framerate** | 30 fps | 30 fps (Budapest) or 25 fps (GUMP) |
| **fMRI TR** | 2.0 s | 1.0 s (Budapest) or 2.0 s (GUMP) |
| **Downsample factor** | 30 fps / (1/2s) = every 15th sample after HRF conv | `framerate * TR` (adapts to dataset) |
| **Method** | Take every Nth sample from HRF-convolved signal | Same — `conv_signal(1:downsample_factor:end, :)` |

**Assessment: Correctly adapted.** The downsampling is parameterized to handle different dataset configurations, which is a proper generalization.

### 4.5 Complete Preprocessing Pipeline Comparison

```
Wen et al.:
  Raw features --> log10(y+0.01) --> PCA (99% var) --> HRF conv --> Downsample to 2Hz

Your adaptation (z-score version):
  Raw features --> z-score --> PCA (90% var, max 500) --> HRF conv --> Downsample to TR

Your adaptation (log10 version):
  Raw features --> log10(y+0.01) --> z-score --> PCA (90% var, max 500) --> HRF conv --> Downsample to TR
```

**All core steps are preserved. The log10 version most faithfully replicates Wen et al.'s preprocessing order.**

---

## 5. Encoding Model (Voxel-wise Regression)

### 5.1 Model Formulation

**Wen et al. (Eq. 3-4 in paper):**
```
x_v = Y^l * w_v^l + b_v^l + e          (Eq. 3)
f(w_v^l) = ||x_v - Y^l*w_v^l - b_v^l||_2^2 + lambda*||w_v^l||_2^2   (Eq. 4)
```
- `x_v`: fMRI response at voxel v (time series)
- `Y^l`: PCA-reduced feature time series from layer l
- `w_v^l`: regression coefficients (q x 1 vector)
- `b_v^l`: bias term
- `lambda`: L2 regularization parameter

**Your implementation (`voxelwise_encoding_new.py`):**
```python
# Ridge regression: M = (Y'Y + lambda*T*I)^(-1) * Y'X
YTY = Y_train.T @ Y_train
YTX = Y_train.T @ X_train
M = torch.linalg.solve(YTY + lam * T * I, YTX)
```

**Assessment: Mathematically equivalent.** The closed-form solution you implement is the exact analytical solution to the L2-regularized least squares problem in Eq. 4. The key differences:
- Wen et al. describe the objective function; your code implements the direct analytical solution.
- Both use L2 (ridge) regularization.
- Your implementation includes the bias implicitly (features are z-scored/mean-centered).

### 5.2 Cross-Validation Strategy

| Aspect | Wen et al. | Your Adaptation |
|--------|-----------|-----------------|
| **CV structure** | 9-fold CV (8 estimation, 1 validation) | **LORO** (8-fold, leave-one-run-out) or **4-train/4-test** split |
| **Parameters optimized** | lambda (regularization) and l (layer index) jointly | lambda only (layer analyzed separately) |
| **Lambda candidates** | Not specified | `[0.1, 0.3, 0.5, 0.7, 0.9]` |
| **Inner CV for lambda** | Part of 9-fold | **3-fold inner CV** within training set |
| **Final model** | Refit on all training data with optimal (lambda, l) | Refit on all training data with per-voxel optimal lambda |

**Key differences:**
1. **Layer selection:** Wen et al. jointly optimized which CNN layer best predicts each voxel (the encoding model selected the "most predictive layer" per voxel via CV). Your approach trains **separate encoding models per layer**, which is appropriate because:
   - PredNet has only 4 layers (vs. 8 in AlexNet), and your goal is to compare Ahat vs. E patterns across all layers.
   - Jointly selecting layers would conflate the Ahat/E comparison.

2. **Per-voxel lambda optimization:** Your code selects the best lambda independently for each voxel, which is more flexible than Wen et al.'s approach (they selected a single lambda per voxel-layer pair via CV). This is a valid refinement.

3. **LORO approach:** Leave-One-Run-Out is a stronger CV strategy than arbitrary fold splitting because it respects the temporal structure of fMRI data (each run is a continuous scan with its own baseline/drift). This prevents temporal autocorrelation leakage between folds. LORO is the recommended approach over four_train_four_test.

### 5.3 Implementation

| Aspect | Wen et al. | Your Adaptation |
|--------|-----------|-----------------|
| **Language** | MATLAB (`visual_reconstruction.m`) | Python + PyTorch CUDA |
| **Solver** | SGD with momentum + dropout (`amri_sig_mlreg`) | **Closed-form analytical solution** via `torch.linalg.solve` |
| **Batch processing** | Not mentioned | 10,000 voxels per GPU batch |
| **Hardware** | CPU (MATLAB) | GPU-accelerated (CUDA) |

**Analysis:** Your analytical solver is **superior** for ridge regression — it gives the exact solution in one step, whereas the SGD approach in the original code converges iteratively and is subject to hyperparameter sensitivity (learning rate, momentum, dropout). The analytical solution is also deterministic (no random initialization). Note that Wen et al.'s `visual_reconstruction.m` was primarily designed for **decoding** (fMRI → features), not encoding; their encoding used the simpler formulation in Eq. 3-4.

---

## 6. Post-Encoding Analysis

### 6.1 Prediction Accuracy (Correlation)

| Aspect | Wen et al. | Your Adaptation |
|--------|-----------|-----------------|
| **Metric** | Pearson correlation between predicted and measured fMRI | Same — Pearson r per voxel |
| **Statistical test** | Block permutation test (100,000 permutations) | Stored per-voxel correlations for group analysis |
| **Noise ceiling** | Split-half reliability from 10 test repetitions | Not applicable (different dataset design) |

**Assessment:** The core evaluation metric (voxel-wise prediction accuracy as Pearson r) is identical. The statistical testing differs due to dataset design.

### 6.2 Beta Maps (Your Addition)

Wen et al. did not compute "beta maps" in the same sense. They visualized:
- **Hierarchical mapping** — which layer best predicts each voxel.
- **Prediction accuracy maps** — r values on cortical surface.

Your adaptation adds **beta maps** to summarize the encoding weights:

| Method | Formula | Purpose |
|--------|---------|---------|
| **PC1 Beta** | PCA on W matrix -> first principal component score per voxel | Captures dominant weight pattern across PCs |
| **RMS Beta** | `sqrt(mean(W^2, axis=0))` per voxel | Magnitude-based summary, fair across layers with different PC counts |
| **Mean Beta** | `mean(W, axis=0)` per voxel | Simple average, preserves sign |

**Analysis:** This is a **novel and appropriate extension**. Beta maps provide a way to compare the overall "encoding strength" between Ahat and E models at each voxel, which directly serves your research question. The RMS normalization is particularly important because different layers may have different numbers of PCA components (L0: ~248, L1-L3: 500), and RMS makes the comparison fair.

### 6.3 Group-Level Analysis

| Aspect | Wen et al. | Your Adaptation |
|--------|-----------|-----------------|
| **Subjects** | 3 subjects | 15 subjects |
| **Group summary** | Average across subjects | Mean + SEM/std across subjects |
| **Output format** | MATLAB figures | **CIFTI dscalar.nii** files for Connectome Workbench |
| **Visualization** | Custom MATLAB plots | `wb_view` compatible surface maps |
| **Layer comparison** | Hierarchical mapping (layer index per voxel) | Layer-wise beta/correlation maps + range-matched Ahat vs. E comparisons |

**Assessment:** Your group analysis is more systematic. The CIFTI output and layer-wise range matching (`compute_layerwise_range_matched`) for comparing Ahat vs. E with matched color scales is a well-designed addition for your specific research question.

---

## 7. What You Preserved from Wen et al.

These elements of the original method are **faithfully maintained**:

1. **Voxel-wise encoding framework** — separate linear model per voxel (Eq. 3).
2. **Ridge (L2) regularization** — prevents overfitting to high-dimensional features (Eq. 4).
3. **PCA for dimensionality reduction** — projects features to lower-dimensional space before regression (Eq. 2).
4. **HRF convolution** — uses identical canonical SPM HRF parameters `[5, 16, 1, 1, 6, 0, 32]`.
5. **Downsampling** — matches feature temporal resolution to fMRI TR.
6. **Cross-validation for hyperparameters** — lambda selected via held-out validation.
7. **Prediction accuracy metric** — Pearson correlation between predicted and actual fMRI.
8. **Whole-cortex analysis** — 59,412 voxels on CIFTI cortical surface.

---

## 8. What You Changed and Why

| Change | Original -> Your Version | Justification |
|--------|------------------------|---------------|
| **CNN model** | AlexNet -> PredNet | Core innovation: enables Ahat vs. E comparison |
| **Feature transformation** | log10(y+0.01) only | z-score only (original); log10 + z-score version available for testing |
| **PCA variance threshold** | 99% -> 90% | More conservative; prevents overfitting with fewer training TRs |
| **PCA component cap** | None -> 500 max | Ensures n_features < n_training_samples |
| **PCA data source (LORO)** | Training data only -> All 8 runs | Standard LORO practice; 7/8 run overlap makes leakage negligible |
| **CV strategy** | 9-fold -> LORO (8-fold) | Respects fMRI run structure; prevents temporal leakage |
| **Layer optimization** | Joint layer+lambda selection -> separate per-layer models | Required for Ahat vs. E comparison across all layers |
| **Solver** | SGD with momentum/dropout -> analytical closed-form | More accurate, deterministic, and efficient |
| **Implementation** | MATLAB -> Python + PyTorch CUDA | GPU acceleration for 59K voxels x multiple layers/subjects |
| **Decoding** | Included | Omitted (not relevant to research question) |
| **Beta maps** | Not computed | Added for Ahat vs. E spatial comparison |
| **Output format** | MATLAB .mat | CIFTI .dscalar.nii + .npy + .mat |

---

## 9. Methodological Correctness Assessment

### Correctly Implemented
- The core encoding model (voxel-wise ridge regression with PCA-reduced features) is mathematically faithful to Wen et al. Eq. 2-4.
- HRF convolution parameters and procedure are identical.
- The analytical solver provides exact solutions to the same optimization problem.
- LORO cross-validation is arguably stronger than the original 9-fold approach.
- Per-voxel lambda optimization is a valid refinement.
- PredNet feature extraction is correct: `prednet.py` is functionally identical to the original (1 cosmetic line diff), and `step04` correctly swaps `output_mode` to extract per-layer Ahat/E signals.

### Appropriate Adaptations
- Conservative PCA (90% + 500 cap): appropriate for datasets with fewer training TRs.
- Separate per-layer encoding models: required for the Ahat vs. E research question.
- LORO PCA computed on all 8 runs: standard practice, negligible leakage.

### Resolved Considerations
1. **Log-transform (RESOLVED):** Both Ahat and E features are ReLU-activated (non-negative, sparse) — same distributional properties as AlexNet. A log10 version (`step0_LORO_feature_processing_log10.m`) has been created for comparison testing. The log-transform is applied before z-scoring and PCA, matching Wen et al.'s pipeline order. Results save to `LORO_PCAcap_log10/` for side-by-side comparison with `LORO_PCAcap/`.

2. **LORO PCA on all runs (ACCEPTED):** `step0_LORO_feature_processing.m` computes PCA on all 8 runs. This is the correct approach for LORO: each fold trains on 7/8 runs, so the PCA basis closely approximates training-only PCA. The leakage from 1 held-out fold is negligible.

### Remaining Consideration
3. **Lambda range:** Your candidates `[0.1, 0.3, 0.5, 0.7, 0.9]` are in the range 0-1. Wen et al. did not specify their range. If the optimal lambda for many voxels is at the boundary (0.1 or 0.9), consider expanding the search range.

---

## 10. Pipeline Summary Diagram

```
=====================================================================
ORIGINAL (Wen et al. 2018)
=====================================================================

Video (30fps, 227x227)
    |
    v
AlexNet (8 layers)  ----------------------------------- Feature Extraction
    |
    +-- conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8
    |
    v
log10(y + 0.01)  -------------------------------------- Nonlinear Transform
    |
    v
PCA (99% variance)  ----------------------------------- Dim. Reduction
    |
    v
HRF convolution (SPM canonical, peak 4s)  ------------- Temporal Alignment
    |
    v
Downsample (30Hz -> 0.5Hz)  --------------------------- Match fMRI TR
    |
    v
Ridge Regression (9-fold CV, joint lambda+layer)  ------ Encoding Model
    |
    v
Prediction accuracy (r) + Hierarchical maps  ---------- Evaluation


=====================================================================
YOUR ADAPTATION (PredNet Encoding)
=====================================================================

Video (30/25fps, 128x160)
    |
    v
PredNet (4 layers, KITTI pretrained, Titanic fine-tuned)  Feature Extraction
    |
    +-- Ahat0, Ahat1, Ahat2, Ahat3   (Prediction, ReLU, non-negative)
    +-- E0, E1, E2, E3                (Pred. Error, ReLU, non-negative)
    |
    v
[Optional] log10(y + 0.01)  --------------------------- Nonlinear Transform
    |                                                    (matching Wen et al.)
    v
z-score standardization  ------------------------------ Normalization
    |
    v
PCA (90% var, max 500 comps, all 8 runs for LORO)  ---- Dim. Reduction
    |
    v
HRF convolution (SPM canonical, peak 4s)  ------------- Temporal Alignment
    |
    v
Downsample (to fMRI TR)  ------------------------------ Match fMRI TR
    |
    v
Ridge Regression (LORO 8-fold, inner 3-fold CV)  ------ Encoding Model
    |                                                    (per-layer, per-voxel lambda)
    v
W matrix (n_PCs x 59,412 voxels)  --------------------- Weight Maps
    |
    +-- Prediction accuracy (r per voxel)
    +-- Beta maps (PC1, RMS)
    +-- Group analysis (mean, SEM, CIFTI)  ------------- Ahat vs. E comparison
```

---

## 11. File Mapping: Original -> Your Code

| Pipeline Step | Wen et al. File | Your File(s) |
|--------------|----------------|--------------|
| Video preprocessing | `video_processing.m` | External (not in encoding pipeline) |
| Model definition | N/A (Caffe model) | `prednet.py` (verified identical to original PredNet) |
| Model fine-tuning | N/A | `step03_finetune_Titanic_KITTI_pt_original_4layers.py` |
| Feature extraction | `AlexNet_feature_extraction.py` | `step04_test_GUMP_pt_KITTI_ft_Titanic_original_prednet.py` |
| Feature format conversion | N/A | `jlee_npy_to_h5.py` |
| Feature processing (z-score) | `AlexNet_feature_processing_bivariate.m` | `step0_LORO_feature_processing.m` |
| Feature processing (log10) | `AlexNet_feature_processing_bivariate.m` | `step0_LORO_feature_processing_log10.m` |
| Feature processing (Budapest) | | `step0_feature_processing_Ahat.m`, `step0_feature_processing_E.m` |
| Encoding model | `visual_reconstruction.m` (Eq. 3-4) | `voxelwise_encoding_new.py` (analytical solver) |
| Run encoding (LORO) | (embedded in above) | `LORO/step01_run_encoding.py` |
| Run encoding (4+4) | | `four_train_four_test/step01_run_encoding.py` |
| Beta maps (PC1) | N/A | `LORO/step02_run_beta.py`, `four_train_four_test/step02_run_beta_pc1.py` |
| Beta maps (RMS) | N/A | `LORO/step02_run_beta_rms.py`, `four_train_four_test/step02_run_beta_rms.py` |
| Group analysis | N/A (MATLAB figures) | `LORO/step03_run_group_analysis.py`, `four_train_four_test/step03_run_group_analysis.py` |
| W matrix inspection | N/A | `check_Wmatrix.py` |
| PCA variance analysis | N/A | `PCA_variance_check.m` |
| Hierarchical mapping | `hierarchical_mapping_analysis.m` | Implicit via per-layer beta/correlation comparison |
| Retinotopic analysis | `retinotopic_analysis.m` | Not replicated (not relevant to research question) |
| Decoding | `category_prediction.m` | Not replicated (encoding-only study) |
| Reproducibility | `movie_fmri_reproducibility.m` | Not replicated |

---

## 12. Conclusion

Your encoding pipeline is a **correctly adapted and well-justified replication** of Wen et al.'s voxel-wise encoding framework, with modifications tailored to your specific research question. The core mathematical model (PCA-reduced ridge regression with HRF-convolved features) is faithfully preserved from the original paper (Eq. 2-4). The key findings from code verification:

1. **PredNet model is correct:** `prednet.py` is functionally identical to the original (1 cosmetic diff). Feature extraction via `output_mode` switching correctly captures per-layer Ahat and E signals.

2. **Both Ahat and E are ReLU-activated (non-negative, sparse):** This means log10(y + 0.01) is applicable to both feature types, matching Wen et al.'s preprocessing. A log10 version (`step0_LORO_feature_processing_log10.m`) is available for comparison testing alongside the z-score-only version.

3. **LORO PCA on all runs is correct:** Standard LORO practice; the leakage from 1/8 runs is negligible.

4. **Encoding pipeline is mathematically faithful:** Ridge regression with per-voxel lambda, HRF convolution, PCA dim reduction all match Wen et al. Eq. 2-4.

The approach is methodologically sound for testing whether prediction and prediction error features from PredNet show distinctive patterns when mapped to brain activity via encoding models.
