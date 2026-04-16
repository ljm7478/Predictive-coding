# Predictive Coding Encoding Project: Analysis Plan

## Project Title
Prediction and Prediction Error Signals Show Distinct Cortical Organization During Naturalistic Vision: Evidence from PredNet-Based Encoding Models

---

## Core Hypothesis

Prediction (Ahat) and prediction error (E) signals from PredNet map onto spatially dissociable cortical regions with distinct organizational principles, providing macroscale evidence for predictive coding during naturalistic visual processing.

---

## Four Research Questions

1. **Spatial dissociation:** Do prediction and prediction-error signals occupy spatially distinct cortical regions?
2. **Hierarchical organization:** How do Ahat and E organize across the cortical visual hierarchy (V1 through higher-order areas)?
3. **Error propagation:** Do prediction errors propagate hierarchically (canonical predictive coding) or reflect local computation?
4. **Temporal horizon:** Does the temporal scope of prediction relate to cortical hierarchical position?

---

## Analysis Architecture

The plan is organized into tiers. Tier 1 is what you already have. Tiers 2-4 build progressively deeper analyses.

```
Tier 1: Current Analyses (DONE)
  |-- Beta maps (PC1, RMS)
  |-- Correlation maps (encoding performance)
  |-- Group-level statistics

Tier 2: Spatial Dissociation (Q1)
  |-- Variance partitioning (Ahat vs E unique/shared)
  |-- 2D preference maps (Ahat vs E dominance)
  |-- Winner-take-all layer assignment

Tier 3: Hierarchical Organization (Q2 + Q3)
  |-- ROI-based encoding profiles across layers
  |-- Best-layer gradient maps
  |-- Error-propagation vs representation-propagation model comparison

Tier 4: Temporal & Cross-Dataset Validation (Q4)
  |-- Temporal context manipulation
  |-- Cross-dataset replication (GUMP + Budapest)
  |-- VAE model comparison
```

---

## Tier 1: Current Analyses (Complete)

### 1.1 Encoding Performance Maps
- **What:** Pearson r between predicted and measured fMRI per voxel, per layer, per feature type (Ahat, E).
- **Output:** `average_correlations_LORO.npy` per subject; group mean correlation dscalar maps.
- **Status:** Done.

### 1.2 Beta Maps
- **What:** Summary of encoding weight matrices (W) into single-value-per-voxel maps.
- **Methods:** PC1 (dominant covariance pattern) and RMS (magnitude, fair across layers).
- **Output:** `beta_pc1_all_layers.npy`, `beta_rms_all_layers.npy` per subject; group dscalar maps.
- **Status:** Done.

### 1.3 Group Analysis
- **What:** Mean and SEM across subjects for correlations and betas. CIFTI output for wb_view.
- **Status:** Done.

---

## Tier 2: Spatial Dissociation of Prediction vs Error (Q1)

### 2.1 Variance Partitioning (Ahat vs E)

**Rationale:** Beta maps and correlation maps show each feature type independently. Variance partitioning reveals whether Ahat and E explain the *same* or *different* aspects of fMRI activity.

**Method (following Lescroart & Gallant 2019; Huth et al. 2016):**
1. For each voxel and each layer l, fit three ridge regression models:
   - **Model_Ahat:** fMRI ~ Ahat features only
   - **Model_E:** fMRI ~ E features only
   - **Model_Joint:** fMRI ~ [Ahat, E] features concatenated
2. Compute R-squared for each model on held-out test data:
   - R2_Ahat, R2_E, R2_Joint
3. Decompose into unique and shared variance:
   - **Unique_Ahat** = R2_Joint - R2_E (variance only Ahat explains)
   - **Unique_E** = R2_Joint - R2_Ahat (variance only E explains)
   - **Shared** = R2_Ahat + R2_E - R2_Joint (variance both explain)
4. Compute a **preference index** per voxel:
   - PI = (Unique_Ahat - Unique_E) / (Unique_Ahat + Unique_E)
   - PI = +1 → purely prediction-driven; PI = -1 → purely error-driven

**Implementation notes:**
- Use banded ridge regression (separate lambda for Ahat and E feature sets in the joint model) to prevent one feature set from dominating. See Nunez-Elizalde et al. 2019 for the method, and the `himalaya` Python package.
- Alternative: if banded ridge is complex to implement, use standard ridge on each model separately. The unique variance computation is still valid.
- Run per layer and also collapsed across layers (concatenate all 4 layers).

**Output:**
- Unique_Ahat, Unique_E, Shared maps (dscalar.nii) per layer and across layers.
- Preference index maps per layer and across layers.

**Key prediction:** If predictive coding holds, Ahat and E should have large unique variance components (they explain different things), not just shared variance.

### 2.2 2D Preference Maps (Inspired by Popham et al. 2021)

**Rationale:** Popham et al. used 2D colormaps to show where visual vs linguistic representations dominate. Apply the same visualization to Ahat vs E.

**Method:**
1. For each voxel, compute normalized encoding performance:
   - r_Ahat = correlation from Ahat-only model (z-scored across voxels)
   - r_E = correlation from E-only model (z-scored across voxels)
2. Create a 2D colormap:
   - X-axis: r_Ahat strength
   - Y-axis: r_E strength
   - Color: Red = Ahat dominant, Blue = E dominant, Purple = both, White = neither
3. Map onto CIFTI surface.

**Output:** 2D-colored cortical surface maps showing the spatial distribution of Ahat vs E encoding.

### 2.3 Winner-Take-All Assignment

**Rationale:** For each voxel, determine which feature type (Ahat or E) and which layer (L0-L3) provides the best encoding.

**Method:**
1. For each voxel, find: `best_type, best_layer = argmax(r[type, layer])` across all 8 combinations (4 layers x 2 types).
2. Create two maps:
   - **Best-type map:** Ahat vs E winner per voxel.
   - **Best-layer map:** L0/L1/L2/L3 winner per voxel (separately for Ahat and E).

**Output:** Winner maps (dscalar.nii), colored by type and layer.

---

## Tier 3: Hierarchical Organization and Error Propagation (Q2 + Q3)

### 3.1 ROI-Based Encoding Profiles

**Rationale:** Wen et al. showed that different CNN layers contribute differently to prediction accuracy at each ROI (their Fig. 3c). Replicate this for PredNet Ahat and E.

**Method:**
1. Define ROIs from Glasser parcellation or HCP multimodal atlas:
   - Early visual: V1, V2, V3, V4
   - Ventral stream: LO, FFA, PPA
   - Dorsal stream: MT, V3A, LIP
   - Higher-order: FEF, PEF, TPJ, STS
   - Default mode: precuneus, mPFC, angular gyrus
2. For each ROI, compute mean encoding performance across voxels, separately for each layer (L0-L3) and each feature type (Ahat, E).
3. Plot **layer-preference curves** per ROI (x-axis: layer, y-axis: mean r), with separate lines for Ahat and E.

**Expected patterns under predictive coding:**
- **Ahat:** Should show gradient from low layers in early visual cortex to high layers in higher-order areas (predictions flow top-down but are generated at each level).
- **E:** If errors propagate hierarchically, E should also show a layer gradient. If errors are computed locally, E should not show a clear layer gradient — each area uses local error regardless of layer.

**Output:**
- Layer-preference curve plots per ROI (Ahat vs E overlaid).
- Table of best-layer per ROI per feature type.

### 3.2 Cortical Gradient Maps (Best-Layer)

**Rationale:** Wen et al. (2018, Fig. 2c) and Guclu & van Gerven (2015) showed that best CNN layer maps onto cortical hierarchy. Do the same for PredNet Ahat and E.

**Method:**
1. For each voxel, find the best layer: `l* = argmax_l(r[l])`.
2. Create separate best-layer maps for Ahat and E.
3. Compute a continuous gradient using weighted average:
   - `layer_index = sum(l * r[l]) / sum(r[l])` (correlation-weighted mean layer)
4. Compare the Ahat gradient to the E gradient using spatial correlation across voxels.

**Key prediction:** If Ahat and E have different hierarchical organizations, their best-layer maps should differ systematically.

**Output:**
- Best-layer maps (discrete and continuous) for Ahat and E (dscalar.nii).
- Spatial correlation between Ahat and E gradient maps.

### 3.3 Error Propagation vs Local Computation (Q3 — Critical Test)

**Rationale:** This is the defining test of canonical predictive coding. The user's introduction frames it precisely: "Do prediction errors propagate hierarchically, or are they computed locally?"

**Method — Model Architecture Comparison:**

PredNet has two architectural variants that directly test this:

**A. Error-Propagation Model (original PredNet = canonical predictive coding):**
- E signals ascend the hierarchy: E_l feeds into A_{l+1} (target for the next layer).
- The forward pathway IS the error signal.
- Higher layers receive error from lower layers.
- `a_{l+1} = Conv(e_l) → MaxPool` — error propagates upward.

**B. Representation-Propagation Model (CNN-like variant):**
- Raw representations (A, not E) ascend the hierarchy.
- Each layer computes error locally between its own prediction and its own target.
- Errors do NOT propagate between layers.
- This is NOT canonical predictive coding — it's hierarchical processing with local error.

**Analysis:**
1. Extract features from BOTH model variants (you already have both in `prednet_original/` and `CNN_like/` directories).
2. Run the same encoding pipeline for both variants.
3. Compare encoding performance:
   - If the error-propagation model produces better encoding (especially for E features at higher layers), it supports canonical predictive coding.
   - If both models perform similarly, hierarchical error propagation is not necessary to explain cortical dynamics.
4. Critical comparison for E features specifically:
   - In the error-propagation model, E_l at layer l>0 reflects accumulated errors from lower layers.
   - In the representation-propagation model, E_l is purely local.
   - If the brain's error signals are better captured by the error-propagation E, this supports hierarchical error propagation.

**Output:**
- Side-by-side encoding performance maps for both model architectures.
- Layer-by-layer ROI comparison for both architectures.
- Statistical test: paired t-test across subjects for encoding performance difference per ROI.

### 3.4 Partial Correlation Analysis (Hierarchical Error Contribution)

**Rationale:** If errors propagate hierarchically, E at layer l+1 should contain information from E at layer l (because lower-layer error feeds into higher-layer computation). Test this with partial correlations.

**Method:**
1. For each voxel, compute encoding using E features from all layers.
2. Compute partial correlation: the unique contribution of E_{l+1} after controlling for E_l.
   - If hierarchical propagation: partial correlation of E_{l+1} should be significant even after removing E_l's contribution.
   - If local computation: E_{l+1} and E_l should explain largely independent variance.
3. Alternatively, use sequential model building:
   - Model 1: fMRI ~ E_0
   - Model 2: fMRI ~ E_0 + E_1
   - Model 3: fMRI ~ E_0 + E_1 + E_2
   - Model 4: fMRI ~ E_0 + E_1 + E_2 + E_3
   - The incremental R2 at each step shows how much new information each layer's error adds.

**Output:**
- Incremental R2 maps per layer.
- Partial correlation maps.

---

## Tier 4: Temporal Horizon and Cross-Validation (Q4)

### 4.1 Temporal Context Manipulation

**Rationale:** Caucheteux et al. showed that different brain regions predict at different temporal horizons (forecast distance). PredNet's recurrent architecture naturally captures temporal dynamics. Test whether varying the temporal context changes which brain regions are best predicted.

**Method:**
1. Run PredNet with different sequence lengths (nt):
   - Short context: nt = 10, 25, 50 frames
   - Medium context: nt = 100, 150 frames (current)
   - Long context: nt = 200, 300 frames (if computationally feasible)
2. Extract Ahat and E features at each temporal context length.
3. Run encoding pipeline for each.
4. Compute the **optimal temporal context** per voxel: `nt* = argmax_{nt}(r[nt])`.
5. Map nt* onto cortical surface.

**Expected pattern:** Early visual areas may prefer shorter contexts (immediate predictions), while higher-order areas may prefer longer contexts (extended temporal integration).

**Output:**
- Encoding performance as function of temporal context per ROI.
- Optimal temporal context maps (dscalar.nii).

### 4.2 Cross-Dataset Replication

**Rationale:** Your findings should replicate across independent datasets to be convincing.

**Method:**
1. Run Tiers 2-3 analyses on both GUMP and Budapest datasets independently.
2. Compare:
   - Spatial correlation of Ahat vs E preference maps across datasets.
   - Consistency of best-layer gradients.
   - Consistency of ROI-level encoding profiles.

**Output:**
- Cross-dataset spatial correlation statistics.
- Side-by-side comparison figures.

### 4.3 VAE Model Comparison (Extended)

**Rationale:** Your introduction mentions testing with additional computational models (clockwise VAE, hierarchical VAE). This controls for model-specific confounds.

**Method:**
1. Train a hierarchical VAE on the same video data.
2. Extract latent representations at each hierarchical level.
3. Split into "prediction-like" (generative/decoder pathway) and "error-like" (encoder/inference pathway) signals.
4. Run encoding pipeline with VAE features.
5. Compare spatial organization of VAE prediction/error with PredNet Ahat/E.

**Purpose:** If the same cortical organization emerges across different model architectures, the finding is not model-specific but reflects a general computational principle.

---

## Statistical Framework

### Within-Subject Statistics
- All encoding models use **LORO cross-validation** (8-fold, leave-one-run-out).
- Lambda selected via inner 3-fold CV.
- Prediction accuracy: Pearson r on held-out test run.

### Group-Level Statistics
- **Random-effects analysis:** Mean and SEM across subjects per voxel.
- **Significance testing:** One-sample t-test against zero for encoding performance; paired t-test for Ahat vs E comparisons.
- **Multiple comparison correction:** FDR (q < 0.05) across all 59,412 cortical vertices.
- **ROI-level tests:** Paired t-test across subjects (Ahat vs E performance within each ROI).

### Effect Size
- Report Cohen's d for Ahat vs E comparisons.
- Report proportion of noise ceiling explained (if available from repeated runs).

### Permutation Testing (for variance partitioning)
- Block permutation of fMRI time series (100,000 permutations with 50-TR blocks) to generate null distributions for unique variance.
- Matches Wen et al.'s approach for accounting for temporal autocorrelation.

---

## Implementation Priority

### Phase 1 (Immediate — builds directly on existing code)
1. **Variance partitioning** (2.1) — requires fitting 3 models per voxel per layer. Modify `step01_run_encoding.py` to accept concatenated Ahat+E regressors.
2. **Winner-take-all maps** (2.3) — straightforward from existing correlation maps.
3. **ROI-based encoding profiles** (3.1) — load existing correlation maps + Glasser parcellation.

### Phase 2 (High priority — key tests)
4. **Error-propagation vs representation-propagation model comparison** (3.3) — need to run encoding for CNN_like variant (features may already be extracted in `CNN_like/` directory).
5. **Best-layer gradient maps** (3.2) — straightforward from existing per-layer correlations.
6. **2D preference maps** (2.2) — visualization of existing data.

### Phase 3 (Important extensions)
7. **Partial correlation / sequential model building** (3.4) — new regression analyses.
8. **Log10 preprocessing comparison** — run encoding with `step0_LORO_feature_processing_log10.m` and compare.

### Phase 4 (Longer-term)
9. **Temporal context manipulation** (4.1) — requires re-running PredNet with different nt values.
10. **Cross-dataset replication** (4.2) — run analyses on Budapest.
11. **VAE comparison** (4.3) — requires training new model.

---

## Expected Figures for Paper

1. **Fig 1 — Model overview:** PredNet architecture showing Ahat/E separation; encoding pipeline diagram.
2. **Fig 2 — Encoding performance maps:** Group-mean correlation maps for Ahat and E, per layer, on cortical surface.
3. **Fig 3 — Spatial dissociation:** 2D preference maps (Ahat vs E) and variance partitioning (unique Ahat, unique E, shared).
4. **Fig 4 — Hierarchical organization:** Best-layer gradient maps for Ahat and E; ROI-level layer-preference curves.
5. **Fig 5 — Error propagation test:** Error-propagation vs representation-propagation model comparison; encoding performance by ROI for both architectures.
6. **Fig 6 — Temporal horizon (if Tier 4 completed):** Optimal temporal context maps; encoding performance as function of sequence length per ROI.
7. **Supplementary:** Cross-dataset replication; individual subject maps; lambda distribution analysis; log10 vs z-score preprocessing comparison.

---

## Key References (Methodological)

| Method | Reference | Relevance |
|--------|-----------|-----------|
| Voxel-wise encoding | Wen et al. 2018, Cerebral Cortex | Foundational method replicated here |
| Variance partitioning | Lescroart & Gallant 2019, NeuroImage | Unique vs shared variance decomposition |
| Banded ridge regression | Nunez-Elizalde et al. 2019, PLOS Comp Bio | Separate regularization per feature set |
| 2D preference maps | Popham et al. 2021, Nature Neuroscience | Visualization of competing feature spaces |
| Forecast distance/depth | Caucheteux et al. 2023, Nature Human Behaviour | Hierarchical temporal prediction gradients |
| Hierarchical mapping | Guclu & van Gerven 2015, J Neuroscience | CNN layer-to-cortex mapping |
| PredNet architecture | Lotter et al. 2017, ICLR | Model separating prediction from error |
| Predictive coding theory | Rao & Ballard 1999, Nature Neuroscience | Theoretical foundation |
| Cortical parcellation | Glasser et al. 2016, Nature | ROI definitions |
| Data-driven parcellation | Rajimehr et al. 2024, Neuron | Naturalistic movie cortical architecture |

---

## File Structure (Proposed)

```
encoding/
|-- step0_LORO_feature_processing.m          # z-score preprocessing
|-- step0_LORO_feature_processing_log10.m    # log10 + z-score preprocessing
|-- LORO/
|   |-- step01_run_encoding.py               # Single feature type encoding
|   |-- step01_run_encoding_joint.py         # [NEW] Joint Ahat+E for variance partitioning
|   |-- step02_run_beta.py                   # Beta maps
|   |-- step02_run_beta_rms.py               # RMS beta maps
|   |-- step03_run_group_analysis.py         # Group statistics
|   |-- step04_variance_partitioning.py      # [NEW] Unique/shared variance decomposition
|   |-- step05_roi_analysis.py               # [NEW] ROI-based layer-preference curves
|   |-- step06_gradient_maps.py              # [NEW] Best-layer and continuous gradient maps
|   |-- step07_model_comparison.py           # [NEW] Error-prop vs representation-prop
|   |-- step08_2d_preference_maps.py         # [NEW] 2D Ahat vs E colormaps
|   |-- voxelwise_encoding_new.py            # Core encoding algorithm
```
