# Beta Analysis Recommendation: Encoding Performance and Weight Interpretation

## What King/Huth/Gallant Actually Do

These labs do NOT compute "beta maps" as a summary statistic of the weight matrix W. Their standard outputs are:

### Encoding Performance (Primary Output — All Labs)
- **Metric:** Pearson correlation (r) or R-squared (r²) between predicted and measured fMRI on held-out test data, per voxel.
- This IS the main result. No further "beta map" is needed to show encoding strength.
- Your existing correlation maps already follow this standard.

### Weight Interpretation (Secondary — Gallant/Huth Only)

| Lab | What they report | How |
|-----|-----------------|-----|
| **Gallant/Huth** | Raw weight vectors W projected onto interpretable dimensions (semantic categories, motion energy filters) | W is meaningful because features are interpretable (WordNet categories, Gabor filters, word embeddings) |
| **King/Caucheteux** | Difference in R between models (forecast score = R_full - R_base) | Never summarize W into a single number |
| **Wen et al.** | Best-layer index per voxel; prediction accuracy maps | No beta maps |

**The key distinction:** In Huth/Gallant's work, the feature space is interpretable (e.g., "this weight means the voxel responds to faces"). Their W vectors can be directly examined because each dimension has semantic meaning. In your pipeline, PCA components of PredNet activations are abstract — you cannot say "PC #47 means motion" — which is why summarizing W into a single number is problematic.

---

## Why W-Based Beta Summaries Are Suboptimal

Your W matrix has shape `(n_PCs, n_voxels)`. Each voxel has n_PCs weights mapping PCA components to fMRI response. Collapsing this to a single value per voxel loses information and introduces interpretation issues:

| Method | Formula | What It Tells You | Problem |
|--------|---------|-------------------|---------|
| **RMS(W)** | `sqrt(mean(W², axis=0))` | Magnitude of encoding weights | Highly correlated with r — redundant with correlation maps |
| **PC1(W)** | PCA on W → first component score | Dominant weight pattern across voxels | Arbitrary sign; meaning is unclear; different across subjects |
| **Mean(W)** | `mean(W, axis=0)` | Average weight | Cancellation across PCs with opposite signs; often near zero |
| **Group-PCA on W** | Average W across subjects → PCA | Shared weight pattern | Still abstract — what does PC1 of W represent neurally? |

### RMS vs r² Redundancy
RMS of W measures the overall magnitude of encoding weights. But encoding weights that are large also produce better predictions — so RMS is essentially a noisier version of r². There is rarely additional information in RMS that isn't already captured by prediction accuracy.

### PC1 Sign Ambiguity
PCA components have arbitrary sign (both +PC1 and -PC1 explain the same variance). This means:
- A voxel with PC1 score = +0.5 and another with -0.5 may have equally strong encoding but opposite PC1 signs.
- Group averaging of PC1 scores can cancel out, producing artifactual near-zero regions.
- The "pattern" captured by PC1 depends on the covariance structure of W, not on any neural property.

---

## Recommended Replacement: Three Analyses

### 1. R-Squared Maps (Encoding Strength)

Simply square your existing correlation maps:

```python
r_squared = correlation_map ** 2
```

This gives the proportion of fMRI variance explained by the model at each voxel — the standard measure of encoding strength in the field.

- Interpretable: "This voxel has 25% of its variance explained by PredNet Ahat features."
- Comparable across layers and feature types.
- Already computed (just square your existing r maps).

### 2. Variance Partitioning (Ahat vs E Comparison)

This replaces beta maps entirely for the core research question: "Do Ahat and E explain the same or different aspects of fMRI activity?"

**Method (Lescroart & Gallant 2019; Popham et al. 2021):**

For each voxel and each layer, fit three ridge regression models:

```
Model_Ahat:  fMRI ~ Ahat features only     --> R²_Ahat
Model_E:     fMRI ~ E features only         --> R²_E
Model_Joint: fMRI ~ [Ahat, E] concatenated  --> R²_Joint
```

Decompose into unique and shared variance:

```
Unique_Ahat = R²_Joint - R²_E       (variance only Ahat explains)
Unique_E    = R²_Joint - R²_Ahat    (variance only E explains)
Shared      = R²_Ahat + R²_E - R²_Joint   (variance both explain)
```

Compute a preference index per voxel:

```
PI = (Unique_Ahat - Unique_E) / (Unique_Ahat + Unique_E)
```

- PI = +1: purely prediction-driven
- PI = -1: purely error-driven
- PI = 0: equal contribution

**Why this is better than any W-based summary:**
- Directly measures what each feature type uniquely contributes to explaining fMRI.
- Does not depend on abstract PCA component weights.
- Standard approach in computational neuroscience for comparing feature spaces.
- Tells you whether Ahat and E are redundant (large shared, small unique) or dissociable (large unique, small shared).

### 3. Winner-Take-All Maps (Spatial Organization)

For each voxel, determine which feature type and layer wins:

```python
# best_type: 'Ahat' or 'E' (which has higher r)
# best_layer: L0/L1/L2/L3 (which layer has highest r)
best_type = 'Ahat' if r_Ahat > r_E else 'E'
best_layer = argmax(r[layer]) across L0-L3
```

This directly shows the spatial organization of prediction vs error dominance across the cortex, without any W-based intermediary.

---

## Summary: Analysis Priority

| Priority | Analysis | Replaces | Standard in Field? |
|----------|----------|----------|--------------------|
| 1 | **r² maps** (square existing r maps) | RMS beta as encoding strength | Yes (all labs) |
| 2 | **Variance partitioning** (unique Ahat, unique E, shared) | PC1 beta for Ahat vs E comparison | Yes (Gallant lab) |
| 3 | **Winner-take-all maps** (best type + best layer per voxel) | Gradient-based layer assignment | Yes (Wen et al., Guclu & van Gerven) |
| Drop | ~~PC1 of W~~ | — | No |
| Drop | ~~RMS of W~~ | — | No (redundant with r²) |
| Drop | ~~Group-PCA on W~~ | — | No |

---

## Key References

- **Lescroart & Gallant 2019** (NeuroImage): Variance partitioning for comparing feature spaces in encoding models.
- **Popham et al. 2021** (Nature Neuroscience): 2D preference maps comparing visual vs linguistic encoding.
- **Nunez-Elizalde et al. 2019** (PLOS Comp Bio): Banded ridge regression for joint models with separate regularization per feature set.
- **Caucheteux et al. 2023** (Nature Human Behaviour): Forecast score (difference in R) as model comparison metric.
- **Huth et al. 2016** (Nature): Semantic weight maps projected onto interpretable dimensions.
