# Variance Partitioning between Prediction (Ahat) and Error (E) Features

## Overview

To dissociate the unique and shared contributions of PredNet's prediction (Ahat) and prediction-error (E) representations to cortical responses, we performed voxelwise variance partitioning following the method established by Lescroart and Gallant (2019, NeuroImage) and used by Popham et al. (2021, Nature Neuroscience) to quantify the alignment between visual and linguistic semantic representations. For each voxel and each PredNet layer, three ridge regression encoding models were fit and compared: a model using only Ahat features, a model using only E features, and a joint model using both feature sets concatenated. The cross-validated coefficients of determination from these three models were then decomposed into unique and shared variance components, and a per-voxel preference index was computed to summarize whether the voxel's response was more strongly driven by prediction or by error signals.

## Encoding Models

For each PredNet layer L (L = 1, 2, 3, 4) and each subject, three ridge regression models were fit using the same leave-one-run-out (LORO) cross-validation scheme as the primary encoding analysis. The Ahat-only model regressed the fMRI BOLD time series against the PCA-reduced Ahat_L features; the E-only model regressed against the PCA-reduced E_L features; and the joint model regressed against the column-wise concatenation of Ahat_L and E_L features. All three models used the same lambda grid ([0.1, 0.3, 0.5, 0.7, 0.9]) and the same nested 3-fold inner cross-validation for regularization parameter selection on the training folds. Regressors and fMRI time series were demeaned using only the training-fold statistics before model fitting, matching the demeaning convention of the primary encoding pipeline. A single shared lambda was used across the joint feature concatenation, providing a first-pass estimate; a banded ridge formulation with separate lambdas per feature group (Nunez-Elizalde et al., 2019) would constitute a more rigorous extension.

## Cross-Validated Coefficient of Determination

For each LORO fold, the model trained on seven runs was used to predict fMRI activity on the held-out run, and the per-voxel Pearson correlation between predicted and observed activity was computed. Correlations were averaged across the eight folds, and negative mean correlations were set to zero before squaring. This sign-preserving conversion to R-squared follows the convention of Wen et al. (2018) and Lescroart and Gallant (2019), and prevents anti-correlated predictions from spuriously inflating the variance estimates. The procedure yielded three cross-validated R-squared maps per voxel and per layer: R2_Ahat, R2_E, and R2_Joint.

## Variance Decomposition

The three R-squared maps were decomposed into unique and shared components per voxel:

    Unique_Ahat = R2_Joint - R2_E
    Unique_E    = R2_Joint - R2_Ahat
    Shared      = R2_Ahat + R2_E - R2_Joint

Unique_Ahat represents variance that the joint model captures beyond what the E-only model can explain, that is, variance only the Ahat features account for. Unique_E is defined symmetrically. The shared component captures variance jointly explained by both feature sets, reflecting redundancy or co-encoding of prediction and error information at that voxel. Slightly negative unique components can occur when the joint model is regularized away from the optimal additive solution; these are interpreted as "essentially zero" rather than as evidence against either feature set.

## Preference Index

To summarize the relative contribution of prediction versus error signals at each voxel, a preference index was computed:

    PI = (Unique_Ahat_pos - Unique_E_pos) / (Unique_Ahat_pos + Unique_E_pos)

where Unique_X_pos = max(Unique_X, 0). Clipping the unique components to non-negative values prior to computing the index keeps PI bounded in the interval [-1, +1] and prevents sign instability arising from small noise-driven negative values. PI = +1 indicates a voxel whose explainable variance is entirely captured by the Ahat features (purely prediction-driven), PI = -1 indicates a voxel entirely captured by the E features (purely error-driven), and PI = 0 indicates equal unique contributions from both feature sets. Voxels for which the sum Unique_Ahat_pos + Unique_E_pos fell below 1e-6 (essentially neither model contributing) were assigned PI = 0 by definition.

## Output Maps

The procedure produced seven cortical maps per subject and per layer: R2_Ahat, R2_E, R2_Joint, Unique_Ahat, Unique_E, Shared, and PI. Each map has shape (4 layers x N_vertices) and was saved in three formats: a bundled MATLAB file (.mat), a compressed NumPy archive (.npz), and one CIFTI dscalar file per metric containing all four layers as separate scalar maps. The CIFTI files use the same cortical-only grayordinate template as the primary encoding outputs and can be visualized directly in Connectome Workbench.

## Bivariate Visualization (Popham 2021 Figure 2 Style)

To visualize the joint prediction-versus-error preference structure across the cortex, we generated bivariate (two-dimensional) colormap renderings analogous to Figure 2 of Popham et al. (2021). For each cortical vertex, the pair (R2_Ahat, R2_E) was binned into a discrete 8 x 8 grid, yielding 64 distinct colors arranged so that the red channel encodes R2_Ahat and the blue channel encodes R2_E. Vertices well-predicted only by Ahat appear red, vertices well-predicted only by E appear blue, vertices well-predicted by both appear magenta, and vertices unpredicted by either appear dark. This is the variance-partitioning analog of the `cortex.Volume2D` rendering used in the original Popham paper, where two model performances are mapped jointly onto a single 2D colormap rather than reported as separate scalar maps.

For each subject, the bivariate index was computed independently per layer, expanded from the cortical CIFTI grayordinates onto the full fsLR_32k surface (32492 vertices per hemisphere) using the brain model axis vertex indices (medial-wall vertices remained unassigned and were rendered as background sulcal shading), and projected onto the subject's own inflated fsLR_32k surface using nilearn. Each figure displays four PredNet layers (rows) by four hemispheric views (left lateral, left medial, right lateral, right medial), with the 2D colormap legend in the rightmost column. A group-mean variant averages R2_Ahat and R2_E vertex-wise across subjects in fsLR_32k space prior to binning and rendering on a reference subject's surface; vertex-wise averaging is valid because all subjects share the same fsLR_32k vertex correspondence after ciftify processing.

## References

- Lescroart, M. D., and Gallant, J. L. (2019). Human scene-selective areas represent 3D configurations of surfaces. *NeuroImage*.
- Popham, S. F., Huth, A. G., Bilenko, N. Y., Deniz, F., Gao, J. S., Nunez-Elizalde, A. O., and Gallant, J. L. (2021). Visual and linguistic semantic representations are aligned at the border of human visual cortex. *Nature Neuroscience*.
- Nunez-Elizalde, A. O., Huth, A. G., and Gallant, J. L. (2019). Voxelwise encoding models with non-spherical multivariate normal priors. *NeuroImage*.
- Wen, H., Shi, J., Chen, W., and Liu, Z. (2018). Deep residual network predicts cortical representation and organization of visual features for rapid categorization. *Scientific Reports*.
