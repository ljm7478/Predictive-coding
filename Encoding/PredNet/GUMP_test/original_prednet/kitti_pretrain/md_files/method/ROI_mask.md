# Data-Driven ROI Definition

## Overview

To define cortical regions of interest (ROIs) for subsequent layer profile analyses, we employed a data-driven approach that identified Glasser parcels where the encoding model significantly and reliably predicted brain activity. Parcel inclusion required jointly satisfying three criteria: (1) statistical significance of encoding accuracy, (2) a minimum effect size, and (3) reliable stimulus-driven neural responses.

## Encoding Accuracy

For each participant, voxelwise encoding models were fit separately for Prediction (Ahat) and Prediction Error (E) units across four PredNet layers (L1--L4), yielding a correlation coefficient (r) between predicted and observed fMRI responses at each cortical vertex. To obtain a single summary statistic per vertex per participant, we took the maximum r across the four layers, reasoning that a vertex qualifies as well-predicted if any layer captures its response profile. This produced a matrix of max-r values with dimensions (N_subjects x N_vertices).

## Statistical Significance (FDR Correction)

At each vertex, we performed a one-sample t-test across participants against the null hypothesis that the population mean encoding accuracy equals zero (H0: r = 0). We used one-tailed tests, as only positive encoding accuracy (r > 0) reflects meaningful prediction. The resulting p-values were corrected for multiple comparisons across all cortical vertices using the Benjamini-Hochberg false discovery rate (FDR) procedure at alpha = 0.05.

This procedure was carried out independently for Prediction and Error signals. A vertex was retained if it reached significance for either signal type (union mask), ensuring that the ROI definition was not biased toward one signal type. This union approach is critical because subsequent analyses compare Prediction and Error signals within the same ROIs; defining ROIs based on only one signal would introduce selection bias.

## Parcel-Level Statistics

Surviving vertices were mapped onto the HCP-MMP1.0 (Glasser) cortical parcellation (180 parcels per hemisphere, 360 total). For each parcel, we computed: the total number of vertices (n_total), the number of FDR-significant vertices (n_sig), the mean encoding accuracy across significant vertices for each signal type (mean r_Ahat, mean r_E) and their maximum (r_max), and the mean inter-subject correlation (ISC) across all vertices in the parcel. Bilateral homologues (e.g., L_V1_ROI and R_V1_ROI) were merged into a single entry by summing vertex counts and computing n_sig-weighted averages of encoding accuracy.

## Parcel Inclusion Criteria

A parcel was included in the final ROI set if it satisfied all three of the following criteria:

1. **Statistical significance.** The parcel contained at least one FDR-significant vertex (n_sig > 0).

2. **Effect size.** The mean encoding accuracy across significant vertices exceeded r_max >= 0.030, corresponding approximately to the top quartile of parcel-level encoding accuracy across all 180 Glasser parcels.

3. **Stimulus-driven reliability.** The mean ISC within the parcel exceeded 0.05, ensuring that the parcel exhibited reliable stimulus-driven responses across participants. This criterion guards against including parcels where encoding accuracy may be artifactually high due to noise rather than genuine stimulus-driven signal.

The effect size threshold was chosen because, with 15 participants and naturalistic movie stimulation, nearly all cortical vertices reached statistical significance after FDR correction. Statistical significance alone was therefore insufficient to distinguish meaningfully predicted regions from minimally predicted ones. The r >= 0.030 threshold provides a principled separation, excluding parcels in somatomotor, auditory, and prefrontal cortex where PredNet features showed negligible encoding performance.

We also evaluated noise-ceiling normalization (r/NC, where NC = ISC) as an alternative thresholding approach. However, this metric systematically inflated values in parcels with very low ISC (e.g., orbitofrontal cortex, ISC ~ 0.02), where dividing a small r by a near-zero ISC produced artifactually high r/NC ratios (> 1.0). The effect size approach proved more appropriate for ROI selection, while noise-ceiling normalization remains suitable for within-ROI comparisons of explained variance.

## Output

This procedure yielded 60 bilateral Glasser parcels, spanning early visual cortex (V1--V4), dorsal visual areas (V3A, V3B, V6, V6A, V7), motion-selective regions (MT, MST, FST), ventral visual areas (V8, LO1--LO3, VMV1--VMV3, FFC, PIT, VVC), scene-selective regions (PH, PHT, PHA1--PHA3), temporal-parietal-occipital junction areas (TPOJ1--TPOJ3, STV, PGi, PGp), and dorsal attention/parietal regions (IPS1, IP0, PCV, POS1, POS2, 7Am, DVT). Each vertex within these parcels was assigned an integer label corresponding to its parcel rank (ordered by descending r_max), producing a cortical ROI mask used in all subsequent analyses.
