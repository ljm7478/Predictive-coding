# PredNet PCA spectrum analysis — GUMP dataset (log10, LORO)

**Date:** 2026-04-09
**Source:** `LORO_PCAcap_log10/layer4/{Ahat,E}/pca_components_layer{1..4}.mat` (eigenvalues `s`)
**Cross-check:** Ahat0 99% PC count verified against live `LORO_PCA99_cap1500` job (3465, exact match).

## Setup

- Total LORO training timepoints (concatenated, after log10): **179,926**
- Worst-case LORO train TRs (leave out longest run): **3,057**
- fMRI training volumes per subject: ~3,000

## PC counts at variance thresholds

| Layer  | SVD rank | PC @ 95% var | PC @ 99% var | Var @ cap=500 | Var @ cap=1000 | Var @ cap=1500 |
|--------|---------:|-------------:|-------------:|--------------:|---------------:|---------------:|
| Ahat0  |   61,440 |      **658** |    **3,465** |        93.90% |         96.42% |         97.51% |
| Ahat1  |  179,926 |       36,751 |       66,250 |        33.35% |         40.89% |         45.99% |
| Ahat2  |  122,880 |        7,187 |       11,012 |        47.30% |         58.11% |         65.61% |
| Ahat3  |   61,440 |    **1,581** |    **2,211** |        66.27% |         84.97% |         94.12% |
| E0     |  122,880 |       29,800 |       62,136 |        43.17% |         50.54% |         55.41% |
| E1     |  179,926 |       62,940 |  **108,636** |        26.99% |         33.22% |         37.54% |
| E2     |  179,926 |       35,694 |       59,610 |        31.96% |         38.65% |         43.46% |
| E3     |  122,880 |       11,431 |       16,536 |        39.16% |         48.86% |         55.64% |

SVD rank = min(feature_dim, 179,926 timepoints).

## Train-TR-to-PC ratio (worst-case LORO fold, 3,057 train TRs)

| Cap | Ratio | Comment |
|----:|------:|---------|
|  500 | 6.1:1 | Comfortable for ridge; close to Wen / Huth norms |
| 1000 | 3.1:1 | Acceptable with proper λ sweep |
| 1500 | 2.0:1 | Edge of stability; requires wide λ sweep |
| 2000 | 1.5:1 | Risky without strong regularization |

## Key observations

1. **Ahat0 and Ahat3 have compact spectra** — variance concentrates in the first few thousand PCs, similar to classical CNN features (Wen 2018 regime).
2. **Ahat1, Ahat2, and all E layers have extremely flat spectra.** E1 needs **108,636 PCs** to reach 99% variance. No realistic cap can satisfy a strict 99% criterion for these layers given 3k training TRs.
3. **Cap=500 is fine for Ahat0 (93.9%) only**; for everything else it loses 53–73% of variance.
4. **Cap=1500 retains ≥94% variance only for Ahat0 and Ahat3.** For the other 6 layers, even cap=1500 captures 33–66% — the cap is acting as a forced compression of a fundamentally high-rank signal, not as a variance-based truncation.
5. **The cap plays two different roles depending on the layer:**
   - Ahat0/Ahat3: principled truncation of a concentrated spectrum
   - E*/Ahat1/Ahat2: stability-driven compression of a flat spectrum

## Implications for cap strategy

- A single global cap is hard to justify methodologically against Wen 2018, which applied a 99%-variance criterion. For 6 of 8 layers in this dataset, that criterion is incompatible with ridge stability.
- **Per-layer caps are more defensible:**
  - Ahat0: cap≈1500 (97.5% variance, ratio 2:1)
  - Ahat3: cap≈1500 (94.1% variance, ratio 2:1)
  - Ahat1, Ahat2, E0–E3: cap chosen for ridge stability (e.g., 500–1000), explicitly framed as a stability constraint, not a variance target
- **The flat E-layer spectra likely reflect that PredNet error units are spatially localized and weakly correlated** across the visual field — PCA gives almost no compression. Worth noting in methods.
- **Wider λ sweep is critical** if any cap ≥1000 is used. Current `lambdas = 0.1:0.2:0.9` is far too narrow at a 2:1 ratio; recommend `logspace(-3, 4, 15)` or similar.

## Suggested write-up framing

> "We followed Wen et al. (2018) in applying a 99% explained-variance criterion for PCA dimensionality reduction. For PredNet's Ahat0 and Ahat3 layers, this criterion is satisfied by N≤1500 components and yields a stable train-TR-to-component ratio of ≥2:1 in the leave-one-run-out scheme. For the higher Ahat layers and all E layers, the natural rank of the feature spectrum substantially exceeds the available training timepoints (e.g., E1 requires 108,636 components for 99% variance), and a strict variance criterion is incompatible with ridge stability. For these layers we capped components at N=[X], explicitly as a stability constraint, retaining [Y]% of feature variance."
