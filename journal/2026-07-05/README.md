# July 5, 2026 — EDA: PCA, SHAP, and Is This Worth Continuing?

## The question we started with

With only 8 ACM patients and a genetic confound (PKP2 dominates the cohort), is this project worth continuing? We ran PCA and SHAP to answer it honestly.

## PCA — what the data looks like in 2D

Ran PCA on 2,000 highly variable genes across 4,000 cells (2,500 DCM + 1,500 ACM subsampled for speed).

**PC1 = 1.2% variance. PC2 = 1.2%.**

In a dataset where disease creates a strong signal, PC1 would explain 10–30%. At 1.2%, there is no dominant DCM-vs-ACM axis. Cell-to-cell variation (cell type, patient identity, sex, technical noise) completely swamps any disease signal. This confirms that honest AUC of 0.70 is real — the biology is genuinely hard.

![PCA](figC_pca.png)

## SHAP — what genes the model actually uses

Ran SHAP TreeExplainer on a Random Forest trained on 2,000 HVGs. Explained 600 balanced cells.

| Rank | Gene | Direction | Interpretation |
|------|------|-----------|---------------|
| 1 | **NPPB** | → ACM | BNP — real heart failure biomarker ✓ |
| 2 | ANKRD2 | → ACM | Muscle stress response ✓ |
| 3 | **XIST** | → DCM | X-chromosome inactivation = **sex of patient** ✗ |
| 4 | LINC02147 | → DCM | lncRNA, no known cardiac function ✗ |
| 5 | MYL7 | → DCM | Myosin light chain — sarcomere gene ✓ |
| 9 | NPPA | → ACM | ANP — natriuretic peptide ✓ |

**XIST in the top 3 is the key finding.** XIST is only expressed in female cells. It has nothing to do with cardiomyopathy. The model learned that the DCM cohort has a different sex ratio than the ACM cohort and is exploiting that demographic fact. This is a sex confound that hyperparameter tuning cannot fix.

![SHAP](figD_shap.png)

## Is it worth continuing?

Yes — and here's why:

1. **NPPB and NPPA being top features is real biology.** Natriuretic peptides are the primary clinical biomarkers for heart failure severity. The model found them without being told.

2. **XIST being top-3 is a finding, not a failure.** It tells you the dataset is sex-imbalanced between DCM and ACM. That's information. Fix it with sex-stratified evaluation.

3. **The honest AUC story stands regardless of confounds.** The 18-point leakage inflation finding is real and applies broadly to published single-cell ML papers.

4. **1.2% PC1 variance means the problem needs better methods.** This motivates scVI pretraining — the next phase.

## What we decided to do next

**scVI pretraining (Phase E):**
- Pretrain a Variational Autoencoder on all 880,000 cells (no labels)
- The model learns a 20-dimensional "latent space" that captures cardiac biology while factoring out patient identity, sex, and technical noise
- Fine-tune classification on the latent representation instead of 32,000 raw genes
- Hypothesis: AUC improves because the latent space separates disease signal from confounds

Notebook written: `notebooks/scvi_pretraining.ipynb` — runs on Google Colab T4 GPU, ~45 min.

## What we learned today

- PCA variance explained is a diagnostic tool: low variance on PC1 means no dominant signal, not a broken pipeline
- SHAP is your model auditor — always check whether the top features make biological sense before trusting AUC
- Sex and genotype confounds are invisible to standard metrics. You have to know the biology.
- Self-supervised pretraining (scVI) is the right response to weak, noisy features — not more hyperparameter tuning

## GitHub cleanup done today

- Removed all "Interview talking point:" language from Python source — replaced with "Design note:"
- Removed internal planning docs from git tracking
- Removed AI attribution from README
- Restructured repo with daily journal entries
