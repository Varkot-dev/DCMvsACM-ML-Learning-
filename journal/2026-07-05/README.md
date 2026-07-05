# July 5, 2026

With the honest evaluation pipeline working, the next question is whether the 0.70 AUC reflects real disease biology or just noise the model is exploiting. I ran three analyses to find out: PCA on the gene expression space, SHAP to audit what the model is actually using, and pseudobulk differential expression at the patient level.

---

## PCA

I ran PCA on 2,000 highly variable genes across a balanced subsample of 4,000 cells (2,500 DCM, 1,500 ACM).

PC1 explains 1.2% of variance. PC2 explains 1.2%.

In a dataset where disease creates a strong transcriptional signal, you would expect PC1 to explain 10–30% of variance and see two separated clouds on the scatter plot. At 1.2%, there is no dominant DCM-vs-ACM axis. The cell-to-cell variation — driven by cell subtype, patient identity, sex, sequencing depth — completely swamps any disease signal. The honest AUC of 0.70 makes biological sense: this is a genuinely hard classification problem, not a fixable pipeline problem.

![PCA](figC_pca.png)

---

## SHAP

I ran SHAP TreeExplainer on a Random Forest trained on 2,000 HVGs, explaining 600 balanced cells. The goal was to check whether the features driving predictions are biologically meaningful.

| Gene | Direction | What it is |
|------|-----------|------------|
| NPPB | → ACM | BNP — the primary clinical biomarker for heart failure severity |
| ANKRD2 | → ACM | Muscle stress response protein, dysregulated in cardiomyopathy |
| **XIST** | → DCM | **X-chromosome inactivation — expressed only in female cells** |
| MYL7 | → DCM | Myosin light chain, sarcomere gene |
| NPPA | → ACM | ANP — natriuretic peptide, cardiac stress marker |
| LINC02147 | → DCM | lncRNA with no known cardiac function |

NPPB and NPPA are encouraging — these are real biology. BNP is what cardiologists measure in blood to assess heart failure severity, and the model found it without being told. MYL7 and sarcomere dysregulation are established DCM mechanisms.

XIST in the top three is the problem. XIST silences one X chromosome in female cells — it has nothing to do with cardiomyopathy. Its presence means the model is exploiting the sex composition of the cohort. The DCM group is 74% male; the ACM group is 62% male. That imbalance is small but with only 8 ACM patients it's enough for the model to latch onto.

![SHAP](figD_shap.png)

---

## Pseudobulk differential expression

The cleanest statistical test available: average all cells from each patient into a single expression profile, then compare DCM and ACM at the patient level. This is the approach computational biologists use when donor counts are small, because it correctly treats patients — not cells — as independent observations.

60 patients × 32,383 genes. Wilcoxon rank-sum test per gene. Benjamini-Hochberg FDR correction for 32,383 simultaneous tests.

**FDR < 0.05: 0 genes. FDR < 0.10: 0 genes.**

No gene survives multiple testing correction. The top raw hit is VTA1 at p = 0.000048, but testing 32,383 genes simultaneously means you expect a few hits at that threshold by chance alone. After correction, none hold up.

The most biologically interesting result is PDGFB: raw p = 0.00028, log₂FC = −1.78, higher in ACM. Platelet-derived growth factor B is a fibrosis signaling molecule. ACM is pathologically defined by fibro-fatty replacement of myocardium — PDGFB being elevated in ACM is a coherent biological hypothesis. It doesn't survive FDR correction at n = 8, but it's the gene I'd look at first if a larger cohort became available.

![Pseudobulk DE](figE_pseudobulk.png)

---

## What this means

The pseudobulk result is not a failure. It's the data giving a clear answer: cardiomyocyte transcriptomes of DCM and ACM patients are not statistically distinguishable at eight ACM donors with this gene panel. The problem isn't the pipeline. It's the sample size.

Three paths forward:

**More patients.** The NLRP3 study (BMC Medicine, 2024) has a separate ACM single-cell dataset from 6 end-stage ARVC patients. Combining it with Reichart gives 14 ACM donors — still small, but meaningfully better. Cross-cohort analysis would also test generalization.

**Different cell types.** This atlas is cardiomyocytes. The PKP2 transcriptional effect is strongest in epicardial cells, not cardiomyocytes. A classifier trained on the full multi-cell-type atlas might find signal that's invisible in cardiomyocytes alone.

**Different question.** Binary classification may be the wrong framing. A pathway analysis asking "which biological processes are perturbed in ACM versus DCM" is statistically better suited to small sample sizes than a per-gene test.

---

## scVI pretraining

Started this on Google Colab with a T4 GPU. The hypothesis: a variational autoencoder pretrained on all 880,000 cells (no labels) learns a 20-dimensional latent space that separates biological signal from technical noise. Classification on the latent representation may outperform classification on 32,000 raw gene values — or it may confirm that the signal is genuinely too weak regardless of feature engineering. Either result is informative.

It did not go smoothly, and the failure is worth documenting because the fix involved a real methodological compromise, not just a bug.

### What broke

The original notebook loaded the full h5ad with `ad.read_h5ad(H5AD_PATH)` — all 880,000 cells, 32,383 genes, into RAM at once — then called `sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')` to select highly variable genes before training.

`seurat_v3` is not a simple statistic. It fits a loess (locally estimated scatterplot smoothing) regression across all genes to model the expected relationship between mean expression and variance, then selects genes with more variance than that curve predicts for their expression level. Fitting and applying that regression requires scanpy to work with dense representations of large chunks of the matrix internally, even though the input is sparse. A full dense version of this matrix would be roughly 880,000 × 32,383 × 4 bytes ≈ 114 GB — scanpy chunks this rather than materializing it all at once, but the peak memory during those chunked operations, combined with the already-loaded sparse matrix and Python/scanpy overhead, exceeded Colab's free-tier 12.7 GB RAM ceiling. The session crashed with "used all available RAM" partway through this cell, more than once, including after a kernel restart.

### What we changed

**Loading:** switched to `ad.read_h5ad(H5AD_PATH, backed='r')` followed by `adata[mask].to_memory()`, filtering to DCM+ACM cells *while* reading rather than after. `backed='r'` opens a file handle without pulling the expression matrix into memory; `.to_memory()` then materializes only the ~166,000 matching rows. This alone cuts memory roughly 5x before any gene-selection computation happens.

**Gene selection:** replaced `seurat_v3` with a sparse Fano factor calculation — variance divided by mean, computed as `E[X²] − E[X]²` using `X.mean(axis=0)` and `X.power(2).mean(axis=0)` directly on the sparse matrix. Both operations are sparse-aware column reductions; neither requires densifying anything. This is the same method already used in the laptop pipeline (`data.py`) for pre-selecting genes before the full 40 GB matrix would otherwise need to be dense.

### Why the replacement is a worse method, not just a different one

This needs to be said plainly: Fano factor and `seurat_v3` are not interchangeable, and the substitution was made for memory reasons, not because it's the better statistic.

`seurat_v3`'s loess correction exists to solve a specific, well-known problem in scRNA-seq: genes with higher mean expression mechanically have higher variance from Poisson counting noise alone, independent of any biology. The loess curve models that expected noise floor, and genes are selected based on their *residual* variance above that curve — variance that can't be explained by expression level alone is a better proxy for genuine biological variability. This method was developed and validated by the Satija lab specifically to correct for that mean-variance confound, and it's the field-standard default for a reason.

Raw Fano factor makes no such correction. It selects genes with high variance relative to mean, full stop, without asking whether that variance is expected from counting noise given the expression level. In practice this biases selection toward highly-expressed genes, some fraction of which are "variable" only because of Poisson noise, not biology. The 3,000 genes selected by Fano factor and the 3,000 that `seurat_v3` would have selected overlap substantially but not completely — some genes with moderate expression and genuinely interesting variance patterns may be excluded in favor of genes that are just noisy because they're highly expressed.

**Concretely, this means:** scVI's pretraining tonight sees a slightly less curated, slightly noisier 3,000-gene panel than the field-standard approach would provide. If the eventual latent-space classification result comes back different from what raw-gene classification produced, part of that difference could be attributable to gene panel quality rather than the value of pretraining itself.

**How much this matters:** for the purpose of tonight's run — a first-pass check on whether latent features help at all — this is an acceptable, honest shortcut. It would not be acceptable for a publication-quality result. The correct fix there is either running `seurat_v3` on a machine with enough RAM (Penn HPC, or Colab Pro's high-RAM runtime), or explicitly validating that Fano-factor-selected genes give comparable downstream results to `seurat_v3`-selected genes before trusting the comparison.

Notebook fix: `notebooks/scvi_pretraining.ipynb`, cells 3–4, commit `71f8bcc`.

Results to be added when training completes.
