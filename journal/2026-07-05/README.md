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

Running on Google Colab T4 GPU tonight. The hypothesis: a variational autoencoder pretrained on all 880,000 cells (no labels) learns a 20-dimensional latent space that separates biological signal from technical noise. Classification on the latent representation may outperform classification on 32,000 raw gene values — or it may confirm that the signal is genuinely too weak regardless of feature engineering. Either result is informative.

Results to be added when training completes.
