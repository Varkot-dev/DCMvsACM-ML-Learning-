# DCM vs ACM — Honest ML on Single-Cell Heart Data

**A study in data leakage, biological confounds, and why most published AUC numbers in single-cell ML are wrong.**

Built by Varshith Kotagiri (University of Pennsylvania) with Claude Code.

---

## What this project is

Dilated cardiomyopathy (DCM) and arrhythmogenic cardiomyopathy (ACM) are two serious heart diseases that look similar clinically but have different underlying biology, genetics, and long-term outcomes. The Reichart 2022 *Science* paper published a single-nucleus RNA-seq atlas of 880,000 nuclei from 79 hearts.

The obvious ML question: **can we train a classifier to distinguish DCM from ACM at single-cell resolution?**

The less obvious question: **can we trust the AUC number we get?**

This project answers the second question. The answer is: **no, not if you follow standard ML practice.** And we show exactly why, with numbers.

---

## The core finding

| Classifier | Naive AUC | Honest AUC | Leakage inflation |
|------------|-----------|------------|-------------------|
| XGBoost    | 0.883     | 0.701      | **+0.182**        |
| Random Forest | 0.834  | 0.705      | **+0.129**        |

**Naive** = standard cell-level cross-validation (what most papers do).
**Honest** = patient-level cross-validation (all cells from one patient stay together).

The naive pipeline overstates performance by 13–18 AUC points because it lets cells from the same patient appear in both training and validation. A model that memorizes patient-specific expression patterns scores high without learning any disease biology.

---

## The three data leakage bugs we fixed

### Bug 1 — Cell-level CV
`StratifiedKFold` splits cells randomly. 80% of one patient's cells end up in training, 20% in validation. The model memorizes patient identity, not disease.

**Fix:** `StratifiedGroupKFold(groups=donor_id)` — all cells from one patient land in the same fold.

### Bug 2 — SMOTE before split
Synthetic minority oversampling before splitting leaks synthetic cells (generated from validation patients) into training.

**Fix:** SMOTE inside an `imblearn.Pipeline` — runs only on training data, skipped at predict time.

### Bug 3 — HVG selection before split
Selecting highly variable genes on the full dataset means test cells influenced gene selection.

**Fix:** HVGSelector as a pipeline step that fits only on training data.

---

## What PCA shows

PC1 = **1.2% variance**. PC2 = **1.2% variance**. There is no dominant DCM-vs-ACM axis in gene expression space. The biology is genuinely hard — which is why honest AUC is 0.70, not 0.95. This is not a pipeline failure. This is what honest evaluation looks like on a hard problem.

---

## What SHAP shows

| Rank | Gene | Direction | What it is |
|------|------|-----------|------------|
| 1 | NPPB | → ACM | BNP — real heart failure biomarker ✓ |
| 2 | ANKRD2 | → ACM | Muscle stress response ✓ |
| 3 | **XIST** | → DCM | **X-chromosome inactivation = sex confound ✗** |
| 5 | MYL7 | → DCM | Myosin light chain — sarcomere gene ✓ |
| 9 | NPPA | → ACM | ANP — natriuretic peptide ✓ |

XIST being a top-3 predictor means the model is learning **sex of the patient**, not disease. This is a demographic confound that hyperparameter tuning cannot fix.

---

## Biological constraints

| | DCM | ACM |
|-|-----|-----|
| Patients | 52 | **8** |
| Cells | 142,946 | 23,573 |
| Genetic diversity | 12 variants | Mostly PKP2 (6/8 patients) |

Effective sample size is **patients, not cells**. With 8 ACM patients, one unusual donor (H29: LMNA mutation but ACM diagnosis) drops fold 4 AUC to 0.57. The model is confused by known genotype-phenotype overlap — real biology, not a code bug.

---

## Project structure

```
DCMvsACM/
├── cardiomyopathy-ml/
│   ├── data.py            # Loading, validation, sparse HVG pre-selection
│   ├── models.py          # RF + XGBoost pipelines with SMOTE + HVGSelector
│   ├── evaluation.py      # Naive vs honest CV
│   ├── run_experiment.py  # CLI entry point
│   └── tests/             # 16 tests including patient-leak assertion
├── results/
│   ├── comparison_table.csv   # The main finding
│   └── fold_results.csv       # Per-fold breakdown
└── docs/
    └── interview-notes.md     # Study guide for defending this work
```

---

## How to run

```bash
pip install anndata scikit-learn imbalanced-learn xgboost shap matplotlib SciencePlots

# Test pipeline logic (no data download needed)
cd cardiomyopathy-ml && python run_experiment.py --synthetic

# Real data (download Reichart 2022 from CELLxGENE collection e75342a8-...)
python run_experiment.py --data cardiomyocytes.h5ad --n-genes 2500 --n-splits 5
```

---

## What's next

| Phase | What | Why |
|-------|------|-----|
| B | Cross-cohort validation | Does it generalize to different labs? |
| C | Pseudobulk analysis | Statistically more rigorous for n=8 ACM |
| D | Sex-stratified evaluation | Fix the XIST confound |
| E | Genotype-stratified evaluation | Fix the PKP2 confound |

---

*Built with Claude Code. The AI wrote the pipeline; the student understood it well enough to defend it.*
