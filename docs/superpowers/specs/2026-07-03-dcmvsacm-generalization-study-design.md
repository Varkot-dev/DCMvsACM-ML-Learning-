# DCMvsACM — Generalization Study Design

**Date:** 2026-07-03
**Status:** Approved (design), pending implementation plan

## Problem

The existing `cardiomyopathy-ml/dcm_vs_acm.py` (699 lines) classifies Dilated
Cardiomyopathy (DCM) vs Arrhythmogenic Right Ventricular Cardiomyopathy (ACM/ARVC)
from single-nucleus RNA-seq. It is a generic "Random-Forest-on-highly-variable-genes"
pipeline of a kind already saturated in the literature, and it contains evaluation
bugs that make its reported accuracy meaningless:

1. **SMOTE runs on the full dataset before the train/test split** — synthetic
   minority cells leak across the split boundary.
2. **`cross_val_score` runs on the already-resampled matrix** — compounds the leak.
3. **Cell-level random splitting** — cells from one patient land in both train and
   test, so the model can memorize the patient rather than learn disease biology.
4. **Single-value `GridSearchCV` grids** — cosmetic tuning, no real search.
5. **~300 lines of dead code** — a transformer, GSEA, PDF report, attention plots,
   all defined but never called.

## Goal

Turn the project from "another classifier" into a **rigorous generalization study**:
does a per-cell cardiomyopathy classifier hold up on a patient it has never seen, and
on an independent cohort? The honest (lower) patient-level number *is the result*.

### Interview narrative
"My first version leaked — SMOTE before the split, and CV on the resampled matrix —
which inflates AUC. I fixed the leakage, then re-evaluated at the **patient level**
(StratifiedGroupKFold on donor) instead of the cell level, and the AUC dropped. That
drop is the finding: per-cell classifiers overstate accuracy. Then I tested whether
the DCM/ACM signature transfers to an independent cohort."

## Data

**Primary (Option A):** Reichart et al. 2022, *Science* — "Pathogenic variants damage
cell composition and single-cell transcription in cardiomyopathies." 880k nuclei,
18 control + 61 failing hearts. The only public atlas whose `disease` obs field carries
both "dilated cardiomyopathy" and "arrhythmogenic right ventricular cardiomyopathy".
Download as `.h5ad` from CZ CELLxGENE collection `e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb`.

**Second cohort (Option B):** Chaffin et al. 2022, *Nature* DCM/HCM atlas (GSE183852),
also on CELLxGENE, for cross-cohort validation.

Both downloads are scripted and cached so the pipeline is reproducible from scratch.

## Architecture

Refactor the monolith into focused, independently testable modules:

```
cardiomyopathy-ml/
├── data.py            # reproducible CELLxGENE download; load + filter to DCM/ACM;
│                      # expose expression matrix, disease labels, and donor/patient IDs
├── preprocessing.py   # HVG selection + scaling as sklearn-compatible pipeline steps
├── evaluation.py      # StratifiedGroupKFold donor-level CV; metrics; naive-vs-honest
│                      # comparison; SHAP interpretation
├── models.py          # imblearn Pipeline(s): SMOTE -> scale -> classifier (RF/XGB/LogReg)
├── run_experiment.py  # CLI entry point; ties it together; saves results + plots
├── tests/             # pytest: data loads with expected columns; no-leakage checks;
│                      # probabilities in [0,1]; no NaNs after scaling
├── requirements.txt   # pinned versions
└── README.md          # honest narrative + results table
```

All dead code (transformer, GSEA, PDF, attention plots) is removed. A deep-learning
branch may be reintroduced later (Option C) only as a real, honestly-evaluated model.

## Data flow (Option A — honest baseline)

1. **Download** Reichart atlas from CELLxGENE (cached locally).
2. **Load + filter** to cells labeled DCM or ACM; retain the donor/patient ID column.
3. **Labels:** `y = disease`, `groups = patient_id`. The `groups` array is the crux.
4. **Split:** `StratifiedGroupKFold` on `groups` — no patient appears in both train and
   test; `Stratified` keeps the DCM:ACM ratio balanced across folds.
5. **Per fold (all fit on the training fold only, to avoid leakage):**
   HVG selection → SMOTE → StandardScaler → fit classifier → evaluate on the untouched
   validation fold. HVG selection is inside the fold because "which genes are highly
   variable" is itself a decision learned from data.
6. **Report** ROC-AUC per fold (mean ± std), confusion matrix, and SHAP top genes —
   all at the patient level.

## Class imbalance

SMOTE, but corrected: wrapped in an `imblearn.pipeline.Pipeline` so it resamples the
training fold only, per fold. This preserves a clean "found and fixed the leakage bug"
story. **Known discussion point:** SMOTE on high-dimensional scRNA-seq is debatable
(interpolating gene-expression vectors can invent biologically implausible cells);
`class_weight='balanced'` is a defensible alternative. We keep SMOTE as primary and are
prepared to discuss the tradeoff.

## What we measure — the deliverable

A comparison table that quantifies the leakage:

| Evaluation method            | Simulates              | Expected AUC     |
|------------------------------|------------------------|------------------|
| Random cell split (original) | recognizing seen patients | ~0.9 (inflated) |
| Patient-level split (honest) | a new patient          | lower — the finding |

Both are computed on purpose so the drop is visible and quantified. Option B
(train on Reichart, test on Chaffin) extends this to "does it transfer across cohorts?"

## Error handling

- Fail fast with clear messages if the expected `disease` / donor-ID columns are absent
  (validate at the data boundary rather than crashing deep in the pipeline).
- Replace the original single catch-all `try/except` with targeted handling.
- CLI args (data path, n_cells, seed) via `argparse`; `logging` instead of `print`.

## Testing

- Data loads and exposes the expected columns.
- **No-leakage check:** assert no patient ID appears in both train and test folds.
- Model outputs probabilities in [0, 1]; no NaNs after scaling.
- Target: cover the pipeline's load-bearing logic (splitting, preprocessing steps).

## Out of scope (YAGNI)

- The transformer / deep-learning branch (possible future Option C).
- GSEA, PDF report generation, gene correlation networks.
- Hyperparameter search beyond a small, honest grid (or documented pre-tuned params).

## Sequence

1. Option A: reproducible download → leakage-free, donor-level pipeline → naive-vs-honest
   comparison table.
2. Option B: cross-cohort validation on the Chaffin atlas.

## Model handoff

Design + implementation plan authored on Opus; coding/testing/debugging on Sonnet after
the plan is committed.
