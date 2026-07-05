# July 4, 2026 — Phase A: Finding the Leakage

## What we did

Built the full honest evaluation pipeline from scratch and ran it on real data from the Reichart 2022 *Science* cardiac atlas (166,519 cells, 60 donors).

## The main discovery

Standard ML cross-validation on single-cell data is broken. When you split cells randomly, cells from the same patient end up in both training and validation. The model memorizes the patient, not the disease. We fixed this and measured the damage.

| Classifier | Naive AUC | Honest AUC | Inflation |
|------------|-----------|------------|-----------|
| XGBoost    | 0.883     | 0.701      | **+0.182** |
| Random Forest | 0.834  | 0.705      | **+0.129** |

## Three bugs we fixed

**Bug 1 — Cell-level CV:** Used `StratifiedGroupKFold(groups=donor_id)` so all cells from one patient stay in the same fold.

**Bug 2 — SMOTE before split:** Moved SMOTE inside an `imblearn.Pipeline` so synthetic cells are only generated from training data.

**Bug 3 — HVG selection before split:** Made HVGSelector a pipeline step that fits only on training folds.

## Memory problem we solved

The full 166k × 32k matrix as dense = ~40 GB RAM. We wrote a sparse Fano-factor pre-selection in `data.py` that computes variance/mean on the sparse matrix before densifying — reducing peak RAM from ~40 GB to ~1.6 GB.

## Key biological insight of the day

8 ACM patients. That is the real constraint. No matter how many cells you have, statistical power is limited by independent donors. Fold 4 AUC collapsed to 0.57 because H29 — a patient with an LMNA mutation but ACM clinical diagnosis — landed in the validation set. The model learned LMNA=DCM from training, then saw H29 and got confused. Real biology, not a code bug.

## Figures

### figA_data_landscape.png
9-panel figure: cells per class, donors per class, donor size variability, gene detection, UMI distributions, genetic variants, honest vs naive AUC per fold, leakage inflation bars, donor size violin.

![Data landscape](figA_data_landscape.png)

### figB_decision_brief.png
The three design decisions laid out as open questions: genetic confound (PKP2 dominates ACM cohort), H29 special case (LMNA/ACM overlap), CV strategy options (nested vs fixed split vs leave-one-ACM-out).

![Decision brief](figB_decision_brief.png)

## What we learned

- AUC without patient-level evaluation is not a trustworthy number
- SMOTE and HVG selection must happen inside the CV loop
- 166,000 cells from 8 patients is still 8 patients
- The genetic confound (6/8 ACM patients carry PKP2) is invisible to AUC — you need to know the biology

## Code shipped

- `cardiomyopathy-ml/data.py` — sparse HVG pre-selection
- `cardiomyopathy-ml/models.py` — RF + XGBoost pipelines
- `cardiomyopathy-ml/evaluation.py` — naive vs honest CV
- `cardiomyopathy-ml/tests/test_pipeline.py` — 16 tests including patient-leak assertion
- `results/comparison_table.csv` — the main finding
- `results/fold_results.csv` — per-fold breakdown
