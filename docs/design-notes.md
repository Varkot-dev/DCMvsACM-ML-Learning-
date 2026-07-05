# Interview Notes — DCMvsACM Generalization Study

This document is your study guide. Read it before interviews. Every section
answers a question an interviewer might ask. The goal: you can explain every
decision in your own words, not just recite it.

---

## What is this project, in one sentence?

"I took a single-cell RNA-seq classifier for two heart diseases, discovered the
evaluation was leaking patient data across the train/test boundary, fixed it, and
turned it into a study of whether these classifiers actually generalize to
unseen patients."

---

## The biology (what you're classifying)

**Dilated Cardiomyopathy (DCM):** The heart muscle is weakened and dilated — the
left ventricle stretches and can't pump efficiently. Causes: mutations in structural
proteins (TTN, the largest human gene), nuclear lamina (LMNA), RNA binding proteins
(RBM20). The chamber looks like a stretched balloon.

**Arrhythmogenic Right Ventricular Cardiomyopathy (ARVC/ACM):** Fibro-fatty tissue
replaces the right ventricular muscle, causing arrhythmias. Causes: mutations in
desmosomal proteins (PKP2, DSP, DSG2) that hold cardiomyocytes together. Named for
the right ventricle but can involve the left.

**Why these two are an interesting classification problem:**
Both cause heart failure. Both can present similarly. But their molecular mechanisms
are distinct. If a classifier can separate them from *gene expression alone*, it means
the transcriptome captures the disease etiology — not just the end-state failure.
That's biologically meaningful.

**The dataset:** Reichart et al. 2022, *Science*. 880k nuclei from 79 hearts (18
control + 61 failing). Single-nucleus RNA-seq of left ventricular tissue. Key columns
in the metadata: `disease` (the label), `donor_id` (the patient), `cell_type`
(cardiomyocyte, fibroblast, endothelial, etc.), `gene` (pathogenic variant).

---

## The original bug — and why it matters

The original `dcm_vs_acm.py` had this evaluation order:
```
SMOTE(whole dataset) → train_test_split → cross_val_score(X_balanced)
```

There are two leaks here:

### Leak 1: SMOTE before the split
SMOTE (Synthetic Minority Over-sampling Technique) creates new minority-class cells
by interpolating between real ones. If a real ACM cell A and real ACM cell B are
in the dataset, SMOTE creates a synthetic cell S halfway between them.

If A ends up in training and B ends up in the test set, S might be in either —
and S is derived from B. So you're testing on a cell that's derived from a cell
in your test set. The model partially "recognises" the test set because it trained
on synthetic cousins of it. AUC is inflated.

**The fix:** split first, SMOTE only on the training fold.

### Leak 2: cross_val_score on X_balanced
After SMOTE on the whole dataset, `cross_val_score(X_balanced, y_balanced, cv=...)`.
Every fold's validation set contains SMOTE'd cells whose real "parents" are in the
training fold. The cross-validation is measuring performance on data that's derived
from the training data. Circular.

**The fix:** put SMOTE inside an `imblearn.Pipeline`, which runs it per-fold,
training-fold-only.

### Leak 3: Cell-level splitting (the biggest one)
Even without SMOTE leakage, the original `StratifiedKFold` splits cells randomly.
Patient 34 has 5,000 cells; 4,000 go to training, 1,000 go to test.

The model learns patient 34's cell-to-cell correlations, their batch signature,
their specific genetic background — thousands of signals that aren't disease biology.
When it sees patient 34's other 1,000 cells in the test set, it partially recognises
them. AUC is inflated relative to a model that's never seen the patient at all.

A real clinical classifier must work on a *new patient who walks in tomorrow*.
The honest evaluation is: put ALL of patient 34's cells in training OR in test —
never both. That's `StratifiedGroupKFold`.

**Expected effect:** AUC drops. That drop quantifies the leakage.

---

## The key class: StratifiedGroupKFold

**sklearn.model_selection.StratifiedGroupKFold** (added in sklearn 1.0):

- `Group` = keep all cells from the same patient in the same fold. Pass `groups=donor_id_array`.
- `Stratified` = maintain the DCM:ACM ratio across folds, so no fold is all-DCM.

You need BOTH:
- Without `Group`: patient leakage.
- Without `Stratified`: a fold could be 90% DCM, making AUC on that fold meaningless.

The call signature: `cv.split(X, y, groups=groups)` — the `groups` parameter is the patient ID array.

---

## Why HVG selection must go inside the fold

Highly Variable Gene (HVG) selection asks: "which genes vary the most across cells?"
It's computed from the data — specifically from expression variance across all cells.

If you select HVGs from the *whole dataset* (train + test cells combined), you've
used information from the test cells to decide which features to use. That's a form
of leakage — the test cells have influenced the feature space.

The fix: put `HVGSelector` inside the pipeline, so it's fit only on the training fold.
This is subtle and most published pipelines miss it. Mentioning it in an interview
signals senior-level understanding.

Our `HVGSelector` uses the **Fano factor** (variance / mean) to rank genes. This is
the gene expression equivalent of coefficient of variation — it's numerically stable
and doesn't depend on the data being count-normalised to a specific range (unlike
scanpy's algorithms).

---

## SMOTE: is it even appropriate for scRNA-seq?

**The argument for SMOTE:**
Class imbalance (more DCM than ACM cells) biases classifiers toward the majority class.
SMOTE creates synthetic minority-class cells by interpolating between real ones,
giving the classifier more ACM examples to learn from.

**The argument against SMOTE for scRNA-seq:**
Gene expression is not a smooth Euclidean space. A synthetic cell created by averaging
two real ACM cells might have an expression profile that no real cell would ever have —
biologically implausible. Interpolating in 2,500-dimensional gene space is risky.

**The alternative:** `class_weight='balanced'` in the classifier. This doesn't create
fake cells — it just tells the model to penalise errors on the minority class more.
Mathematically equivalent to over-sampling but without the fake-biology concern.

**What to say in an interview:**
"I used SMOTE because it's what the original code used and it let me demonstrate the
leakage fix clearly. But I'm aware that SMOTE on high-dimensional gene expression
is debatable — interpolating in 2500-gene space can produce biologically implausible
profiles. An alternative I'd explore is `class_weight='balanced'`, which avoids
synthesising cells entirely."

---

## The pipeline architecture and why it's correct

```
imblearn.Pipeline([
    ("hvg", HVGSelector(n_top_genes=2500)),   # fit on train fold only
    ("scaler", StandardScaler()),              # fit on train fold only
    ("smote", SMOTE(k_neighbors=5)),           # fit on train fold only, skipped at predict time
    ("clf", XGBClassifier()),                  # fit on SMOTE'd train fold
])
```

**imblearn's Pipeline is not sklearn's Pipeline.** The key difference: imblearn's Pipeline
knows that `SMOTE` is a resampler — it runs it during `fit()` but skips it during
`predict()` and `transform()`. This is exactly correct: you want to balance the training
data, but at prediction time you don't want to synthesise cells.

Each call to `cv.split()` produces a new train/val pair. We `clone()` the pipeline for
each fold so there's no state contamination between folds. Then we `fit()` the whole
pipeline on the training fold and `predict_proba()` on the untouched validation fold.

---

## The deliverable: the comparison table

```
NAIVE (random cell split — inflated):
  XGBoost    AUC 0.93 ± 0.02

HONEST (patient-level split):
  XGBoost    AUC 0.71 ± 0.08

LEAKAGE INFLATION:
  XGBoost    +0.22 AUC   (naive - honest)
```

*(These numbers are illustrative — actual values come from running on real data.)*

The gap (+0.22 in this example) is how much the original code was "cheating."
It's not a failure of the honest model — it's the honest measurement of what
the classifier can actually do on patients it hasn't seen.

---

## Things that will come up in an interview

**Q: Why not just use more data?**
A: More cells from the same patients doesn't help — you still have the same number
of patients. The effective sample size is patients, not cells. This is the core
insight. Pseudobulk (Phase G) is the logical conclusion: aggregate each patient
to one vector and classify at n=79.

**Q: What's the expected AUC drop?**
A: In the literature, cell-level to patient-level drops of 10-25 AUC points are common
for scRNA-seq classifiers. If the drop is small, it means the signal is genuinely
biological (not patient-specific). If the drop is large, batch/patient effects dominate.
Either result is interesting.

**Q: Why XGBoost over a neural network?**
A: For tabular scRNA-seq (cells × genes matrix), gradient boosting consistently
matches or outperforms neural networks. scRNA-seq cells don't have sequence
structure, so transformers have no natural inductive bias here. We test a
transformer in Phase C specifically to show this empirically — "I tested and
it didn't help, here's why" is a stronger statement than "I heard trees win."

**Q: Why StratifiedGroupKFold and not leave-one-patient-out?**
A: Leave-one-out gives you n=79 folds (one per patient), which is very high
variance. 5-fold gives ~16 held-out patients per fold — enough to estimate
AUC reliably and far faster to compute. Leave-one-patient-out is the most
honest possible, but computationally expensive and high-variance.

**Q: Is DCM vs ACM actually a hard classification problem?**
A: That's the right question. If the honest AUC is 0.95, the expression
signature is very robust. If it's 0.6, the diseases are transcriptomically
similar (they may share a common heart-failure endpoint). Either is publishable
and interesting — we report what we find.

---

## What you built — file by file

| File | What it does | Key decision |
|------|-------------|--------------|
| `data.py` | Load, filter, validate, extract arrays | Validate at boundary; fail loudly |
| `preprocessing.py` | HVGSelector (Fano factor) + StandardScaler | Sklearn-compatible transformers; fit-on-train-only |
| `models.py` | imblearn Pipeline builder for RF/XGB/LogReg | SMOTE inside pipeline = no leakage |
| `evaluation.py` | naive vs honest CV; FoldResult/EvalResult dataclasses; comparison table | StratifiedGroupKFold; explicit leakage-inflation row |
| `run_experiment.py` | CLI entry point; ties everything together; saves CSV | `--synthetic` flag for testing without real data |
| `tests/test_pipeline.py` | 16 tests including `test_no_patient_leaks_across_folds` | The leakage test is the guardian rail |

---

## Git history (the story)

```
chore: initial baseline — existing ML pipeline and gitignore
docs: add generalization-study design spec
docs: add full 12-phase project roadmap (A through L)
feat: implement Phase A — leakage-free donor-level evaluation pipeline
feat: add Phase A modules and tests to git tracking
```

The commit history tells the same story as the project: here's where we started,
here's what we planned, here's what we built and why it's correct.

---

## What's next (Phases B–L)

When you've run the real data and have the comparison table numbers, continue:

- **Phase B:** Train on Reichart, test on Chaffin 2022 (cross-cohort). Does it transfer?
- **Phase G:** Pseudobulk — aggregate each patient to one vector, classify at n=79.
  This is the conceptual capstone of the leakage story.
- **Phase H:** Take top SHAP genes, cross-reference against known DCM/ACM genes.
  Do the top features overlap cardiomyopathy pathogenic genes above chance?
- **Phase L:** Write the blog post. "How I discovered my heart disease classifier was
  cheating." That's the piece that makes this project remembered.
