# DCMvsACM — Full Project Roadmap

**Date:** 2026-07-03
**Purpose:** The master, far-horizon plan. Every phase below is ordered so each one
earns the right to the next. Phases A–C are the committed core; D onward are the
"where this can actually go" arc — increasingly ambitious, increasingly original,
each one a real contribution rather than a resume line.

Read this as the answer to *"where is this going and why does it matter?"* — the
question an interviewer, a PI, or a hiring manager will eventually ask.

---

## The through-line (say this out loud until it's yours)

> "I started with a binary classifier for two cardiomyopathies from single-cell RNA-seq.
> I discovered my own evaluation was leaking and inflating accuracy. Fixing that led me
> to a bigger question — *do these models generalize to unseen patients and unseen
> cohorts at all?* — and that question reshaped the whole project into a study of
> **honest evaluation and generalization in single-cell disease classification**."

Everything below serves that through-line. The value is not the model. The value is
the rigor and the questions.

---

## PHASE A — Honest baseline (COMMITTED, build first)

**Question:** Does a per-cell DCM-vs-ACM classifier survive a patient-level split?

**Work:**
- Reproducible CELLxGENE download (`data.py`).
- Refactor monolith → `data / preprocessing / evaluation / models / run_experiment`.
- Leakage-free pipeline: `StratifiedGroupKFold(donor)` → per-fold HVG → SMOTE → scale → fit.
- Deliverable: the **naive-vs-honest comparison table** (random-cell AUC vs patient-level AUC).
- Tests: no-patient-in-both-folds assertion; probabilities in [0,1]; no NaNs.

**Why it matters:** This is the credibility foundation. Without it, everything after is
built on sand. The AUC *drop* is the first real finding.

**Interview line:** "I quantified how much cell-level splitting inflates AUC — [X] points."

**Personal growth:** You will deeply internalize data leakage, group-aware CV, and
sklearn/imblearn pipelines. This is the single most transferable ML skill you'll gain.

---

## PHASE B — Cross-cohort generalization (COMMITTED, build second)

**Question:** Does the DCM signature transfer to a *different* study's patients/platform?

**Work:**
- Add Chaffin 2022 (GSE183852) as a second cohort via CELLxGENE.
- Harmonize gene identifiers between cohorts (intersection of gene sets; consistent IDs).
- Train on Reichart, test on Chaffin (and vice-versa). Report transfer AUC.
- Handle batch effects honestly — document, don't hide, the domain shift.

**Why it matters:** Cross-cohort transfer is the real-world test. Most published
biomarker papers never do it. If it transfers, that's strong; if it doesn't, *that's
also a publishable, honest result* about batch effects dominating biology.

**Interview line:** "In-cohort it looked great; across cohorts it dropped to [Y] —
batch effects, not biology, were doing a lot of the work."

**Personal growth:** Batch correction, domain shift, gene-ID harmonization — the messy
reality of multi-dataset ML that textbooks skip.

---

## PHASE C — Deep learning done honestly (COMMITTED, build third)

**Question:** Does a neural model beat gradient-boosting on this task, under the *same*
honest evaluation?

**Work:**
- Revive the transformer/MLP as a *real* branch, not dormant scaffolding.
- Same `StratifiedGroupKFold(donor)` protocol — no special treatment.
- Compare fairly: XGBoost vs MLP vs transformer, identical splits, identical metrics.
- Report honestly if the tree model wins (it often does on tabular scRNA-seq).

**Why it matters:** Demonstrates you can build DL *and* the maturity to say "it didn't
help here." That maturity is rarer and more valuable than the model itself.

**Interview line:** "I tested whether a transformer beat XGBoost under identical honest
CV. It didn't, and I can explain why — [tabular, per-cell, no sequence structure]."

**Personal growth:** PyTorch training loops, attention, and — crucially — *fair model
comparison* discipline.

---

## PHASE D — From binary to the full disease landscape

**Question:** Why only DCM vs ACM? The atlas has HCM, controls, and more. Can we build a
multi-class cardiomyopathy classifier — and where does it confuse diseases?

**Work:**
- Extend labels to multi-class (DCM / ACM / HCM / control / ...).
- Multi-class metrics: macro-AUC, per-class confusion, one-vs-rest curves.
- **The interesting part:** *which diseases get confused with which?* A confusion
  structure that mirrors known clinical/biological overlap is itself a finding.

**Why it matters:** Moves from a toy binary to a clinically-shaped problem. The confusion
matrix becomes a map of disease similarity.

**Interview line:** "DCM and HCM were most confusable — consistent with their shared
final-common-pathway heart-failure biology."

**Personal growth:** Multi-class evaluation, class imbalance at scale, reading confusion
structure as biology.

---

## PHASE E — Genotype-aware analysis (the paper's actual point)

**Question:** Reichart 2022 is *about genotype* (TTN, LMNA, PKP2, DSP, RBM20...). Does the
model's difficulty track genotype? Are some genetic subtypes harder to classify?

**Work:**
- Pull the pathogenic-variant / genotype annotations from the atlas metadata.
- Stratify performance by genotype. Do LMNA patients misclassify more than TTN?
- Ask: is "DCM vs ACM" really "genotype A vs genotype B" in disguise?

**Why it matters:** This is where the project stops being a generic classifier and starts
engaging with the actual science of the dataset. Deeply original angle.

**Interview line:** "I found the classifier was really separating genotypes, not clinical
labels — which reframes what the model is 'diagnosing'."

**Personal growth:** Connecting ML behavior to underlying causal biology; skepticism about
what a label actually encodes.

---

## PHASE F — Which cell type carries the signal?

**Question:** The signal isn't uniform across cell types. Do cardiomyocytes, fibroblasts,
endothelial cells, or immune cells best distinguish the diseases?

**Work:**
- Train separate classifiers per cell type (using the atlas cell-type annotations).
- Rank cell types by how diagnostic they are.
- Cross-reference with known biology (e.g., fibrosis → fibroblasts in DCM).

**Why it matters:** Turns the model into a *scientific instrument* — it tells you where
in the tissue the disease signature lives.

**Interview line:** "Fibroblasts were the most diagnostic cell type, consistent with the
fibrotic remodeling literature."

**Personal growth:** Cell-type-aware analysis, subsetting AnnData, biological interpretation.

---

## PHASE G — Pseudobulk vs single-cell (a methods contribution)

**Question:** Do we even need single-cell resolution? Would aggregating each patient to a
"pseudobulk" profile classify just as well — with far fewer leakage traps?

**Work:**
- Build pseudobulk (mean/sum expression per patient) → classify at the *patient* level.
- Compare: single-cell (many rows/patient) vs pseudobulk (one row/patient).
- This is the cleanest possible answer to the leakage problem — n = patients, not cells.

**Why it matters:** A genuine methodological insight. If pseudobulk matches single-cell,
you've shown the single-cell complexity was unnecessary for *classification* — a strong,
counterintuitive, defensible claim.

**Interview line:** "Pseudobulk at the patient level matched single-cell AUC — the honest
n was always ~79 patients, not 880k cells, and pretending otherwise was the whole trap."

**Personal growth:** This is the conceptual capstone of the leakage story. Master this and
you understand the project better than most people who'd interview you.

---

## PHASE H — Biological validation & interpretability that means something

**Question:** Are the model's top genes *real* disease biology or artifacts?

**Work:**
- SHAP / coefficients → top genes.
- Cross-reference against curated cardiomyopathy gene panels (ClinGen, known DCM/ACM genes).
- Enrichment (GSEA/enrichr) — but only as validation of the classifier, not decoration.
- Report overlap: "N of top 20 are known cardiomyopathy genes (vs chance)."

**Why it matters:** Closes the loop from prediction back to biology. Guards against the
"model learned batch/technical noise" failure mode.

**Interview line:** "My top features overlapped known pathogenic genes above chance —
evidence the model learned biology, not batch."

**Personal growth:** Interpretability with a validation target; distinguishing signal from
artifact.

---

## PHASE I — Uncertainty & calibration (does it know when it doesn't know?)

**Question:** When the model is wrong, is it confidently wrong? Can it abstain?

**Work:**
- Calibration curves (reliability diagrams), Brier score.
- Confidence-vs-correctness; a "reject option" for low-confidence cells/patients.
- Especially interesting for the *cross-cohort* setting (Phase B) where it should be
  *uncertain* on out-of-distribution data — and usually isn't.

**Why it matters:** Clinical ML that can't express uncertainty is dangerous. This is a
maturity signal that separates you from "accuracy is all that matters" candidates.

**Interview line:** "On the out-of-cohort test it was confidently wrong — poorly
calibrated under domain shift, which is the real clinical risk."

**Personal growth:** Calibration, uncertainty quantification, OOD detection.

---

## PHASE J — Foundation models (the frontier stretch)

**Question:** Do single-cell foundation models (scGPT, Geneformer, scFoundation) beat
classical ML on this task under honest evaluation?

**Work:**
- Fine-tune or extract embeddings from a pretrained single-cell foundation model.
- Feed embeddings into the *same* honest donor-level protocol.
- Fair comparison vs XGBoost baseline.

**Why it matters:** Puts you at the current research frontier of the field. Even a
negative result ("foundation model didn't beat XGBoost under honest CV") is a strong,
current, defensible contribution — and shows you can wield the latest tools critically.

**Interview line:** "I benchmarked a single-cell foundation model against XGBoost under
identical patient-level CV — here's where it helped and where it didn't."

**Personal growth:** Transfer learning, embeddings, working with large pretrained models,
critical benchmarking of hype.

---

## PHASE K — Make it real: demo, model card, reproducibility

**Question:** Can someone else run this, understand it, and trust it in 10 minutes?

**Work:**
- A one-command reproduce (`make all` / script): download → run → figures → report.
- A **model card** (intended use, data, metrics, limitations, the honest AUCs).
- Optional small demo (Streamlit/Gradio): upload a patient's profile → prediction +
  uncertainty + top genes.
- Clean README with the results table front and center.

**Why it matters:** The difference between "a script" and "a project someone can trust."
Reproducibility and honest documentation are exactly what "not another Claude project"
looks like from the outside.

**Interview line:** "It reproduces end-to-end from one command, with a model card that
states the limitations up front."

**Personal growth:** Reproducible ML engineering, documentation, packaging, deployment.

---

## PHASE L — Write it up (the multiplier)

**Question:** Who learns from this besides you?

**Work:**
- A blog post / technical writeup: "How I discovered my cardiomyopathy classifier was
  cheating — a case study in data leakage and honest evaluation."
- Optionally a short preprint / notebook-as-paper.
- The narrative *is* the leakage story: naive → fixed → patient-level → cross-cohort →
  pseudobulk. That arc teaches something. Publishing it makes you the person who found it.

**Why it matters:** Writing forces complete understanding and multiplies the project's
reach. A clear writeup of a subtle bug is more impressive than a flashy model.

**Interview line:** "I wrote it up because the leakage lesson generalizes far beyond
cardiology — anyone doing grouped data ML hits it."

**Personal growth:** Technical communication — the skill that compounds every other one.

---

## Dependency graph (what unlocks what)

```
A (honest baseline)  ──►  B (cross-cohort)  ──►  I (calibration under shift)
   │                        │
   ├──►  C (deep learning)  ┤
   ├──►  D (multi-class)    │
   ├──►  E (genotype)       │
   ├──►  F (cell-type)      │
   ├──►  G (pseudobulk) ◄───┘   ← conceptual capstone of the leakage story
   │
   └──►  H (bio validation)  ──►  J (foundation models)  ──►  K (demo)  ──►  L (writeup)
```

- **A is the gate.** Nothing is trustworthy until A is done.
- **G (pseudobulk)** is the intellectual high point of the core arc — do it right after B.
- **L (writeup)** can begin as soon as A+B exist; keep a running lab notebook from day one.

## Minimum impressive slice (if time is short)

**A + B + G + K + L.** That's: honest baseline, cross-cohort, the pseudobulk insight, a
reproducible demo, and a writeup. That alone is a standout portfolio project and a
complete, defensible story. C/D/E/F/H/I/J are the "keep going until you crack it"
extensions you asked for — pursue them in whatever order excites you, because genuine
curiosity is what stops this from reading as generated.

## The one rule that keeps it yours

At every phase, before moving on, you must be able to answer three questions in your own
words: **What did I measure? What did I find? Why is it true?** If you can't, we slow down
until you can. That constraint — not the code — is what makes "I did this" honest.
