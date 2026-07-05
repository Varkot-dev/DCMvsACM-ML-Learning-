# Professor Outreach — Drafts

Method-forward: lead with the finding, then the ask. Personalise brackets. One per day. Attach or link 2–3 figures (leakage dumbbell, SHAP, pseudobulk volcano) — not the whole repo.

---

## 1 — Dr. Sharlene Day (Penn CVI · inherited cardiomyopathies)

Best fit: she runs the Penn Center for Inherited Cardiac Disease and sees ACM patients clinically.

**Subject:** Patient-level evaluation of DCM/ACM single-cell classifiers — a leakage result + a data question

Dr. Day,

I'm Varshith Kotagiri, a second-year at Penn. Working independently with the Reichart 2022 cardiac atlas, I found that the standard way single-cell classifiers are evaluated inflates performance substantially: splitting cells randomly across cross-validation folds lets a model recognise individual patients instead of learning disease, and it inflated DCM-vs-ACM AUC from 0.70 to 0.88 in my hands. Enforcing patient-level folds removes 13–18 AUC points.

Pushing further, a pseudobulk differential-expression analysis at the patient level (Wilcoxon, BH-corrected across 32,383 genes) returned no significant genes — which I read as an honest consequence of eight ACM donors rather than an absence of biology. The strongest raw signal was PDGFB, higher in ACM, which fits ACM's fibro-fatty pathology but doesn't survive correction at this n.

I'd value fifteen minutes of your time. Concretely, I'm hoping to learn whether your group has additional ACM single-cell or bulk RNA-seq data that could raise the donor count, and whether the evaluation problem is worth writing up. Three summary figures are attached; code and full notes are at [REPO URL].

Thank you for considering it.

Varshith Kotagiri
Penn Class of 2029 · varkot06@sas.upenn.edu

---

## 2 — Zoltan Arany Lab (Penn CVI · cardiovascular metabolism / snRNA-seq)

Fit: infrastructure and heart-failure snRNA-seq data, though less ACM-specific.

**Subject:** Leakage in single-cell cardiomyopathy classifiers — result + collaboration question

Dr. Arany,

I'm Varshith Kotagiri, a second-year at Penn. Analysing the Reichart 2022 atlas, I found that cell-level cross-validation inflates DCM-vs-ACM AUC by 13–18 points versus patient-level folds — the model is largely recognising patients, not disease. A patient-level pseudobulk analysis then found no genes surviving FDR across 32,383 tests at eight ACM donors, with PDGFB as the strongest uncorrected hit (higher in ACM; consistent with fibrosis).

The analysis is capped by sample size, so I'm looking for ways to add patients. If your lab has cardiac single-cell data from heart-failure donors — especially any with genetic diagnoses — I'd be glad to discuss whether the framework could extend to it. Figures attached; repo at [REPO URL].

Grateful for any pointer, even to a better-suited group.

Varshith Kotagiri
Penn Class of 2029 · varkot06@sas.upenn.edu

---

## 3 — Corresponding author, BMC Medicine 2024 ARVC paper

Their 6-patient ARVC dataset would roughly double the ACM cohort. Most direct path to more data.
Paper: https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-023-03232-8 — pull the corresponding author's name/email from it.

**Subject:** Data-sharing inquiry — your ARVC single-cell dataset (BMC Medicine 2024)

Dear Dr. [NAME],

I read your 2024 BMC Medicine study on single-cell profiling in ARVC closely. I'm Varshith Kotagiri, a second-year at Penn, building a patient-level evaluation framework for DCM-vs-ACM classification on the Reichart 2022 atlas. My main constraint is its eight ACM donors; your six ARVC patients would nearly double that and let me test cross-cohort generalisation.

Is your single-cell data publicly deposited, or is there a sharing pathway I could follow? I'm happy to share my results and discuss a joint analysis. Summary at [REPO URL].

Thank you,
Varshith Kotagiri
Penn Class of 2029 · varkot06@sas.upenn.edu

---

### Checklist before sending
- [ ] Fill [REPO URL] and [NAME]
- [ ] Attach figF_leakage, figD_shap, figE_pseudobulk (PNG)
- [ ] Confirm the BMC corresponding author from the paper
- [ ] Send one per day; single follow-up after ~2 weeks if silent
