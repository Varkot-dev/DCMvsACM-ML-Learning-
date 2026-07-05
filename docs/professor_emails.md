# Professor Outreach — Draft Emails

These are cold emails to Penn faculty doing cardiac genomics. Goal: access to a larger ACM cohort, or collaboration on the evaluation framework paper. Send one at a time, personalize the bracketed sections before sending.

---

## Target 1 — Dr. Sharlene Day (Penn CVI, genetic cardiomyopathies)

**Why her:** Runs the Penn Center for Inherited Cardiac Disease. Sees ACM patients clinically. Has published on genetic heart muscle diseases. Most likely to have patient samples or know who does.

**Email:**

Subject: Single-cell ML evaluation framework for ACM/DCM — potential collaboration

Dr. Day,

I'm Varshith Kotagiri, a sophomore at Penn studying [your major]. I've been working independently on a computational analysis of the Reichart 2022 single-cell cardiac atlas, focused on classifying dilated versus arrhythmogenic cardiomyopathy.

The main finding is methodological: standard cross-validation in single-cell ML inflates AUC by 13–18 points by leaking patient identity across folds. After fixing the evaluation to respect patient boundaries, honest AUC drops from 0.88 to 0.70. A follow-up pseudobulk differential expression analysis at the patient level found no genes surviving FDR correction, which I think correctly reflects the statistical reality of n=8 ACM donors rather than a biological absence of signal.

The most interesting raw hit is PDGFB (log₂FC = −1.78, higher in ACM, p = 0.00028 uncorrected) — consistent with ACM's fibro-fatty pathology — but it doesn't survive multiple testing at this sample size.

I'm writing because the analysis is statistically limited by the eight ACM patients in Reichart. I'd like to ask whether your lab has access to additional ACM single-cell or bulk RNA-seq data, or whether you'd be interested in discussing the evaluation framework. The code and results are on GitHub: [your repo URL].

I'm happy to come by during office hours or whenever works for you.

Best,
Varshith Kotagiri
University of Pennsylvania, Class of 2029
varkot06@sas.upenn.edu

---

## Target 2 — Zoltan Arany Lab (Penn CVI, cardiovascular metabolism)

**Why him:** Arany Lab at Penn CVI does metabolic and transcriptomic profiling of heart failure. Published snRNA-seq work on cardiac remodeling. Less directly focused on ACM but has the infrastructure and dataset access.

**Email:**

Subject: Honest evaluation framework for single-cell cardiomyopathy classification

Dr. Arany,

I'm Varshith Kotagiri, a sophomore at Penn. I've been building a machine learning pipeline for classifying dilated versus arrhythmogenic cardiomyopathy from the Reichart 2022 single-cell atlas.

The core finding is about evaluation methodology: the standard approach of randomly splitting cells across CV folds inflates AUC by 13–18 points because it lets the model memorize patient-specific expression patterns rather than learning disease biology. Correcting for patient boundaries drops performance from 0.88 to 0.70 AUC. I then ran pseudobulk differential expression at the patient level and found no genes surviving FDR correction across 32,383 genes at n=8 ACM donors — which I think is the honest statistical answer at that sample size.

I'm trying to figure out whether adding patients changes the picture. I'm aware of the 2024 BMC Medicine ARVC dataset (6 end-stage patients, Huang et al.) and am planning to test cross-cohort generalization. If your lab has cardiac single-cell data from heart failure patients — particularly any with known genetic diagnoses — I'd be interested in discussing whether there's a way to extend the analysis.

The repo is here: [your repo URL]. Happy to share more detail or come by the lab.

Best,
Varshith Kotagiri
University of Pennsylvania, Class of 2029
varkot06@sas.upenn.edu

---

## Target 3 — Cold email to corresponding author of BMC Medicine 2024 ARVC paper

**Why:** They have a separate ACM single-cell dataset (6 ARVC patients). Combining with Reichart gets to 14 ACM donors. This is the most direct path to more data.

**Find the paper:** "Single-cell RNA sequencing in donor and end-stage heart failure patients identifies NLRP3 as a therapeutic target for arrhythmogenic right ventricular cardiomyopathy" — BMC Medicine, 2024. Find the corresponding author and their email on the paper.

**Email:**

Subject: Data sharing inquiry — ARVC single-cell dataset (BMC Medicine 2024)

Dear Dr. [Name],

I read your 2024 BMC Medicine paper on single-cell RNA sequencing in ARVC with great interest. I'm Varshith Kotagiri, a sophomore at the University of Pennsylvania, working independently on a machine learning evaluation framework for DCM vs ACM classification using the Reichart 2022 cardiac atlas.

The main constraint in my analysis is sample size — Reichart has 8 ACM patients, which limits statistical power at the patient level. Your dataset (6 end-stage ARVC patients) would roughly double the ACM cohort available for analysis. I'm writing to ask whether your single-cell data is publicly deposited or whether there's a data sharing pathway I could pursue.

My current results and code are at [your repo URL]. I'm happy to share more detail about the analysis, discuss a potential collaboration, or work through whatever data access process your institution requires.

Thank you for considering this.

Varshith Kotagiri
University of Pennsylvania, Class of 2029
varkot06@sas.upenn.edu

---

## Notes before sending

- Personalize the bracketed sections
- Add the GitHub repo URL throughout
- For Target 3: find the paper at https://bmcmedicine.biomedcentral.com/articles/10.1186/s12916-023-03232-8 and get the corresponding author's name and email from the paper itself
- Send one per day, not all at once
- If no response in two weeks, one follow-up is appropriate
