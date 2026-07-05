# DCMvsACM: Substate DE, Cross-Cohort Transfer, and Calibration Under Shift

Spec date: 2026-07-06. Follows the generalization-study design (2026-07-03) and roadmap phases G, H, K.
Written after a literature review that repositioned the vCM3.0 finding. Read the "Framing" section first, it changes what we are allowed to claim.

## Framing (what the literature review settled)

The vCM3.0 substate is not a novel discovery. It is the established "stressed ventricular cardiomyocyte" state
from the adult human heart atlas (Litvinukova 2020, Nature), canonically marked by ANKRD1, XIRP1/XIRP2, FHL1.
We rediscovered its markers independently, which validates our pipeline but is not itself a claim.

What is ours, and defensible:
1. DCM vs ACM classification signal concentrates in that stressed substate (0.775 vs 0.726 whole population).
   Not found reported elsewhere.
2. FLNC causes both DCM and ACM but has never been compared between the two at single-cell resolution.
   The one FLNC-ACM transcriptomic paper (Brun 2020) is bulk, 7 patients, ACM-vs-control only. Real gap.

Honest-evaluation finding (0.88 naive to 0.70 patient-level) remains the strongest asset. Do not let the
biology angle dilute it.

## Ordering and why (the load-bearing logic)

Phase 1 first because it is cheap, self-contained, and answers a clean biology question regardless of what
the cross-cohort gate shows. Phase 2 is the gate: Phase 3 (calibration) has nothing to calibrate if the
signature does not transfer. Gate means "Phase 3 depends on it," not "it runs first."

## Phase 1: FLNC-within-vCM3.0 differential expression

Question: within the vCM3.0 stressed substate only, do DCM and ACM differ, and is FLNC among the differences?

Method (mirrors the July 5 pseudobulk approach, but subset to vCM3.0 first):
- Subset cardiomyocytes to cell_states == "vCM3.0" (21,084 cells, all 60 donors present, confounds already
  checked equal on 2026-07-06).
- Pseudobulk per donor within the subset (sum counts per donor, then CPM/log). Donor-level, not cell-level.
  Per-cell DE lies here, established July 5.
- Wilcoxon rank-sum DCM vs ACM per gene, BH-FDR. Expression filter first (mean > 0.05 in at least one group),
  the KCNA10/OR8K3 division-artifact mistake from 2026-07-06 must not recur.
- Report: does FLNC (and FLNC-AS1) appear? Where does the broader stress program (ANKRD1, XIRP2) rank in the
  DCM-vs-ACM contrast specifically, as opposed to the substate-vs-rest contrast that produced the 1,903 genes.

Two honest outcomes, both fine:
- FLNC/program separates the diseases inside the stressed state: novel, worth the professor emails.
- It does not: the substate AUC advantage comes from something else. Protects against over-claiming.

Compute: laptop, minutes. Reuses cardiomyocytes.h5ad. New script, does not touch existing pipeline.

Caveat to surface in the writeup: 8 ACM donors is the same sample-size ceiling that killed whole-atlas
pseudobulk DE (zero genes survived, twice confirmed). Subsetting to one substate reduces cells per donor,
it does not add donors. A null result may be power, not biology. Say so.

## Phase 2: cross-cohort signature transfer (option a)

The Chaffin 2022 atlas is DCM vs HCM. No ACM. So this cannot be a DCM-vs-ACM classifier transfer, there is
nothing to map the ACM half onto. Instead:

- Test whether the vCM3.0 stressed-state signature replicates in Chaffin (does an analogous stressed CM
  substate exist, marked by the same genes, expanded in disease).
- Test whether the DCM-associated component of our signal (the genes driving DCM in our classifier) behaves
  consistently on Chaffin's DCM cells.

This is a domain-shift experiment. Expect AUC to drop across cohorts. That drop is the lesson: batch effect,
platform difference, why a model trained on one cohort degrades on another. Learn integration
(scVI/harmony) as the mitigation, and measure how much it recovers.

Data: Chaffin 2022 (GSE183852), CELLxGENE. Download as h5ad, same path as the primary atlas.
Compute: medium. May need RunPod again for integration at scale.

Stretch goal (option b), documented not scheduled: a true second ACM cohort (2024 BMC ARVC paper, ~6 ACM
patients, in the draft outreach emails). Preserves the real DCM-vs-ACM boundary across cohorts but adds an
ACM-vs-ACM batch-effect fight on thin numbers. Only if outreach lands.

## Phase 3: calibration under shift (depends on Phase 2)

Once a model predicts on a cohort it was not trained on, ask whether its confidence is honest.
- Reliability diagram (predicted probability vs observed frequency).
- Expected Calibration Error before and after.
- Temperature scaling on a held-out slice, re-measure ECE.
- The interesting question: does calibration degrade specifically under the cross-cohort shift, and does
  temperature scaling fit on the source cohort survive the shift to the target.

Rare skill, natural capstone. Only meaningful with Phase 2 predictions in hand.

## Working agreement for this spec

User has approved proceeding. Teaching happens inline during implementation, not as upfront checkpoints.
Explain the ML reasoning as each piece is built so the user learns by doing and can defend it later.
No em dashes or en dashes anywhere. Commit often, conventional commits, journal each day under journal/.
Switch to Sonnet for the coding once this spec is committed.
