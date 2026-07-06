"""
koenig_vcm3_replication.py — Phase 2: does the vCM3.0 stressed-cardiomyocyte
signature replicate in the Koenig/Lavine 2022 cohort (GSE183852)?

This substitutes for the Chaffin 2022 DCM/HCM atlas named in the spec (not
available via CELLxGENE — see journal 2026-07-06). Koenig/Lavine is DCM vs
non-diseased donor, not DCM vs HCM, so this tests replication of the stress
program in a DCM-vs-donor contrast, not DCM-vs-HCM.

Method (marker-score approach, not full reclustering — this cohort ships with
no cell-type/substate annotation, unlike Reichart's 'cell_states' column):
  1. Normalize + log1p raw counts.
  2. Score every cell for canonical cardiomyocyte markers (TTN, MYH6, MYH7,
     TNNT2, ACTC1). This score's distribution across all 269,794 cells is
     smooth and unimodal, not bimodal — there is no natural "cardiomyocyte vs
     not" gap to cut at, unlike a real annotated cell-type column. Any single
     threshold is therefore an arbitrary judgment call.
  3. Rather than commit to one threshold, sweep the CM-score cutoff across
     percentiles (50th, 60th, 70th, 80th, 90th, 95th) and repeat the DCM-vs-
     donor stress-score comparison at each. This is a robustness/sensitivity
     check: if the DCM-vs-donor difference in stress score is real biology,
     it should hold up whether we keep the top 50% or top 5% most CM-like
     cells. If it only appears at some thresholds and flips or vanishes at
     others, that is a sign the "signal" was an artifact of an arbitrary
     cutoff, not a property of the underlying biology.
  4. At each threshold: score CM-like cells for the vCM3.0 stress program
     (ANKRD1, XIRP1, XIRP2, FHL1, ACTA1, MYH9), then compare donor-level mean
     stress score, DCM vs non-diseased donor (Wilcoxon rank-sum, donor-level
     not cell-level, matching the pseudobulk logic used in Phase 1 to avoid
     pseudoreplication).

Caveat that must survive into the writeup: this marker-score proxy is weaker
evidence than the original Reichart finding, which used real expert-annotated
substates. A consistent result across the threshold sweep is suggestive
replication under a rough cell-type proxy, not independent confirmation with
the same rigor as Phase 1.
"""

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ranksums

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "../data/chaffin_koenig/koenig_atlas.h5ad"
RESULTS_PATH = "../results/koenig_vcm3_replication.csv"

DCM_LABEL = "dilated cardiomyopathy"
DONOR_LABEL = "non-diseased donor"

CM_MARKERS = ["TTN", "MYH6", "MYH7", "TNNT2", "ACTC1"]
STRESS_PROGRAM = ["ANKRD1", "XIRP1", "XIRP2", "FHL1", "ACTA1", "MYH9"]
CM_SCORE_PERCENTILES = [50, 60, 70, 80, 90, 95]  # sweep instead of one fixed cutoff


def run_at_threshold(adata: ad.AnnData, percentile: float,
                      stress_genes_present: list[str]) -> dict:
    """Run the DCM-vs-donor stress-score comparison keeping only cells with a
    CM score above the given percentile. Returns summary stats for this threshold."""
    cutoff = np.percentile(adata.obs["cm_score"], percentile)
    cm_mask = (adata.obs["cm_score"] > cutoff).values
    n_cm = cm_mask.sum()

    cm_adata = adata[cm_mask].copy()
    n_donors_cm = cm_adata.obs["donor_id"].nunique()

    sc.tl.score_genes(cm_adata, stress_genes_present, score_name="stress_score")

    donor_scores = cm_adata.obs.groupby("donor_id").agg(
        stress_score=("stress_score", "mean"),
        disease=("disease", "first"),
        n_cm_cells=("stress_score", "size"),
    ).reset_index()

    dcm_scores = donor_scores.loc[donor_scores["disease"] == DCM_LABEL, "stress_score"]
    donor_scores_ = donor_scores.loc[donor_scores["disease"] == DONOR_LABEL, "stress_score"]

    stat, pval = ranksums(dcm_scores, donor_scores_)

    result = {
        "percentile": percentile,
        "cm_score_cutoff": cutoff,
        "n_cm_cells": int(n_cm),
        "pct_cells_kept": 100 * n_cm / adata.n_obs,
        "n_donors_cm": n_donors_cm,
        "n_dcm_donors": len(dcm_scores),
        "n_donor_donors": len(donor_scores_),
        "dcm_mean_stress": dcm_scores.mean(),
        "dcm_sd_stress": dcm_scores.std(),
        "donor_mean_stress": donor_scores_.mean(),
        "donor_sd_stress": donor_scores_.std(),
        "wilcoxon_stat": stat,
        "pval": pval,
    }
    logger.info(
        f"  percentile={percentile}: kept {n_cm} cells ({result['pct_cells_kept']:.1f}%), "
        f"{n_donors_cm} donors | DCM mean={result['dcm_mean_stress']:.4f}, "
        f"donor mean={result['donor_mean_stress']:.4f}, p={pval:.4f}"
    )
    return result


def main() -> None:
    logger.info("Loading Koenig/Lavine atlas...")
    adata = ad.read_h5ad(DATA_PATH)
    logger.info(f"{adata.n_obs} cells x {adata.n_vars} genes, {adata.obs['donor_id'].nunique()} donors")

    logger.info("Normalizing (total-count + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cm_markers_present = [g for g in CM_MARKERS if g in adata.var_names]
    stress_genes_present = [g for g in STRESS_PROGRAM if g in adata.var_names]
    logger.info(f"CM markers present: {cm_markers_present}")
    logger.info(f"Stress program genes present: {stress_genes_present}")

    logger.info("Scoring cells for canonical cardiomyocyte markers...")
    sc.tl.score_genes(adata, cm_markers_present, score_name="cm_score")

    logger.info("Sweeping CM-score percentile threshold (robustness check, not "
                "one fixed arbitrary cutoff)...")
    sweep_results = []
    for percentile in CM_SCORE_PERCENTILES:
        sweep_results.append(run_at_threshold(adata, percentile, stress_genes_present))

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Wrote {RESULTS_PATH}")

    logger.info("\nFull threshold sweep:")
    logger.info(sweep_df.to_string())

    n_significant = (sweep_df["pval"] < 0.05).sum()
    all_same_direction = (sweep_df["dcm_mean_stress"] > sweep_df["donor_mean_stress"]).all() or \
                          (sweep_df["dcm_mean_stress"] < sweep_df["donor_mean_stress"]).all()
    logger.info(f"\n{n_significant}/{len(sweep_df)} thresholds significant at p<0.05")
    logger.info(f"Consistent direction across all thresholds: {all_same_direction}")


if __name__ == "__main__":
    main()
