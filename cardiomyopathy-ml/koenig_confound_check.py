"""
koenig_confound_check.py — rules out sequencing-technology leakage in the
Koenig/Lavine DCM-vs-donor classifier result.

koenig_classifier.py found a very high honest patient-level AUC (0.92-0.97) for
DCM vs non-diseased donor. Before trusting that number, one confound needed to be
checked: DCM samples are 4x more likely to be "Single Cell" technology than donor
samples (28% vs 7%, from the scraped GSM metadata) — 5/18 DCM donors are Single
Cell vs 2/27 donor donors. Single-cell and single-nucleus RNA-seq capture
systematically different gene sets for protocol reasons unrelated to disease
biology (nuclei-only capture misses more cytoplasmic transcripts). If technology
correlates with disease label, a classifier could partly or largely be learning
"which protocol was used" rather than "is this tissue diseased" — this is the
single most common way single-cell disease-classification results get invalidated.

The test: restrict to Single Nuclei samples ONLY (13 DCM, 25 donor — technology
held constant, no imbalance to exploit) and re-run the identical honest
patient-level classifier. If the AUC collapses toward chance, the original result
was largely a technology artifact. If it holds up, that is real evidence the
signal is disease biology, not protocol.
"""

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "../data/chaffin_koenig/koenig_atlas.h5ad"
RESULTS_PATH = "../results/koenig_confound_check_auc.csv"

DCM_LABEL = "dilated cardiomyopathy"
DONOR_LABEL = "non-diseased donor"

CM_MARKERS = ["TTN", "MYH6", "MYH7", "TNNT2", "ACTC1"]
CM_SCORE_PERCENTILES = [50, 60, 70, 80, 90, 95]
N_TOP_GENES = 2000
N_SPLITS = 5


def run_at_threshold(adata: ad.AnnData, percentile: float) -> dict:
    cutoff = np.percentile(adata.obs["cm_score"], percentile)
    cm_mask = (adata.obs["cm_score"] > cutoff).values
    cm_adata = adata[cm_mask].copy()

    y = (cm_adata.obs["disease"] == DCM_LABEL).astype(int).values
    groups = cm_adata.obs["donor_id"].values
    n_donors = cm_adata.obs["donor_id"].nunique()
    n_dcm_donors = cm_adata.obs.loc[cm_adata.obs["disease"] == DCM_LABEL, "donor_id"].nunique()
    n_donor_donors = n_donors - n_dcm_donors

    if n_dcm_donors < N_SPLITS or n_donor_donors < N_SPLITS:
        logger.info(f"  percentile={percentile}: skipped, too few donors per class "
                    f"(DCM={n_dcm_donors}, donor={n_donor_donors})")
        return None

    sc.pp.highly_variable_genes(cm_adata, n_top_genes=N_TOP_GENES, flavor="seurat")
    X = cm_adata[:, cm_adata.var["highly_variable"]].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    aucs = []
    for train_idx, val_idx in cv.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], preds))

    result = {
        "percentile": percentile,
        "n_cells": int(cm_mask.sum()),
        "n_donors": n_donors,
        "n_dcm_donors": n_dcm_donors,
        "n_donor_donors": n_donor_donors,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
    }
    logger.info(f"  percentile={percentile}: {cm_mask.sum()} cells, {n_donors} donors "
                f"(DCM={n_dcm_donors}, donor={n_donor_donors}) | "
                f"honest patient-level AUC = {result['mean_auc']:.4f} +/- {result['std_auc']:.4f}")
    return result


def main() -> None:
    logger.info("Loading Koenig/Lavine atlas...")
    adata = ad.read_h5ad(DATA_PATH)

    logger.info("Restricting to Single Nuclei samples only (removes the technology "
                "imbalance: original data was 28%% Single Cell in DCM vs 7%% in donor)...")
    adata = adata[adata.obs["technology"] == "Single Nuclei"].copy()
    n_dcm = adata.obs.loc[adata.obs["disease"] == DCM_LABEL, "donor_id"].nunique()
    n_donor = adata.obs.loc[adata.obs["disease"] == DONOR_LABEL, "donor_id"].nunique()
    logger.info(f"After restriction: {adata.n_obs} cells, {n_dcm} DCM donors, "
                f"{n_donor} non-diseased donors (technology held constant)")

    logger.info("Normalizing (total-count + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cm_markers_present = [g for g in CM_MARKERS if g in adata.var_names]
    sc.tl.score_genes(adata, cm_markers_present, score_name="cm_score")

    logger.info(f"Running honest patient-level classification at {len(CM_SCORE_PERCENTILES)} "
                f"CM-score thresholds, Single Nuclei samples only...")
    results = []
    for percentile in CM_SCORE_PERCENTILES:
        result = run_at_threshold(adata, percentile)
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Wrote {RESULTS_PATH}")

    logger.info("\nFull threshold sweep (Single Nuclei only, technology confound removed):")
    logger.info(results_df.to_string())


if __name__ == "__main__":
    main()
