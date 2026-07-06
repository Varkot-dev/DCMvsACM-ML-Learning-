"""
koenig_ckd_confound_check.py — rules out chronic kidney disease (CKD) as a
confound in the Koenig/Lavine DCM-vs-donor classifier result.

The technology confound (Single Cell vs Single Nuclei) was already checked and
ruled out in koenig_confound_check.py. A second, larger imbalance turned up
when patient-level clinical metadata was pulled from the original paper's
supplementary Table 20 ("Patient-level metadata", MOESM3, not present in GEO's
own per-sample fields): 7/18 DCM patients have CKD (39%) vs 1/27 healthy donors
(3.7%) — roughly a 10x skew, larger than the technology imbalance. Cardiorenal
syndrome is a real clinical association (heart failure and kidney disease
often co-occur), so this imbalance is not surprising, but it is not safe to
ignore: CKD has its own transcriptomic signature (altered fluid balance,
uremic toxins, systemic inflammation), and if that signature is detectable in
heart tissue, the classifier could be partly learning "signs of CKD" rather
than "signs of DCM."

The test: restrict to CKD-negative patients only (11 DCM, 26 healthy — still
above the 5-donor-per-class minimum for 5-fold StratifiedGroupKFold) and
re-run the identical honest patient-level classifier. If the AUC collapses,
CKD was a real contributor. If it holds up, this confound is also ruled out.

Patient metadata source: Koenig/Lavine 2022 Nature Cardiovascular Research,
Supplementary Table 20, cross-referenced by sample name (100% match, 45/45
donors) against the h5ad's donor_id column.
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
CKD_METADATA_PATH = "../data/chaffin_koenig/ckd_metadata.csv"
RESULTS_PATH = "../results/koenig_ckd_confound_check_auc.csv"

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
    logger.info("Loading Koenig/Lavine atlas and CKD metadata...")
    adata = ad.read_h5ad(DATA_PATH)
    ckd_meta = pd.read_csv(CKD_METADATA_PATH)
    ckd_negative_donors = set(ckd_meta.loc[ckd_meta["CKD"] == 0, "Sample"])

    logger.info("Restricting to CKD-negative patients only (removes the clinical "
                "imbalance: 39%% of DCM patients have CKD vs 3.7%% of donors)...")
    adata = adata[adata.obs["donor_id"].isin(ckd_negative_donors)].copy()
    n_dcm = adata.obs.loc[adata.obs["disease"] == DCM_LABEL, "donor_id"].nunique()
    n_donor = adata.obs.loc[adata.obs["disease"] == DONOR_LABEL, "donor_id"].nunique()
    logger.info(f"After restriction: {adata.n_obs} cells, {n_dcm} DCM donors, "
                f"{n_donor} non-diseased donors (all CKD-negative)")

    logger.info("Normalizing (total-count + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cm_markers_present = [g for g in CM_MARKERS if g in adata.var_names]
    sc.tl.score_genes(adata, cm_markers_present, score_name="cm_score")

    logger.info(f"Running honest patient-level classification at {len(CM_SCORE_PERCENTILES)} "
                f"CM-score thresholds, CKD-negative patients only...")
    results = []
    for percentile in CM_SCORE_PERCENTILES:
        result = run_at_threshold(adata, percentile)
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Wrote {RESULTS_PATH}")

    logger.info("\nFull threshold sweep (CKD-negative only, kidney disease confound removed):")
    logger.info(results_df.to_string())


if __name__ == "__main__":
    main()
