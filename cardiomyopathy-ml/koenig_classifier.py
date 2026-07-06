"""
koenig_classifier.py — does the DCM classification signal, not just the gene-score
signal, also concentrate in cardiomyocyte-like cells in the Koenig/Lavine cohort?

Phase 2's replication check (koenig_vcm3_replication.py) tested one gene program's
average activity. This asks a different, complementary question: can a classifier
trained on a cell's full gene expression profile actually predict DCM vs
non-diseased donor, and does that predictive signal get stronger as cells are
filtered down to more confidently cardiomyocyte-like ones? That mirrors the Phase 1
finding on Reichart, where honest AUC was higher in the vCM3.0 substate (0.775) than
the whole population (0.726).

No ACM cohort exists in Koenig/Lavine, so this is DCM-vs-donor, not DCM-vs-ACM —
same substitution and same caveat as the rest of Phase 2.

Method: honest, patient-level evaluation via StratifiedGroupKFold (all cells from one
donor stay in one fold), matching the "naive vs honest AUC" logic from evaluation.py
that was the strongest result in Phase 1 (0.88 naive to 0.70 patient-level, revealing
donor-identity leakage in the naive split). Run at the same six CM-score percentile
thresholds used in the replication check, for direct comparability.

Runs entirely on CPU (RandomForest on ~2,000 top-variance genes) — no GPU needed.
The GPU rental in Phase 1 was for scVI (a neural network trained with gradient
descent over many epochs); a RandomForest here is a fundamentally lighter
computation that trains in minutes on a laptop even at the largest cell count.
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
RESULTS_PATH = "../results/koenig_classifier_auc.csv"

DCM_LABEL = "dilated cardiomyopathy"
DONOR_LABEL = "non-diseased donor"

CM_MARKERS = ["TTN", "MYH6", "MYH7", "TNNT2", "ACTC1"]
CM_SCORE_PERCENTILES = [50, 60, 70, 80, 90, 95]
N_TOP_GENES = 2000
N_SPLITS = 5


def run_at_threshold(adata: ad.AnnData, percentile: float) -> dict:
    """Honest patient-level AUC for DCM vs donor, keeping only cells above the
    given CM-score percentile."""
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
                    f"for {N_SPLITS}-fold CV (DCM donors={n_dcm_donors}, "
                    f"non-diseased donors={n_donor_donors})")
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
        "cm_score_cutoff": cutoff,
        "n_cells": int(cm_mask.sum()),
        "n_donors": n_donors,
        "n_dcm_donors": n_dcm_donors,
        "n_donor_donors": n_donor_donors,
        "mean_auc": float(np.mean(aucs)),
        "std_auc": float(np.std(aucs)),
        "fold_aucs": aucs,
    }
    logger.info(f"  percentile={percentile}: {cm_mask.sum()} cells, {n_donors} donors "
                f"(DCM={n_dcm_donors}, donor={n_donor_donors}) | "
                f"honest patient-level AUC = {result['mean_auc']:.4f} +/- {result['std_auc']:.4f}")
    return result


def main() -> None:
    logger.info("Loading Koenig/Lavine atlas...")
    adata = ad.read_h5ad(DATA_PATH)
    logger.info(f"{adata.n_obs} cells x {adata.n_vars} genes, {adata.obs['donor_id'].nunique()} donors")

    logger.info("Normalizing (total-count + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cm_markers_present = [g for g in CM_MARKERS if g in adata.var_names]
    logger.info("Scoring cells for canonical cardiomyocyte markers...")
    sc.tl.score_genes(adata, cm_markers_present, score_name="cm_score")

    logger.info(f"Running honest patient-level classification at {len(CM_SCORE_PERCENTILES)} "
                f"CM-score thresholds (RandomForest, top {N_TOP_GENES} HVGs, "
                f"{N_SPLITS}-fold StratifiedGroupKFold)...")
    results = []
    for percentile in CM_SCORE_PERCENTILES:
        result = run_at_threshold(adata, percentile)
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Wrote {RESULTS_PATH}")

    logger.info("\nFull threshold sweep:")
    logger.info(results_df[["percentile", "n_cells", "n_donors", "mean_auc", "std_auc"]].to_string())


if __name__ == "__main__":
    main()
