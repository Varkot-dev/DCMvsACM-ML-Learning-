"""
reichart_dcm_vs_normal_classifier.py — Phase 3, step 1: train an honest,
patient-level DCM-vs-normal classifier on Reichart, to be tested cross-cohort
on Koenig/Lavine in a later step.

This is a different disease contrast than Phase 1 (DCM vs ACM). Phase 3's
calibration-under-shift question needs a train/test pair with the SAME
disease contrast on both cohorts. Koenig/Lavine has no ACM patients, so
DCM-vs-normal (here) / DCM-vs-donor (Koenig) is the contrast that lets a
model trained on one cohort be meaningfully tested on the other.

Critical detail for cross-cohort transfer: the model can only be applied
later to genes it saw during training, in a fixed feature set. A first
attempt selected the top 2,000 highly-variable genes from Reichart alone,
then checked overlap with Koenig's gene list afterward — 477/2000 (24%)
turned out to be absent from Koenig (mostly Ensembl IDs and obscure
lncRNAs/antisense transcripts that different annotation pipelines label
inconsistently between labs). Rather than approximate the missing 24% with
zero-fill (a real, meaningful expression value, not a safe stand-in for
"unknown"), gene selection here is restricted to the intersection of
Reichart's and Koenig's gene lists from the start — the model is trained on
exactly the features it will see at test time, nothing improvised after
the fact. This script saves the exact selected gene list (as gene symbols,
matching Koenig's naming) alongside the trained model.

Honest evaluation: StratifiedGroupKFold (all cells from one donor stay in
one fold), same logic as evaluation.py and today's Koenig classifier work.
"""

import logging
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "cardiomyocytes.h5ad"
KOENIG_PATH = "../data/chaffin_koenig/koenig_atlas.h5ad"
MODEL_PATH = "../results/reichart_dcm_vs_normal_model.pkl"
GENE_LIST_PATH = "../results/reichart_dcm_vs_normal_genes.csv"
RESULTS_PATH = "../results/reichart_dcm_vs_normal_auc.csv"
HELD_OUT_PREDICTIONS_PATH = "../results/reichart_held_out_predictions.csv"

DCM_LABEL = "dilated cardiomyopathy"
NORMAL_LABEL = "normal"

N_TOP_GENES = 2000
N_SPLITS = 5


def main() -> None:
    logger.info("Loading Reichart atlas, filtering to DCM + normal...")
    adata = ad.read_h5ad(DATA_PATH)
    mask = adata.obs["disease"].isin([DCM_LABEL, NORMAL_LABEL])
    adata = adata[mask].copy()
    n_donors = adata.obs["donor_id"].nunique()
    n_dcm = adata.obs.loc[adata.obs["disease"] == DCM_LABEL, "donor_id"].nunique()
    n_normal = adata.obs.loc[adata.obs["disease"] == NORMAL_LABEL, "donor_id"].nunique()
    logger.info(f"{adata.n_obs} cells, {n_donors} donors (DCM={n_dcm}, normal={n_normal})")

    logger.info("Using .X as-is (Reichart's .X is already log-normalized, "
                "confirmed 2026-07-06 when raw counts were needed elsewhere "
                "and found in .raw.X instead).")

    logger.info("Loading Koenig gene list to restrict HVG selection to genes present "
                "in both cohorts (required for cross-cohort model application)...")
    koenig_genes = set(ad.read_h5ad(KOENIG_PATH, backed="r").var_names)
    shared_gene_mask = adata.var["feature_name"].isin(koenig_genes).values
    logger.info(f"{shared_gene_mask.sum()} / {adata.n_vars} Reichart genes are present in Koenig")

    logger.info(f"Selecting top {N_TOP_GENES} highly-variable genes from the shared set...")
    adata_shared = adata[:, shared_gene_mask].copy()
    sc.pp.highly_variable_genes(adata_shared, n_top_genes=N_TOP_GENES, flavor="seurat")
    hvg_mask = adata_shared.var["highly_variable"].values
    gene_symbols = adata_shared.var.loc[hvg_mask, "feature_name"].values
    logger.info(f"Selected {len(gene_symbols)} genes (all confirmed present in Koenig)")
    adata = adata_shared

    X = adata[:, hvg_mask].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = (adata.obs["disease"] == DCM_LABEL).astype(int).values
    groups = adata.obs["donor_id"].values

    logger.info(f"Running honest patient-level {N_SPLITS}-fold CV...")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    aucs = []
    fold_models = []
    held_out_predictions = []  # saved for the calibration analysis (in-domain baseline)
    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict_proba(X[val_idx])[:, 1]
        auc = roc_auc_score(y[val_idx], preds)
        aucs.append(auc)
        fold_models.append(clf)
        held_out_predictions.append(pd.DataFrame({
            "fold": fold_i + 1,
            "donor_id": groups[val_idx],
            "y_true": y[val_idx],
            "predicted_proba": preds,
        }))
        logger.info(f"  fold {fold_i+1}/{N_SPLITS}: AUC={auc:.4f}")

    logger.info(f"Honest patient-level AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    logger.info("Training final model on ALL DCM+normal data (for cross-cohort "
                "application — the CV folds above are for the honest AUC estimate "
                "only; this final model uses every available donor).")
    final_model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=42, n_jobs=-1)
    final_model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    logger.info(f"Wrote {MODEL_PATH}")

    pd.DataFrame({"gene_symbol": gene_symbols}).to_csv(GENE_LIST_PATH, index=False)
    logger.info(f"Wrote {GENE_LIST_PATH}")

    pd.DataFrame({
        "fold": range(1, N_SPLITS + 1),
        "auc": aucs,
    }).to_csv(RESULTS_PATH, index=False)
    logger.info(f"Wrote {RESULTS_PATH}")

    held_out_df = pd.concat(held_out_predictions, ignore_index=True)
    held_out_df.to_csv(HELD_OUT_PREDICTIONS_PATH, index=False)
    logger.info(f"Wrote {HELD_OUT_PREDICTIONS_PATH} "
                f"({len(held_out_df)} held-out predictions, in-domain calibration baseline)")

    logger.info(f"\nSummary: honest patient-level AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} "
                f"across {N_SPLITS} folds, {n_donors} donors, {len(gene_symbols)} genes.")


if __name__ == "__main__":
    main()
