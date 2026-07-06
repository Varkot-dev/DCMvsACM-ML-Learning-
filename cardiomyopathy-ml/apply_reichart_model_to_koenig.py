"""
apply_reichart_model_to_koenig.py — Phase 3, step 2: apply the frozen
Reichart-trained DCM-vs-normal model to Koenig/Lavine, producing cross-cohort
predictions for the calibration analysis.

The model is NOT retrained here. It stays exactly as trained on Reichart
(reichart_dcm_vs_normal_classifier.py) — this script only runs inference,
which is the whole point: measuring how a model trained on one cohort
behaves when it sees a genuinely different cohort it has never encountered.

Gene alignment: the model expects exactly the 2,000 genes saved during
training, in a fixed order. Koenig's cells are subset and reordered to match
that exact list before being passed to the model. All 2,000 genes were
already confirmed present in Koenig during training, so no imputation or
zero-filling is needed here.

Output: per-cell predicted probabilities on Koenig, alongside true disease
labels and donor_id, for the calibration step that follows.
"""

import logging
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

KOENIG_PATH = "../data/chaffin_koenig/koenig_atlas.h5ad"
MODEL_PATH = "../results/reichart_dcm_vs_normal_model.pkl"
GENE_LIST_PATH = "../results/reichart_dcm_vs_normal_genes.csv"
RESULTS_PATH = "../results/koenig_cross_cohort_predictions.csv"

DCM_LABEL = "dilated cardiomyopathy"
DONOR_LABEL = "non-diseased donor"


def main() -> None:
    logger.info("Loading frozen Reichart-trained model and its gene list...")
    # Pickle load is safe here: MODEL_PATH is written by
    # reichart_dcm_vs_normal_classifier.py in this same repo, never fetched
    # from a network source or accepted as external input.
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    gene_list = pd.read_csv(GENE_LIST_PATH)["gene_symbol"].tolist()
    logger.info(f"Model expects {len(gene_list)} genes, in a fixed order")

    logger.info("Loading Koenig/Lavine atlas...")
    adata = ad.read_h5ad(KOENIG_PATH)
    logger.info(f"{adata.n_obs} cells, {adata.obs['donor_id'].nunique()} donors")

    missing = [g for g in gene_list if g not in adata.var_names]
    if missing:
        raise ValueError(f"{len(missing)} genes missing from Koenig despite prior "
                          f"verification during training — data may have changed: {missing}")

    logger.info("Normalizing Koenig (total-count + log1p, same as training pipeline)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    logger.info("Subsetting and reordering Koenig's genes to match the model's exact "
                "training feature order...")
    adata_aligned = adata[:, gene_list].copy()
    X = adata_aligned.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    logger.info("Running inference (no training, frozen model)...")
    predicted_proba = model.predict_proba(X)[:, 1]

    y_true = (adata.obs["disease"] == DCM_LABEL).astype(int).values

    results = pd.DataFrame({
        "cell_id": adata.obs_names,
        "donor_id": adata.obs["donor_id"].values,
        "disease": adata.obs["disease"].values,
        "y_true": y_true,
        "predicted_proba": predicted_proba,
    })
    results.to_csv(RESULTS_PATH, index=False)
    logger.info(f"Wrote {RESULTS_PATH}")

    from sklearn.metrics import roc_auc_score
    cross_cohort_auc = roc_auc_score(y_true, predicted_proba)
    logger.info(f"\nCross-cohort AUC (Reichart-trained model, Koenig test data): "
                f"{cross_cohort_auc:.4f}")
    logger.info(f"(Compare to Reichart's own honest patient-level AUC: 0.9668)")
    logger.info(f"(Compare to Koenig's own honest patient-level AUC, trained fresh on "
                f"Koenig: 0.92-0.97 range depending on threshold — today's earlier result)")


if __name__ == "__main__":
    main()
