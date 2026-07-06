"""
koenig_permutation_test.py — is the Koenig/Lavine DCM-vs-donor classifier AUC
real, or could it arise by chance given this sample size (18 DCM, 27 donor)?

The technology and CKD confound checks (koenig_confound_check.py,
koenig_ckd_confound_check.py) tested two SPECIFIC hypotheses about what could
be inflating the AUC. This test is more general: it does not assume any
particular confound, and instead directly measures how much AUC you would
expect from pure chance at this exact sample size, using this exact pipeline.

Method: shuffle the disease labels across donors (breaking any real
connection between a donor's true disease status and the label the model
sees), then re-run the identical honest patient-level classifier. Repeat many
times to build a null distribution of "chance-level AUC with this many
donors, this many cells, this feature set." If the real (unshuffled) AUC
sits far above that null distribution, the result is unlikely to be a
small-sample artifact. If the null distribution's spread reaches anywhere
near the real AUC, the sample size is too small for this result to be
trustworthy on its own.

Run at one representative threshold (80th percentile CM score, the middle of
the sweep already run) rather than all six, since permutation testing
multiplies compute by the number of shuffles.
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
RESULTS_PATH = "../results/koenig_permutation_test.csv"

DCM_LABEL = "dilated cardiomyopathy"
DONOR_LABEL = "non-diseased donor"

CM_MARKERS = ["TTN", "MYH6", "MYH7", "TNNT2", "ACTC1"]
CM_SCORE_PERCENTILE = 80
N_TOP_GENES = 2000
N_SPLITS = 5
N_PERMUTATIONS = 20


def compute_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float:
    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    aucs = []
    for train_idx, val_idx in cv.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], preds))
    return float(np.mean(aucs))


def main() -> None:
    logger.info("Loading Koenig/Lavine atlas...")
    adata = ad.read_h5ad(DATA_PATH)

    logger.info("Normalizing (total-count + log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cm_markers_present = [g for g in CM_MARKERS if g in adata.var_names]
    sc.tl.score_genes(adata, cm_markers_present, score_name="cm_score")

    cutoff = np.percentile(adata.obs["cm_score"], CM_SCORE_PERCENTILE)
    cm_mask = (adata.obs["cm_score"] > cutoff).values
    cm_adata = adata[cm_mask].copy()
    logger.info(f"CM-score percentile {CM_SCORE_PERCENTILE}: {cm_mask.sum()} cells, "
                f"{cm_adata.obs['donor_id'].nunique()} donors")

    sc.pp.highly_variable_genes(cm_adata, n_top_genes=N_TOP_GENES, flavor="seurat")
    X = cm_adata[:, cm_adata.var["highly_variable"]].X
    if hasattr(X, "toarray"):
        X = X.toarray()

    y_true = (cm_adata.obs["disease"] == DCM_LABEL).astype(int).values
    groups = cm_adata.obs["donor_id"].values

    logger.info("Computing real (unshuffled) AUC...")
    real_auc = compute_auc(X, y_true, groups, seed=42)
    logger.info(f"Real AUC: {real_auc:.4f}")

    # Build donor-level label map, shuffle at the DONOR level (not cell level) —
    # shuffling individual cells would break the honest per-donor structure and
    # let cells from the same donor end up with different shuffled labels,
    # which is not a fair comparison to the real, donor-consistent labels.
    donor_labels = cm_adata.obs.groupby("donor_id")["disease"].first()
    donor_ids = donor_labels.index.values
    true_donor_labels = donor_labels.values

    logger.info(f"Running {N_PERMUTATIONS} label-shuffled permutations "
                f"(donor-level shuffle, not cell-level)...")
    rng = np.random.default_rng(42)
    perm_aucs = []
    for i in range(N_PERMUTATIONS):
        shuffled_labels = rng.permutation(true_donor_labels)
        donor_to_shuffled = dict(zip(donor_ids, shuffled_labels))
        y_shuffled = np.array([
            1 if donor_to_shuffled[d] == DCM_LABEL else 0
            for d in cm_adata.obs["donor_id"].values
        ])

        n_dcm_shuffled = (shuffled_labels == DCM_LABEL).sum()
        n_donor_shuffled = len(shuffled_labels) - n_dcm_shuffled
        if n_dcm_shuffled < N_SPLITS or n_donor_shuffled < N_SPLITS:
            continue

        auc = compute_auc(X, y_shuffled, groups, seed=42 + i)
        perm_aucs.append(auc)
        if (i + 1) % 5 == 0:
            logger.info(f"  permutation {i+1}/{N_PERMUTATIONS}: AUC={auc:.4f}")

    perm_aucs = np.array(perm_aucs)
    p_value = (perm_aucs >= real_auc).sum() / len(perm_aucs)

    logger.info(f"\nReal AUC: {real_auc:.4f}")
    logger.info(f"Null distribution (shuffled labels): mean={perm_aucs.mean():.4f}, "
                f"std={perm_aucs.std():.4f}, min={perm_aucs.min():.4f}, max={perm_aucs.max():.4f}")
    logger.info(f"Permutation p-value (fraction of shuffles >= real AUC): {p_value:.4f}")

    pd.DataFrame({
        "permutation": range(len(perm_aucs)),
        "shuffled_auc": perm_aucs,
    }).to_csv(RESULTS_PATH, index=False)
    with open(RESULTS_PATH.replace(".csv", "_summary.txt"), "w") as f:
        f.write(f"real_auc={real_auc:.4f}\n")
        f.write(f"null_mean={perm_aucs.mean():.4f}\n")
        f.write(f"null_std={perm_aucs.std():.4f}\n")
        f.write(f"null_max={perm_aucs.max():.4f}\n")
        f.write(f"p_value={p_value:.4f}\n")
    logger.info(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
