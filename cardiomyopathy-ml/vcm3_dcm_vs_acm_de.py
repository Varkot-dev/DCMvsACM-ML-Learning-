"""
vcm3_dcm_vs_acm_de.py — Phase 1: DCM vs ACM differential expression within vCM3.0.

Question: within the vCM3.0 stressed substate only, do DCM and ACM differ,
and is FLNC among the differences?

Method: pseudobulk by summing raw counts per donor within the vCM3.0 subset,
CPM-normalize, log1p, then Wilcoxon rank-sum DCM vs ACM per gene with BH-FDR.
Genes are filtered to mean expression > 0.05 in at least one group before
testing, avoiding the KCNA10/OR8K3 near-zero-denominator artifact found on
2026-07-06.

Distinct from the 2026-07-06 analysis: that contrast was vCM3.0-vs-rest
(what defines the substate). This contrast is DCM-vs-ACM inside vCM3.0 only
(does disease separate cells already in the stressed state).
"""

import logging

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, ranksums

from data import ACM_LABEL, DCM_LABEL

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = "cardiomyocytes.h5ad"
EXPR_FILTER_THRESHOLD = 0.05
RESULTS_PATH = "../results/vcm3_dcm_vs_acm_de.csv"
OBS_PATH = "../results/vcm3_dcm_vs_acm_pseudobulk_obs.csv"


def pseudobulk_sum_counts(adata: ad.AnnData) -> tuple[np.ndarray, pd.DataFrame]:
    """Sum raw counts per donor, then CPM-normalize and log1p.

    Donor-level pseudobulk, not cell-level averaging: per-cell DE was already
    established on 2026-07-05 and is a different question from this one.
    """
    donors = adata.obs["donor_id"].unique()
    raw_X = adata.raw.X
    if not sp.issparse(raw_X):
        raw_X = sp.csr_matrix(raw_X)

    rows = []
    meta = []
    for donor in donors:
        mask = (adata.obs["donor_id"] == donor).values
        counts = np.asarray(raw_X[mask].sum(axis=0)).ravel()
        disease = adata.obs.loc[mask, "disease"].iloc[0]
        rows.append(counts)
        meta.append({
            "donor": donor,
            "label": "DCM" if disease == DCM_LABEL else "ACM",
            "n_cells": int(mask.sum()),
        })

    counts_matrix = np.vstack(rows)
    cpm = counts_matrix / counts_matrix.sum(axis=1, keepdims=True) * 1e6
    log_cpm = np.log1p(cpm)
    return log_cpm, pd.DataFrame(meta)


def differential_expression(log_cpm: np.ndarray, meta: pd.DataFrame, gene_names: np.ndarray) -> pd.DataFrame:
    """Wilcoxon rank-sum DCM vs ACM per gene, BH-FDR, with an expression filter first."""
    dcm_mask = (meta["label"] == "DCM").values
    acm_mask = (meta["label"] == "ACM").values

    mean_dcm = log_cpm[dcm_mask].mean(axis=0)
    mean_acm = log_cpm[acm_mask].mean(axis=0)
    keep = (mean_dcm > EXPR_FILTER_THRESHOLD) | (mean_acm > EXPR_FILTER_THRESHOLD)
    logger.info(f"Expression filter: {keep.sum()} / {len(keep)} genes retained "
                f"(mean log-CPM > {EXPR_FILTER_THRESHOLD} in at least one group)")

    filtered_genes = gene_names[keep]
    filtered_log_cpm = log_cpm[:, keep]
    filtered_mean_dcm = mean_dcm[keep]
    filtered_mean_acm = mean_acm[keep]

    pvals = np.array([
        ranksums(filtered_log_cpm[dcm_mask, g], filtered_log_cpm[acm_mask, g])[1]
        for g in range(filtered_log_cpm.shape[1])
    ])

    n = len(pvals)
    fdr = np.minimum(pvals * n / rankdata(pvals), 1.0)

    results = pd.DataFrame({
        "gene": filtered_genes,
        "pval": pvals,
        "fdr": fdr,
        "mean_dcm": filtered_mean_dcm,
        "mean_acm": filtered_mean_acm,
    }).sort_values("pval")
    results["log2fc"] = (
        np.log2(results["mean_dcm"] + 1e-6) - np.log2(results["mean_acm"] + 1e-6)
    )
    return results.reset_index(drop=True)


def main() -> None:
    logger.info("Loading atlas and subsetting to vCM3.0, DCM+ACM only...")
    adata = ad.read_h5ad(DATA_PATH)
    mask = (
        adata.obs["disease"].isin([DCM_LABEL, ACM_LABEL])
        & (adata.obs["cell_states"] == "vCM3.0")
    )
    adata = adata[mask].copy()

    n_donors = adata.obs["donor_id"].nunique()
    n_dcm = (adata.obs["disease"] == DCM_LABEL).sum()
    n_acm = (adata.obs["disease"] == ACM_LABEL).sum()
    logger.info(f"vCM3.0 subset: {adata.n_obs} cells, {n_donors} donors "
                f"(DCM cells={n_dcm}, ACM cells={n_acm})")

    logger.info("Building donor-level pseudobulk (sum raw counts, CPM, log1p)...")
    log_cpm, meta = pseudobulk_sum_counts(adata)
    n_dcm_donors = (meta["label"] == "DCM").sum()
    n_acm_donors = (meta["label"] == "ACM").sum()
    logger.info(f"Pseudobulk matrix: {log_cpm.shape[0]} donors x {log_cpm.shape[1]} genes "
                f"(DCM donors={n_dcm_donors}, ACM donors={n_acm_donors})")

    gene_names = (
        adata.var["feature_name"].values
        if "feature_name" in adata.var.columns
        else adata.var_names.values
    )

    logger.info("Running Wilcoxon rank-sum DCM vs ACM per gene...")
    results = differential_expression(log_cpm, meta, gene_names)

    results.to_csv(RESULTS_PATH, index=False)
    meta.to_csv(OBS_PATH, index=False)

    n_sig_05 = (results["fdr"] < 0.05).sum()
    n_sig_10 = (results["fdr"] < 0.10).sum()
    logger.info(f"FDR < 0.05: {n_sig_05} genes")
    logger.info(f"FDR < 0.10: {n_sig_10} genes")

    flnc_rows = results[results["gene"].isin(["FLNC", "FLNC-AS1"])]
    logger.info(f"FLNC / FLNC-AS1 in DCM-vs-ACM contrast within vCM3.0:\n{flnc_rows.to_string()}")

    stress_program = ["ANKRD1", "XIRP2", "ACTA1", "MYH9"]
    program_rows = results[results["gene"].isin(stress_program)]
    logger.info(f"Stress program genes (rank within this contrast):\n"
                f"{program_rows[['gene', 'pval', 'fdr', 'log2fc']].to_string()}")

    logger.info("Top 15 genes by p-value:")
    logger.info(results.head(15)[["gene", "pval", "fdr", "log2fc"]].to_string())

    logger.info(
        f"\nCaveat: {n_acm_donors} ACM donors is the same sample-size ceiling that "
        f"produced zero surviving genes in whole-atlas pseudobulk DE. A null result "
        f"here may reflect power, not biology."
    )


if __name__ == "__main__":
    main()
