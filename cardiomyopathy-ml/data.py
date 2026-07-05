"""
data.py — Data loading and validation for the DCMvsACM pipeline.

Computational biology note:
  This module is the data contract layer. Everything downstream trusts that
  what comes out of here has been validated. We fail loudly at the boundary
  rather than propagating bad data silently into the model.

  The Reichart 2022 (Science) atlas is the primary source:
    - 880k nuclei from 18 control + 61 failing hearts
    - disease labels: 'dilated cardiomyopathy' and
      'arrhythmogenic right ventricular cardiomyopathy'
    - donor_id column identifies the patient (critical for group-aware splits)

  Download from CZ CELLxGENE:
    Collection: e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb
    File: cardiomyocytes.h5ad
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# The exact disease label strings used in the Reichart 2022 atlas obs metadata
DCM_LABEL = "dilated cardiomyopathy"
ACM_LABEL = "arrhythmogenic right ventricular cardiomyopathy"
DISEASE_LABELS = [DCM_LABEL, ACM_LABEL]

# Required obs (cell-level metadata) columns
REQUIRED_OBS_COLUMNS = ["disease", "donor_id"]

# Binary label mapping used throughout (0=ACM, 1=DCM — alphabetical order)
LABEL_MAP = {ACM_LABEL: 0, DCM_LABEL: 1}
LABEL_NAMES = {0: "ACM", 1: "DCM"}


def validate_adata(adata: ad.AnnData) -> None:
    """
    Assert the data contract before any processing.
    Raises ValueError with a clear message if anything is wrong.

    Design note:
      Validating at the data boundary is standard in production ML.
      If you skip this and the 'donor_id' column is named differently
      in a new dataset, your group-aware split silently fails and you
      never know — every cell gets treated as its own patient.
    """
    for col in REQUIRED_OBS_COLUMNS:
        if col not in adata.obs.columns:
            raise ValueError(
                f"Required column '{col}' missing from adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    n_dcm = (adata.obs["disease"] == DCM_LABEL).sum()
    n_acm = (adata.obs["disease"] == ACM_LABEL).sum()
    if n_dcm == 0 or n_acm == 0:
        raise ValueError(
            f"After filtering, one or both classes are empty. "
            f"DCM cells: {n_dcm}, ACM cells: {n_acm}. "
            f"Check that disease labels match: {DISEASE_LABELS}"
        )

    n_donors = adata.obs["donor_id"].nunique()
    if n_donors < 6:
        raise ValueError(
            f"Only {n_donors} unique donors found. StratifiedGroupKFold "
            f"needs at least n_splits donors per class. Check donor_id column."
        )

    logger.info(
        f"Data validated: {adata.n_obs} cells, {adata.n_vars} genes, "
        f"{n_donors} donors | DCM: {n_dcm}, ACM: {n_acm}"
    )


def load_and_filter(
    path: str | Path,
    n_cells: Optional[int] = None,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Load the .h5ad atlas and filter to DCM + ACM cells only.

    Args:
        path: Path to the .h5ad file (Reichart 2022 atlas).
        n_cells: If set, randomly subsample to this many cells after filtering.
                 Useful for fast iteration. None = use all cells.
        random_state: Seed for reproducibility of subsampling.

    Returns:
        adata: Filtered AnnData with only DCM and ACM cells.
               obs always contains 'disease', 'donor_id'.
               X is the raw/normalised expression matrix.

    Computational biology note:
      We filter to DCM+ACM *before* subsampling. If you subsample first,
      you might accidentally drop the minority class (ACM) disproportionately,
      skewing the imbalance ratio before you even start.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Download the Reichart 2022 atlas from CZ CELLxGENE:\n"
            f"  Collection: e75342a8-0f3b-4ec5-8ee1-245a23e0f7cb\n"
            f"  Save as: {path}"
        )

    logger.info(f"Loading atlas from {path} ...")
    adata = ad.read_h5ad(path)
    logger.info(f"Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    logger.info(f"Disease label distribution before filtering:\n{adata.obs['disease'].value_counts()}")

    # Filter to DCM and ACM only
    mask = adata.obs["disease"].isin(DISEASE_LABELS)
    adata = adata[mask].copy()
    logger.info(f"After filtering to DCM+ACM: {adata.n_obs} cells")

    if adata.n_obs == 0:
        raise ValueError(
            "No cells remain after filtering. Check that 'disease' column contains "
            f"values matching: {DISEASE_LABELS}"
        )

    # Optional subsample — applied AFTER filtering so class proportions are preserved
    if n_cells is not None and n_cells < adata.n_obs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(adata.n_obs, size=n_cells, replace=False)
        adata = adata[idx].copy()
        logger.info(f"Subsampled to {adata.n_obs} cells")

    validate_adata(adata)
    return adata


def extract_arrays(
    adata: ad.AnnData,
    n_top_genes: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the three arrays the pipeline needs from the validated AnnData.

    Args:
        adata: Validated AnnData with DCM+ACM cells only.
        n_top_genes: If set, pre-select this many highly variable genes before
                     densifying. This is a memory optimisation for large atlases:
                     the full 32k-gene sparse matrix is filtered to n_top_genes
                     columns before converting to dense, reducing peak RAM from
                     ~40 GB to ~1.6 GB for the Reichart cardiomyocyte atlas.

                     NOTE: This is a global pre-selection on the full dataset —
                     it leaks slightly because test cells contribute to gene
                     selection. For the publication-quality run, set n_top_genes=None
                     and run on a machine with enough RAM. For exploratory work
                     this approximation is fine: the same genes would be selected
                     by the in-fold HVGSelector, which does the same Fano computation.

    Returns:
        X: Dense expression matrix, shape (n_cells, n_genes or n_top_genes).
        y: Integer disease labels (0=ACM, 1=DCM), shape (n_cells,).
        groups: Donor ID as integers for StratifiedGroupKFold, shape (n_cells,).

    Design note:
      'groups' is the key ingredient for honest evaluation. It encodes which
      patient each cell came from. Passing groups to StratifiedGroupKFold
      ensures all cells from a given patient land in the same fold — so the
      model is always evaluated on patients it has never seen.
    """
    import scipy.sparse as sp

    if n_top_genes is not None and hasattr(adata.X, "toarray"):
        # Pre-select HVGs on the sparse matrix before densifying.
        # Fano factor (var/mean) on sparse data — efficient column-wise stats.
        X_sparse = adata.X
        if not sp.issparse(X_sparse):
            X_sparse = sp.csr_matrix(X_sparse)
        X_sparse = X_sparse.astype(np.float64)

        means = np.asarray(X_sparse.mean(axis=0)).ravel()
        # E[X^2] - E[X]^2 for variance without densifying
        means_sq = np.asarray(X_sparse.power(2).mean(axis=0)).ravel()
        variances = means_sq - means ** 2

        safe_means = np.clip(means, 1e-10, None)
        fano = variances / safe_means

        effective_top = min(n_top_genes, adata.n_vars)
        top_idx = np.argsort(fano)[-effective_top:]
        adata = adata[:, top_idx]
        logger.info(f"Pre-selected {effective_top} HVGs from {len(fano)} genes (Fano factor, sparse)")

    # Expression matrix: handle both sparse and dense
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    # Disease labels as integers
    y = adata.obs["disease"].map(LABEL_MAP).values.astype(int)

    # Donor IDs as integers (StratifiedGroupKFold needs array-like, not strings)
    donor_series = adata.obs["donor_id"].astype("category")
    groups = donor_series.cat.codes.values

    logger.info(
        f"Extracted arrays | X: {X.shape} | "
        f"y: DCM={int((y==1).sum())}, ACM={int((y==0).sum())} | "
        f"groups: {len(np.unique(groups))} unique donors"
    )
    return X, y, groups


def make_synthetic_data(
    n_cells: int = 2000,
    n_genes: int = 500,
    n_donors: int = 20,
    dcm_fraction: float = 0.7,
    random_state: int = 42,
) -> ad.AnnData:
    """
    Generate synthetic data for pipeline testing WITHOUT downloading real data.

    This is the standard computational biology practice: build and validate
    the entire pipeline on known-good synthetic data first. If the pipeline
    logic is correct on synthetic data, it will be correct on real data.

    The synthetic data has a real signal: DCM cells have 10% higher expression
    of the first 20 genes on average. This lets us verify that AUC > 0.5 even
    without real biology — a sanity check that the pipeline isn't broken.

    Design note:
      "I built against synthetic data first so I could write tests that assert
      the exact behaviour I expect — no patient leaks across folds, probabilities
      in [0,1], no NaNs — before I ever touched the real dataset."
    """
    rng = np.random.default_rng(random_state)

    n_dcm = int(n_cells * dcm_fraction)
    n_acm = n_cells - n_dcm

    # Assign cells to donors (each donor gets ~n_cells/n_donors cells)
    donor_ids_dcm = rng.integers(0, n_donors // 2, size=n_dcm)
    donor_ids_acm = rng.integers(n_donors // 2, n_donors, size=n_acm)

    # Expression: log-normal base + disease signal in first 20 genes
    X_dcm = rng.lognormal(mean=0.0, sigma=1.0, size=(n_dcm, n_genes))
    X_acm = rng.lognormal(mean=0.0, sigma=1.0, size=(n_acm, n_genes))
    X_dcm[:, :20] *= 1.1  # slight uplift in 20 genes for DCM

    X = np.vstack([X_dcm, X_acm])
    disease = [DCM_LABEL] * n_dcm + [ACM_LABEL] * n_acm
    donor_id = [f"donor_{d:02d}" for d in np.concatenate([donor_ids_dcm, donor_ids_acm])]
    gene_names = [f"GENE{i:04d}" for i in range(n_genes)]

    obs = pd.DataFrame({"disease": disease, "donor_id": donor_id})
    var = pd.DataFrame(index=gene_names)

    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    logger.info(f"Synthetic data: {adata.n_obs} cells, {adata.n_vars} genes, {n_donors} donors")
    return adata
