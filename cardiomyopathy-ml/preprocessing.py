"""
preprocessing.py — Leakage-free preprocessing for the DCMvsACM pipeline.

Computational biology note:
  The cardinal rule: ALL preprocessing decisions that look at the data
  must be fit on the training fold only, then applied to the validation fold.

  This module implements two preprocessing steps as sklearn-compatible
  transformers so they can live inside an imblearn Pipeline:

  1. HVGSelector — selects highly variable genes (scanpy logic).
     Why inside the fold: "highly variable" is computed from expression
     variance across cells. If you compute it on the full dataset, you
     use test-cell variance to choose features. Subtle but real leakage.

  2. StandardScaler — zero-mean, unit-variance scaling per gene.
     Why inside the fold: the mean and SD used for scaling should come
     from training cells only. Standard sklearn practice, extended here
     to the bioinformatics context.

  Interview talking point:
    "HVG selection inside the fold is the subtlety most published pipelines
    miss — it's a data-driven feature selection step, so it belongs inside
    cross-validation just like the scaler does."
"""

import logging
from typing import Optional

import numpy as np
import anndata as ad
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class HVGSelector(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that selects highly variable genes.

    Wraps scanpy's pp.highly_variable_genes logic so it can be placed inside
    an imblearn/sklearn Pipeline and correctly fit only on training data.

    Parameters:
        n_top_genes: Number of highly variable genes to select.
                     2500 was used in the original pipeline; we keep this
                     as the default but make it configurable.
        flavor: 'seurat_v3' uses the Seurat v3 HVG algorithm (variance
                stabilising transform), which is the current best practice
                for count data. 'seurat' (older) is the classic
                normalised-dispersion approach.

    Attributes (set after fit):
        selected_genes_: Boolean mask of shape (n_genes,) indicating which
                         genes were selected. Stored so we can apply the
                         same selection to the test fold.
    """

    def __init__(self, n_top_genes: int = 2500, flavor: str = "seurat_v3"):
        self.n_top_genes = n_top_genes
        self.flavor = flavor

    def fit(self, X: np.ndarray, y=None):
        """
        Identify highly variable genes from training data only.

        We use a variance/mean ratio (Fano factor) approach rather than
        calling scanpy directly inside the fold. This is numerically stable
        on any expression scale, doesn't require count data assumptions,
        and avoids the binning overflow that scanpy's seurat flavor hits on
        lognormal or other non-count inputs.

        Computational biology note:
          Scanpy's HVG algorithms (seurat, seurat_v3) are designed for
          count matrices that have been log-normalised in a specific way.
          Using them inside a pipeline on arbitrary floats causes numerical
          issues (overflow in expm1). Our Fano-factor approach is equivalent
          in spirit (high variance relative to mean = informative gene) and
          is numerically robust for any input scale.
        """
        n_genes = X.shape[1]
        effective_top = min(self.n_top_genes, n_genes)

        # Fano factor = variance / mean, robust to any expression scale
        # Clip to avoid division by zero and numerical issues
        means = np.mean(X, axis=0)
        variances = np.var(X, axis=0)
        safe_means = np.clip(means, 1e-10, None)
        fano = variances / safe_means

        # Select top-k genes by Fano factor
        top_indices = np.argsort(fano)[-effective_top:]
        self.selected_genes_ = np.zeros(n_genes, dtype=bool)
        self.selected_genes_[top_indices] = True

        n_selected = self.selected_genes_.sum()
        logger.debug(f"HVGSelector: selected {n_selected}/{n_genes} genes (Fano factor)")
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Apply the gene selection learned during fit."""
        if not hasattr(self, "selected_genes_"):
            raise ValueError("HVGSelector must be fit before transform.")
        return X[:, self.selected_genes_]

    def get_feature_names_out(self, input_features=None):
        """Return indices of selected genes (for SHAP feature naming)."""
        return np.where(self.selected_genes_)[0]


def make_preprocessor(n_top_genes: int = 2500) -> "Pipeline":
    """
    Build a preprocessing pipeline: HVG selection -> StandardScaler.

    This is a sklearn Pipeline (not imblearn) — it handles only preprocessing,
    not resampling. SMOTE is added in models.py when building the full pipeline.

    Returns a fitted-able pipeline that can be placed as the first steps of
    the full imblearn Pipeline in models.py.
    """
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("hvg", HVGSelector(n_top_genes=n_top_genes)),
        ("scaler", StandardScaler()),
    ])
