"""
evaluation.py — Honest evaluation for the DCMvsACM pipeline.

This is the intellectual heart of the project. It implements:

  1. NAIVE evaluation (random cell-level split) — what the original code did.
     Inflated because cells from the same patient land in both train/test.

  2. HONEST evaluation (patient-level split via StratifiedGroupKFold) — what
     the project now does. All cells from a given patient stay together,
     so the model is always evaluated on patients it has never seen.

  The gap between the two numbers IS THE FINDING.

Computational biology note:
  StratifiedGroupKFold (sklearn >= 1.0):
    - 'Group' means all cells from the same donor stay in the same fold.
    - 'Stratified' means the DCM:ACM ratio is preserved across folds.
  We need both: without Stratified, one fold could be all-DCM and
  AUC would be undefined or misleading.

  n_splits=5 means ~80% train / 20% val per fold. With ~79 donors in
  Reichart 2022, each fold uses ~63 donors for training and ~16 for val.
  This is a realistic held-out set size.

  With synthetic data (20 donors, 5 splits), each val fold has ~4 donors.
  Small but sufficient to test the pipeline logic.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""
    fold: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray  # probability of positive class (DCM)
    n_train: int
    n_val: int
    n_donors_val: int
    roc_auc: float = field(init=False)
    avg_precision: float = field(init=False)

    def __post_init__(self):
        self.roc_auc = roc_auc_score(self.y_true, self.y_prob)
        self.avg_precision = average_precision_score(self.y_true, self.y_prob)


@dataclass
class EvalResult:
    """Aggregated results across all folds for one classifier + split strategy."""
    classifier_name: str
    split_type: str  # 'naive_cell_level' or 'honest_patient_level'
    fold_results: list[FoldResult]

    @property
    def mean_auc(self) -> float:
        return float(np.mean([f.roc_auc for f in self.fold_results]))

    @property
    def std_auc(self) -> float:
        return float(np.std([f.roc_auc for f in self.fold_results]))

    @property
    def mean_avg_precision(self) -> float:
        return float(np.mean([f.avg_precision for f in self.fold_results]))

    def summary_row(self) -> dict:
        return {
            "classifier": self.classifier_name,
            "split_type": self.split_type,
            "mean_roc_auc": round(self.mean_auc, 4),
            "std_roc_auc": round(self.std_auc, 4),
            "ci_95": f"{self.mean_auc:.3f} ± {2 * self.std_auc:.3f}",
            "mean_avg_precision": round(self.mean_avg_precision, 4),
            "n_folds": len(self.fold_results),
        }


def _run_cv(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    cv,
    classifier_name: str,
    split_type: str,
    smote_k_fallback: int = 3,
) -> EvalResult:
    """
    Internal: run cross-validation with given splitter and collect FoldResults.
    """
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        n_donors_val = (
            len(np.unique(groups[val_idx])) if groups is not None else len(val_idx)
        )

        # Clone the pipeline so each fold starts fresh (no state bleed)
        from sklearn.base import clone
        fold_pipeline = clone(pipeline)

        # SMOTE can fail if minority class has < k_neighbors samples in train fold.
        # Fall back to fewer neighbors if needed.
        minority_count = int((y_train == 1).sum())
        if minority_count < 5:
            logger.warning(
                f"Fold {fold_idx}: only {minority_count} minority samples in train. "
                f"Reducing SMOTE k_neighbors to {smote_k_fallback}."
            )
            fold_pipeline.set_params(smote__k_neighbors=smote_k_fallback)

        try:
            fold_pipeline.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Fold {fold_idx} fit failed: {e}")
            raise

        y_prob = fold_pipeline.predict_proba(X_val)[:, 1]
        y_pred = fold_pipeline.predict(X_val)

        result = FoldResult(
            fold=fold_idx,
            y_true=y_val,
            y_pred=y_pred,
            y_prob=y_prob,
            n_train=len(train_idx),
            n_val=len(val_idx),
            n_donors_val=n_donors_val,
        )
        fold_results.append(result)

        logger.info(
            f"  [{split_type}] {classifier_name} | Fold {fold_idx+1} | "
            f"train={len(train_idx)}, val={len(val_idx)} donors_val={n_donors_val} | "
            f"AUC={result.roc_auc:.4f}"
        )

    return EvalResult(
        classifier_name=classifier_name,
        split_type=split_type,
        fold_results=fold_results,
    )


def evaluate_naive(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> EvalResult:
    """
    NAIVE evaluation: random cell-level stratified K-fold.

    This is what the original code did — cells from the same patient
    land on both sides of the split. AUC is inflated because the model
    can partially recognise patients it has already seen.

    We compute this explicitly so we can show the inflation.

    Interview talking point:
      "The naive AUC was X. After switching to patient-level splitting,
      it dropped to Y. The gap of Z is how much the original code was
      cheating — the model was memorising patients, not learning disease."
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    logger.info(f"Running NAIVE (cell-level) CV for {classifier_name} ...")
    return _run_cv(pipeline, X, y, groups=None, cv=cv,
                   classifier_name=classifier_name, split_type="naive_cell_level")


def evaluate_honest(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classifier_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> EvalResult:
    """
    HONEST evaluation: patient-level StratifiedGroupKFold.

    All cells from the same donor stay in the same fold. The model is
    evaluated only on patients it has never seen in training.

    'Stratified' = DCM:ACM ratio preserved across folds.
    'Group' = donor integrity maintained.

    This is the correct evaluation for clinical ML.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits)
    logger.info(f"Running HONEST (patient-level) CV for {classifier_name} ...")
    return _run_cv(pipeline, X, y, groups=groups, cv=cv,
                   classifier_name=classifier_name, split_type="honest_patient_level")


def build_comparison_table(results: list[EvalResult]) -> pd.DataFrame:
    """
    Build the key deliverable: naive vs honest AUC comparison table.

    This table is the finding. A large gap means the naive evaluation
    was inflated by patient-level data leakage.

    Returns:
        DataFrame with columns: classifier, split_type, mean_roc_auc,
        std_roc_auc, ci_95, mean_avg_precision, n_folds.
    """
    rows = [r.summary_row() for r in results]
    df = pd.DataFrame(rows)

    # Compute leakage inflation: naive AUC - honest AUC per classifier
    naive = df[df["split_type"] == "naive_cell_level"].set_index("classifier")
    honest = df[df["split_type"] == "honest_patient_level"].set_index("classifier")

    inflation_rows = []
    for clf in naive.index.intersection(honest.index):
        inflation_rows.append({
            "classifier": clf,
            "split_type": "LEAKAGE_INFLATION",
            "mean_roc_auc": round(naive.loc[clf, "mean_roc_auc"] - honest.loc[clf, "mean_roc_auc"], 4),
            "std_roc_auc": None,
            "ci_95": f"naive-honest = {naive.loc[clf, 'mean_roc_auc']:.3f} - {honest.loc[clf, 'mean_roc_auc']:.3f}",
            "mean_avg_precision": None,
            "n_folds": None,
        })

    if inflation_rows:
        df = pd.concat([df, pd.DataFrame(inflation_rows)], ignore_index=True)

    return df


def print_comparison_table(df: pd.DataFrame) -> None:
    """Pretty-print the comparison table to stdout."""
    print("\n" + "=" * 70)
    print("NAIVE vs HONEST EVALUATION — THE KEY FINDING")
    print("=" * 70)

    for split_type in ["naive_cell_level", "honest_patient_level", "LEAKAGE_INFLATION"]:
        subset = df[df["split_type"] == split_type]
        if subset.empty:
            continue

        label = {
            "naive_cell_level": "NAIVE (random cell split — inflated)",
            "honest_patient_level": "HONEST (patient-level split)",
            "LEAKAGE_INFLATION": "LEAKAGE INFLATION (naive - honest)",
        }[split_type]

        print(f"\n{label}:")
        for _, row in subset.iterrows():
            if split_type == "LEAKAGE_INFLATION":
                print(f"  {row['classifier']:25s}  +{row['mean_roc_auc']:.4f} AUC  |  {row['ci_95']}")
            else:
                print(f"  {row['classifier']:25s}  AUC {row['ci_95']}  |  AP {row['mean_avg_precision']:.4f}")

    print("=" * 70 + "\n")
