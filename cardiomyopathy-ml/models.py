"""
models.py — Leakage-free ML pipelines for the DCMvsACM project.

Computational biology note:
  The imblearn Pipeline is NOT the same as sklearn's Pipeline.
  imblearn's Pipeline knows that SMOTE is a resampling step that should
  ONLY run during fit(), not during predict() or transform(). If you use
  sklearn's Pipeline with SMOTE, it works, but the intent is clearer with
  imblearn because it was designed for this pattern.

  Pipeline order per fold:
    1. HVGSelector   — fit on train fold, selects n_top_genes
    2. StandardScaler — fit on train fold, zero-mean unit-variance
    3. SMOTE          — fit on train fold ONLY, synthesises minority cells
    4. Classifier     — fit on SMOTE'd train fold, predict on raw val fold

  Steps 1-2 are transformer steps (imblearn applies them to val fold too,
  but only the transform — not the fit). Step 3 (SMOTE) is a resample step;
  imblearn knows to skip it during predict/transform on val data. This is
  exactly the correct behaviour.

  Interview talking point:
    "SMOTE inside an imblearn Pipeline means the synthetic cells never appear
    in the validation fold. This is the fix for the original code's leakage,
    where SMOTE ran on the full dataset before the split."
"""

import logging
from typing import Optional

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocessing import HVGSelector

logger = logging.getLogger(__name__)


def build_pipeline(
    classifier_name: str,
    n_top_genes: int = 2500,
    smote_k_neighbors: int = 5,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Build a complete imblearn Pipeline: HVG -> Scale -> SMOTE -> Classifier.

    The pipeline is not yet fit — it is returned ready for use inside
    StratifiedGroupKFold. Fitting happens per-fold in evaluation.py.

    Args:
        classifier_name: One of 'rf', 'xgb', 'logreg'.
        n_top_genes: Number of highly variable genes to select.
        smote_k_neighbors: SMOTE k parameter. 5 is the default; reduce to 3
                           if minority class has very few samples per fold.
        random_state: For reproducibility.

    Returns:
        ImbPipeline ready to be fit on a training fold.
    """
    classifiers = {
        "rf": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "xgb": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            n_jobs=-1,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        ),
        "logreg": LogisticRegression(
            C=1.0,
            max_iter=1000,
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    if classifier_name not in classifiers:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. "
            f"Choose from: {list(classifiers.keys())}"
        )

    clf = classifiers[classifier_name]

    # Note: SMOTE k_neighbors must be < minority class size in each fold.
    # If a fold has very few ACM cells, this will raise; handled in evaluation.py.
    smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)

    pipeline = ImbPipeline([
        ("hvg", HVGSelector(n_top_genes=n_top_genes)),
        ("scaler", StandardScaler()),
        ("smote", smote),
        ("clf", clf),
    ])

    logger.debug(f"Built pipeline: HVG({n_top_genes}) -> Scale -> SMOTE -> {classifier_name}")
    return pipeline


def get_all_pipelines(n_top_genes: int = 2500, random_state: int = 42) -> dict:
    """
    Return all three classifier pipelines, keyed by name.
    Used by evaluation.py to iterate over classifiers.
    """
    return {
        "Random Forest": build_pipeline("rf", n_top_genes, random_state=random_state),
        "XGBoost": build_pipeline("xgb", n_top_genes, random_state=random_state),
        "Logistic Regression": build_pipeline("logreg", n_top_genes, random_state=random_state),
    }
