"""
run_experiment.py — CLI entry point for the DCMvsACM generalization study.

Usage:
  # Run on synthetic data (no download needed — for testing pipeline logic):
  python run_experiment.py --synthetic

  # Run on real Reichart 2022 data:
  python run_experiment.py --data cardiomyocytes.h5ad

  # Fast iteration with a subset:
  python run_experiment.py --data cardiomyocytes.h5ad --n-cells 5000

  # Full run:
  python run_experiment.py --data cardiomyocytes.h5ad --n-genes 2500 --n-splits 5

Outputs:
  results/comparison_table.csv   — the naive vs honest AUC table (the finding)
  results/fold_results.csv       — per-fold AUC for each classifier + split type
  (plots generated when matplotlib is available)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- set up sys.path so imports work whether called from project root or here ---
sys.path.insert(0, str(Path(__file__).parent))

from data import extract_arrays, load_and_filter, make_synthetic_data
from evaluation import (
    build_comparison_table,
    evaluate_honest,
    evaluate_naive,
    print_comparison_table,
)
from models import get_all_pipelines


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def save_results(
    comparison_df: pd.DataFrame,
    fold_rows: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)

    fold_df = pd.DataFrame(fold_rows)
    fold_path = output_dir / "fold_results.csv"
    fold_df.to_csv(fold_path, index=False)

    logging.getLogger(__name__).info(f"Results saved to {output_dir}/")
    logging.getLogger(__name__).info(f"  {comparison_path}")
    logging.getLogger(__name__).info(f"  {fold_path}")


def run(
    data_path: str | None,
    synthetic: bool,
    n_cells: int | None,
    n_genes: int,
    n_splits: int,
    random_state: int,
    output_dir: Path,
    verbose: bool,
) -> pd.DataFrame:
    """
    Main experiment runner. Returns the comparison DataFrame.
    Separated from CLI args for testability.
    """
    setup_logging(verbose)
    logger = logging.getLogger(__name__)

    # --- 1. Load data ---
    if synthetic:
        logger.info("Using SYNTHETIC data — pipeline logic test mode.")
        adata = make_synthetic_data(
            n_cells=n_cells or 2000,
            n_genes=min(n_genes, 500),
            n_donors=20,
            random_state=random_state,
        )
    else:
        if data_path is None:
            raise ValueError("Provide --data <path.h5ad> or use --synthetic.")
        adata = load_and_filter(data_path, n_cells=n_cells, random_state=random_state)

    # Pre-select HVGs on the sparse matrix before densifying when using real data.
    # The full 32k-gene atlas would require ~40 GB dense RAM; pre-selecting to
    # n_genes columns first reduces this to ~1.6 GB for the cardiomyocyte atlas.
    # The in-fold HVGSelector still runs (refining on train-fold only), but since
    # we pre-selected the same n_genes, it essentially re-confirms the selection.
    presort_genes = n_genes if not synthetic else None
    X, y, groups = extract_arrays(adata, n_top_genes=presort_genes)
    logger.info(f"Working array: X={X.shape}, y classes={np.bincount(y).tolist()}, "
                f"n_donors={len(np.unique(groups))}")

    # --- 2. Get all pipelines ---
    pipelines = get_all_pipelines(n_top_genes=n_genes, random_state=random_state)

    # --- 3. Run naive + honest evaluation for each classifier ---
    all_results = []
    fold_rows = []

    for clf_name, pipeline in pipelines.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Classifier: {clf_name}")

        # NAIVE — cell-level split (the original, inflated approach)
        naive_result = evaluate_naive(
            pipeline, X, y,
            classifier_name=clf_name,
            n_splits=n_splits,
            random_state=random_state,
        )
        all_results.append(naive_result)
        for fr in naive_result.fold_results:
            fold_rows.append({
                "classifier": clf_name,
                "split_type": "naive",
                "fold": fr.fold,
                "roc_auc": fr.roc_auc,
                "avg_precision": fr.avg_precision,
                "n_train": fr.n_train,
                "n_val": fr.n_val,
                "n_donors_val": fr.n_donors_val,
            })

        # HONEST — patient-level split (the corrected approach)
        honest_result = evaluate_honest(
            pipeline, X, y, groups,
            classifier_name=clf_name,
            n_splits=n_splits,
            random_state=random_state,
        )
        all_results.append(honest_result)
        for fr in honest_result.fold_results:
            fold_rows.append({
                "classifier": clf_name,
                "split_type": "honest",
                "fold": fr.fold,
                "roc_auc": fr.roc_auc,
                "avg_precision": fr.avg_precision,
                "n_train": fr.n_train,
                "n_val": fr.n_val,
                "n_donors_val": fr.n_donors_val,
            })

    # --- 4. Build and display comparison table ---
    comparison_df = build_comparison_table(all_results)
    print_comparison_table(comparison_df)

    # --- 5. Save results ---
    save_results(comparison_df, fold_rows, output_dir)

    return comparison_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DCMvsACM generalization study: naive vs honest evaluation"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to Reichart 2022 .h5ad file"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no download needed; tests pipeline logic)"
    )
    parser.add_argument(
        "--n-cells", type=int, default=None,
        help="Subsample to N cells after filtering (None = all)"
    )
    parser.add_argument(
        "--n-genes", type=int, default=2500,
        help="Number of highly variable genes to select per fold (default: 2500)"
    )
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results (default: results/)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show DEBUG-level logs"
    )
    args = parser.parse_args()

    if not args.synthetic and args.data is None:
        parser.error("Provide --data <path.h5ad> or use --synthetic")

    run(
        data_path=args.data,
        synthetic=args.synthetic,
        n_cells=args.n_cells,
        n_genes=args.n_genes,
        n_splits=args.n_splits,
        random_state=args.seed,
        output_dir=Path(args.output_dir),
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
