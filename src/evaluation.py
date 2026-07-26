"""
Evaluation utilities for the rental price estimation project.
"""

from __future__ import annotations

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.modeling import build_model_pipeline


def evaluate_model_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, object]:
    """
    Evaluate a regression model using cross-validation.

    The complete pipeline is refitted independently inside each fold.
    Target transformation is handled internally by the model pipeline.

    MAE is therefore reported directly on the original rental-price scale.
    """

    pipeline = build_model_pipeline(
        model_name=model_name
    )

    price_deciles = pd.qcut(
        y,
        q=10,
        labels=False,
        duplicates="drop",
    )

    stratified_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    cv = list(
        stratified_cv.split(
            X,
            price_deciles,
        )
    )

    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=n_jobs,
    )

    fold_mae = -scores

    return {
        "model": model_name,
        "fold_mae": fold_mae,
        "mean_mae": float(
            fold_mae.mean()
        ),
        "std_mae": float(
            fold_mae.std()
        ),
    }


def print_cv_results(
    results: dict[str, object],
) -> None:
    """
    Print cross-validation results in a readable format.
    """

    print(
        f"\n=== {results['model']} ==="
    )

    print(
        "Fold MAE:",
        results["fold_mae"],
    )

    print(
        f"Mean MAE: "
        f"{results['mean_mae']:.3f}"
    )

    print(
        f"Std MAE: "
        f"{results['std_mae']:.3f}"
    )