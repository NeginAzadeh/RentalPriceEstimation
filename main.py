"""
Command-line entry point for the Rental Price Estimation project.

Examples
--------
Run data sanity checks:
    python main.py --mode check

Evaluate LightGBM:
    python main.py --mode evaluate --model lightgbm

Train the final model and generate predictions:
    python main.py --mode predict --model lightgbm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from src.visualization import generate_all_figures
from src.evaluation import evaluate_model_cv, print_cv_results
from src.modeling import build_model_pipeline
from src.preprocessing import prepare_base_datasets, run_sanity_checks


AVAILABLE_MODELS = [
    "dummy",
    "linear_regression",
    "ridge",
    "random_forest",
    "catboost",
    "lightgbm",
]

OUTPUT_DIR = Path("outputs") / "submissions"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rental Price Estimation"
    )

    parser.add_argument(
        "--mode",
        choices=["check", "evaluate", "predict", "visualize"],
        default="evaluate",
        help="Operation to perform.",
    )

    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        default="lightgbm",
        help="Model to use.",
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )

    return parser.parse_args()


def run_data_check() -> None:
    development, evaluation = prepare_base_datasets()

    print("\n=== Development dataset ===")
    print(f"Shape: {development.shape}")

    run_sanity_checks(development)

    print("\n=== Evaluation dataset ===")
    print(f"Shape: {evaluation.shape}")


def run_evaluation(
    model_name: str,
    n_splits: int,
) -> None:
    development, _ = prepare_base_datasets()

    X = development.drop(columns=["price"])
    y = development["price"]

    results = evaluate_model_cv(
        X=X,
        y=y,
        model_name=model_name,
        n_splits=n_splits,
        n_jobs=1,
    )

    print_cv_results(results)


def run_prediction(
    model_name: str,
) -> None:
    development, evaluation = prepare_base_datasets()

    X_train = development.drop(columns=["price"])
    y_train = development["price"]

    model = build_model_pipeline(
        model_name=model_name
    )

    print(
        f"\nTraining {model_name} "
        "on the full development dataset..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Generating evaluation predictions...")

    predictions = model.predict(
        evaluation
    )

    submission = pd.DataFrame(
        {
            "Id": evaluation["id"],
            "Predicted": predictions,
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"submission_{model_name}.csv"
    )

    submission.to_csv(
        output_path,
        index=False,
    )

    print("\n=== Prediction complete ===")
    print(f"Rows: {len(submission)}")
    print(f"Saved to: {output_path}")

    print("\nFirst 5 predictions:")
    print(submission.head())


def run_visualization() -> None:
    development, _ = prepare_base_datasets()

    figure_paths = generate_all_figures(
        development
    )

    print("\n=== Figures generated ===")

    for path in figure_paths:
        print(path)

def main() -> None:
    args = parse_arguments()

    if args.mode == "check":
        run_data_check()

    elif args.mode == "evaluate":
        run_evaluation(
            model_name=args.model,
            n_splits=args.folds,
        )

    elif args.mode == "predict":
        run_prediction(
            model_name=args.model,
        )

    elif args.mode == "visualize":
        run_visualization()


if __name__ == "__main__":
    main()