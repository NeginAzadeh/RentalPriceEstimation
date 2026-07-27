"""
Visualization utilities for the Rental Price Estimation project.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"


MODEL_RESULTS = {
    "LightGBM": 163.049,
    "CatBoost": 189.799,
    "Random Forest": 199.626,
    "Dummy": 545.700,
}


def _prepare_output_directory(
    output_dir: str | Path | None = None,
) -> Path:
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_FIGURE_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def plot_price_distribution(
    development: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Plot the rental-price distribution.

    The x-axis is limited to the 99th percentile so that a small
    number of extreme listings do not dominate the visualization.
    """

    output_dir = _prepare_output_directory(output_dir)

    prices = development["price"].dropna()
    upper_limit = prices.quantile(0.99)

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.hist(
        prices[prices <= upper_limit],
        bins=50,
        edgecolor="black",
        alpha=0.8,
    )

    axis.set_title(
        "Distribution of Monthly Rental Prices"
    )
    axis.set_xlabel("Monthly Rent")
    axis.set_ylabel("Number of Listings")

    figure.tight_layout()

    output_path = (
        output_dir
        / "price_distribution.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_price_vs_square_feet(
    development: pd.DataFrame,
    output_dir: str | Path | None = None,
    sample_size: int = 5000,
) -> Path:
    """
    Plot rental price against apartment size.

    Extreme values are removed only for visualization.
    """

    output_dir = _prepare_output_directory(output_dir)

    data = development[
        ["square_feet", "price"]
    ].dropna()

    square_feet_limit = data[
        "square_feet"
    ].quantile(0.99)

    price_limit = data[
        "price"
    ].quantile(0.99)

    data = data[
        (data["square_feet"] <= square_feet_limit)
        & (data["price"] <= price_limit)
    ]

    if len(data) > sample_size:
        data = data.sample(
            n=sample_size,
            random_state=42,
        )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.scatter(
        data["square_feet"],
        data["price"],
        alpha=0.25,
        s=12,
    )

    axis.set_title(
        "Rental Price vs. Apartment Size"
    )
    axis.set_xlabel("Square Feet")
    axis.set_ylabel("Monthly Rent")

    figure.tight_layout()

    output_path = (
        output_dir
        / "price_vs_square_feet.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def plot_model_comparison(
    output_dir: str | Path | None = None,
) -> Path:
    """
    Plot reconstructed cross-validation MAE.

    Lower MAE indicates better performance.
    """

    output_dir = _prepare_output_directory(output_dir)

    results = (
        pd.Series(MODEL_RESULTS)
        .sort_values()
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    bars = axis.bar(
        results.index,
        results.values,
    )

    axis.set_title(
        "Model Comparison — 5-Fold CV"
    )
    axis.set_ylabel("Mean Absolute Error (MAE)")
    axis.set_xlabel("Model")

    for bar, value in zip(
        bars,
        results.values,
    ):
        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 7,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    output_path = (
        output_dir
        / "model_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def generate_all_figures(
    development: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """
    Generate all project figures.
    """

    return [
        plot_price_distribution(
            development,
            output_dir,
        ),
        plot_price_vs_square_feet(
            development,
            output_dir,
        ),
        plot_model_comparison(
            output_dir,
        ),
    ]