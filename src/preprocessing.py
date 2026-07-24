"""
Data loading, sanity checks, and basic cleaning utilities
for the rental price estimation project.
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

TARGET_COLUMN = "price"

OUTLIER_COLUMNS = [
    "square_feet",
    "bedrooms",
    "bathrooms",
    TARGET_COLUMN,
]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_datasets(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the development and evaluation datasets.

    Parameters
    ----------
    data_dir:
        Directory containing development.csv and evaluation.csv.
        If not provided, the project's data/ directory is used.

    Returns
    -------
    development:
        Development dataset containing the target variable.
    evaluation:
        Evaluation dataset without the target variable.
    """

    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR

    development_path = data_dir / "development.csv"
    evaluation_path = data_dir / "evaluation.csv"

    if not development_path.exists():
        raise FileNotFoundError(
            f"Development dataset not found: {development_path}"
        )

    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {evaluation_path}"
        )

    development = pd.read_csv(development_path)
    evaluation = pd.read_csv(evaluation_path)

    if TARGET_COLUMN not in development.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing "
            "from the development dataset."
        )

    return development, evaluation


# ---------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------

def missing_value_percentages(df: pd.DataFrame) -> pd.Series:
    """Return the percentage of missing values for each column."""

    return (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )


def count_iqr_outliers(series: pd.Series) -> int:
    """
    Count outliers using the 1.5 * IQR rule.
    """

    clean_series = series.dropna()

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return int(
        (
            (clean_series < lower_bound)
            | (clean_series > upper_bound)
        ).sum()
    )


def run_sanity_checks(df: pd.DataFrame) -> None:
    """
    Print the main data-quality checks used in the original project.
    """

    print("\n=== Missing values ===")
    print(missing_value_percentages(df).head(10))

    duplicate_count = int(df.duplicated().sum())

    print("\n=== Duplicate rows ===")
    print(f"Exact duplicates: {duplicate_count}")

    print("\n=== IQR outliers ===")

    for column in OUTLIER_COLUMNS:
        if column not in df.columns:
            continue

        count = count_iqr_outliers(df[column])
        percentage = 100 * count / len(df)

        print(
            f"{column}: "
            f"{count} outliers "
            f"({percentage:.1f}%)"
        )


# ---------------------------------------------------------------------
# Basic cleaning
# ---------------------------------------------------------------------

def _convert_binary_column(series: pd.Series) -> pd.Series:
    """
    Convert Yes/No, True/False and 1/0 representations to integers.
    """

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    }

    converted = normalized.map(mapping)

    # Keep already-numeric values when possible.
    numeric_values = pd.to_numeric(series, errors="coerce")
    converted = converted.fillna(numeric_values)

    return converted.fillna(0).astype("int8")


def clean_base_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic basic cleaning.

    This stage does not use the target variable, which makes it safe
    to apply independently to development and evaluation data.
    """

    cleaned = df.copy()

    # Address is more than 90% missing in the original dataset.
    cleaned = cleaned.drop(
        columns=["address"],
        errors="ignore",
    )

    # Preserve information about missing pet-policy values.
    if "pets_allowed" in cleaned.columns:
        cleaned["pets_allowed_isna"] = (
            cleaned["pets_allowed"]
            .isna()
            .astype("int8")
        )

        cleaned["pets_allowed"] = (
            cleaned["pets_allowed"]
            .fillna("Unknown")
        )

    # Low-missing categorical location features.
    for column in ("state", "cityname"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("Unknown")

    # Missing text represents unavailable information.
    for column in ("amenities", "body"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].fillna("")

    # Convert binary categorical variables into numeric form.
    for column in ("fee", "has_photo"):
        if column in cleaned.columns:
            cleaned[column] = _convert_binary_column(
                cleaned[column]
            )

    return cleaned


def prepare_base_datasets(
    data_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean development/evaluation datasets.
    """

    development, evaluation = load_datasets(data_dir)

    development = clean_base_data(development)
    evaluation = clean_base_data(evaluation)

    return development, evaluation