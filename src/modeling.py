"""
Model definitions and machine-learning pipeline construction
for the rental price estimation project.
"""

from __future__ import annotations

import numpy as np

from catboost import CatBoostRegressor
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import RentalFeatureEngineer


# ---------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------

NUMERIC_FEATURES = [
    "square_feet",
    "bedrooms",
    "bathrooms",
    "fee",
    "has_photo",
    "pets_allowed_isna",
    "city_price_mean",
    "city_price_std",
    "city_listing_count",
    "geo_price_mean",
    "geo_price_std",
    "beds_per_bath_ratio",
    "latitude",
    "longitude",
]

LOW_CARDINALITY_CATEGORICALS = [
    "currency",
    "price_type",
    "pets_allowed",
]

HIGH_CARDINALITY_CATEGORICALS = [
    "cityname",
    "amenities",
    "category",
    "source",
]

TEXT_FEATURE = "body"


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing stage used before regression.

    Numerical features:
        median imputation + standardization

    Low-cardinality categorical features:
        constant imputation + one-hot encoding

    High-cardinality categorical features:
        constant imputation + target encoding

    Text:
        TF-IDF + truncated SVD
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    low_cardinality_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    high_cardinality_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "target_encoder",
                TargetEncoder(
                    handle_unknown="value",
                    handle_missing="value",
                ),
            ),
        ]
    )

    text_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    token_pattern=r"(?u)\b\w\w+\b",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=40_000,
                    stop_words="english",
                ),
            ),
            (
                "svd",
                TruncatedSVD(
                    n_components=300,
                    random_state=42,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "low_cardinality",
                low_cardinality_pipeline,
                LOW_CARDINALITY_CATEGORICALS,
            ),
            (
                "high_cardinality",
                high_cardinality_pipeline,
                HIGH_CARDINALITY_CATEGORICALS,
            ),
            (
                "text",
                text_pipeline,
                TEXT_FEATURE,
            ),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def get_models() -> dict[str, object]:
    """
    Return the regression models used for comparison.
    """

    return {
        "dummy": DummyRegressor(
            strategy="mean",
        ),

        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=25,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),

        "catboost": CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
        ),

        "lightgbm": LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=256,
            subsample=0.8,
            colsample_bytree=0.6,
            objective="mae",
            random_state=42,
            verbosity=-1,
        ),
    }


# ---------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------

def build_model_pipeline(
    model_name: str = "lightgbm",
) -> Pipeline:
    """
    Build the complete leakage-safe modeling pipeline.
    """

    models = get_models()

    if model_name not in models:
        available = ", ".join(models.keys())

        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    base_model = models[model_name]

    # The Dummy baseline should predict the arithmetic mean
    # of rental prices on the original target scale.
    if model_name == "dummy":
        final_model = base_model
    else:
        final_model = TransformedTargetRegressor(
            regressor=base_model,
            func=np.log1p,
            inverse_func=np.expm1,
        )

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                RentalFeatureEngineer(
                    n_geo_clusters=200,
                    random_state=42,
                ),
            ),
            (
                "preprocessing",
                build_preprocessor(),
            ),
            (
                "model",
                final_model,
            ),
        ]
    )