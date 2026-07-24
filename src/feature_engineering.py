"""
Feature engineering for the rental price estimation project.

Target-dependent features are implemented as a scikit-learn transformer
so that they can be fitted only on training data during cross-validation,
preventing target leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted


class RentalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create the main engineered features used in the original project.

    Features
    --------
    City-level:
        - city_price_mean
        - city_price_std
        - city_listing_count

    Geographic:
        - geo_cluster
        - geo_price_mean
        - geo_price_std

    Property:
        - beds_per_bath_ratio

    The transformer learns all target-dependent statistics only during fit().
    """

    def __init__(
        self,
        n_geo_clusters: int = 200,
        random_state: int = 42,
    ):
        self.n_geo_clusters = n_geo_clusters
        self.random_state = random_state

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
    ) -> "RentalFeatureEngineer":
        """
        Learn city statistics, coordinate imputations, and geographic clusters.
        """

        if y is None:
            raise ValueError(
                "RentalFeatureEngineer requires the target values during fit()."
            )

        X = X.copy()

        y = pd.Series(
            np.asarray(y),
            index=X.index,
            name="_target",
        )

        # -------------------------------------------------------------
        # Global target statistics
        # -------------------------------------------------------------

        self.global_price_mean_ = float(y.mean())
        self.global_price_std_ = float(y.std())

        # -------------------------------------------------------------
        # City-level target statistics
        # -------------------------------------------------------------

        city_data = pd.DataFrame(
            {
                "cityname": X["cityname"],
                "_target": y,
            },
            index=X.index,
        )

        self.city_statistics_ = (
            city_data
            .groupby("cityname")["_target"]
            .agg(
                city_price_mean="mean",
                city_price_std="std",
                city_listing_count="size",
            )
        )

        # -------------------------------------------------------------
        # Geographic preprocessing
        # -------------------------------------------------------------

        latitude = pd.to_numeric(
            X["latitude"],
            errors="coerce",
        )

        longitude = pd.to_numeric(
            X["longitude"],
            errors="coerce",
        )

        self.latitude_median_ = float(latitude.median())
        self.longitude_median_ = float(longitude.median())

        coordinates = pd.DataFrame(
            {
                "latitude": latitude.fillna(
                    self.latitude_median_
                ),
                "longitude": longitude.fillna(
                    self.longitude_median_
                ),
            },
            index=X.index,
        )

        # Prevent an invalid number of clusters on very small samples.
        n_clusters = min(
            self.n_geo_clusters,
            len(coordinates),
        )

        self.geo_kmeans_ = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )

        geo_labels = self.geo_kmeans_.fit_predict(
            coordinates
        )

        # -------------------------------------------------------------
        # Geographic target statistics
        # -------------------------------------------------------------

        geo_data = pd.DataFrame(
            {
                "geo_cluster": geo_labels,
                "_target": y,
            },
            index=X.index,
        )

        self.geo_statistics_ = (
            geo_data
            .groupby("geo_cluster")["_target"]
            .agg(
                geo_price_mean="mean",
                geo_price_std="std",
            )
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the learned feature engineering to new data.
        """

        check_is_fitted(
            self,
            attributes=[
                "city_statistics_",
                "geo_kmeans_",
                "geo_statistics_",
                "latitude_median_",
                "longitude_median_",
            ],
        )

        transformed = X.copy()

        # -------------------------------------------------------------
        # City-level features
        # -------------------------------------------------------------

        transformed = transformed.join(
            self.city_statistics_,
            on="cityname",
        )

        # Handle cities that were not present during training.
        transformed["city_price_mean"] = (
            transformed["city_price_mean"]
            .fillna(self.global_price_mean_)
        )

        transformed["city_price_std"] = (
            transformed["city_price_std"]
            .fillna(self.global_price_std_)
        )

        transformed["city_listing_count"] = (
            transformed["city_listing_count"]
            .fillna(0)
        )

        # -------------------------------------------------------------
        # Geographic features
        # -------------------------------------------------------------

        transformed["latitude"] = pd.to_numeric(
            transformed["latitude"],
            errors="coerce",
        ).fillna(self.latitude_median_)

        transformed["longitude"] = pd.to_numeric(
            transformed["longitude"],
            errors="coerce",
        ).fillna(self.longitude_median_)

        coordinates = transformed[
            ["latitude", "longitude"]
        ]

        transformed["geo_cluster"] = (
            self.geo_kmeans_.predict(coordinates)
        )

        transformed = transformed.join(
            self.geo_statistics_,
            on="geo_cluster",
        )

        transformed["geo_price_mean"] = (
            transformed["geo_price_mean"]
            .fillna(self.global_price_mean_)
        )

        transformed["geo_price_std"] = (
            transformed["geo_price_std"]
            .fillna(self.global_price_std_)
        )

        # -------------------------------------------------------------
        # Beds-to-bathrooms ratio
        # -------------------------------------------------------------

        bedrooms = (
            pd.to_numeric(
                transformed["bedrooms"],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
        )

        bathrooms = (
            pd.to_numeric(
                transformed["bathrooms"],
                errors="coerce",
            )
            .fillna(0)
            .to_numpy()
        )

        transformed["beds_per_bath_ratio"] = np.divide(
            bedrooms,
            bathrooms,
            out=np.zeros(
                len(transformed),
                dtype=float,
            ),
            where=bathrooms != 0,
        )

        return transformed