from src.feature_engineering import RentalFeatureEngineer
from src.preprocessing import prepare_base_datasets


def main():
    development, evaluation = prepare_base_datasets()

    y = development["price"]

    X_development = development.drop(
        columns=["price"]
    )

    engineer = RentalFeatureEngineer(
        n_geo_clusters=200,
        random_state=42,
    )

    X_development_engineered = engineer.fit_transform(
        X_development,
        y,
    )

    X_evaluation_engineered = engineer.transform(
        evaluation
    )

    engineered_columns = [
        "city_price_mean",
        "city_price_std",
        "city_listing_count",
        "geo_cluster",
        "geo_price_mean",
        "geo_price_std",
        "beds_per_bath_ratio",
    ]

    print("Development:")
    print(X_development_engineered.shape)

    print("\nEvaluation:")
    print(X_evaluation_engineered.shape)

    print("\nEngineered features:")
    print(
        X_development_engineered[
            engineered_columns
        ].head()
    )


if __name__ == "__main__":
    main()