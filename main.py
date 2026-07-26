from src.evaluation import evaluate_model_cv, print_cv_results
from src.preprocessing import prepare_base_datasets


def main():
    development, _ = prepare_base_datasets()

    X = development.drop(columns=["price"])
    y = development["price"]

    results = evaluate_model_cv(
        X=X,
        y=y,
        model_name="linear_regression",
        n_splits=5,
        n_jobs=1,
    )

    print_cv_results(results)


if __name__ == "__main__":
    main()