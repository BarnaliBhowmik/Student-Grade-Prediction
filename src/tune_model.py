
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from prepare_features import load_processed_data, prepare_features


def main():
    df = load_processed_data()

    # Experiment A: include previous grades G1 and G2.
    X, y = prepare_features(df, include_previous_grades=True)

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("tree", DecisionTreeClassifier(random_state=42)),
    ])

    parameter_grid = {
        "tree__criterion": ["gini"],
        "tree__max_depth": [3, 5, 8, 12, None],
        "tree__min_samples_split": [2, 4, 8, 12],
        "tree__min_samples_leaf": [1, 2, 4, 6],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    print("Running hyperparameter tuning...")
    print("Experiment: With G1 and G2")
    print("Selection metric: Macro F1")

    search.fit(X, y)

    print("\nBEST PARAMETERS")
    print(search.best_params_)

    print(
        f"\nBest mean cross-validation macro F1: "
        f"{search.best_score_:.4f}"
    )

    # Evaluate the selected configuration using both metrics.
    best_model = search.best_estimator_

    scores = cross_validate(
        best_model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
        },
        n_jobs=-1,
    )

    print("\nBEST MODEL: CROSS-VALIDATION SUMMARY")
    print(f"Mean accuracy: {scores['test_accuracy'].mean():.4f}")
    print(
        "Accuracy standard deviation: "
        f"{scores['test_accuracy'].std():.4f}"
    )
    print(f"Mean macro F1: {scores['test_macro_f1'].mean():.4f}")
    print(
        "Macro F1 standard deviation: "
        f"{scores['test_macro_f1'].std():.4f}"
    )

    print("\nTOP 10 CONFIGURATIONS")

    results = (
        pd.DataFrame(search.cv_results_)
        .sort_values("rank_test_score")
        [["rank_test_score", "mean_test_score", "std_test_score", "params"]]
        .head(10)
    )

    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
