
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, f1_score

from prepare_features import load_processed_data, prepare_features
from c45_tree import C45DecisionTree


def make_standard_tree(criterion, categorical_columns):
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

    tree = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )

    return make_pipeline(preprocessor, tree)


def evaluate_custom_tree(X, y, cv):
    accuracies = []
    macro_f1_scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = C45DecisionTree(
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        from sklearn.metrics import accuracy_score
        accuracies.append(accuracy_score(y_test, predictions))
        macro_f1_scores.append(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        )

    return accuracies, macro_f1_scores


def evaluate_experiment(X, y, experiment_name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    models = {
        "Standard Gini": make_standard_tree("gini", categorical_columns),
        "Standard Entropy": make_standard_tree("entropy", categorical_columns),
    }

    print(f"\n--- {experiment_name} ---")

    custom_accuracy, custom_f1 = evaluate_custom_tree(X, y, cv)
    print(
        f"Custom C4.5-style | "
        f"Accuracy: {sum(custom_accuracy) / len(custom_accuracy):.4f} | "
        f"Macro F1: {sum(custom_f1) / len(custom_f1):.4f}"
    )

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": make_scorer(
                    f1_score, average="macro", zero_division=0
                ),
            },
            n_jobs=1,
        )

        print(
            f"{name:20} | "
            f"Accuracy: {scores['test_accuracy'].mean():.4f} "
            f"(± {scores['test_accuracy'].std():.4f}) | "
            f"Macro F1: {scores['test_macro_f1'].mean():.4f}"
        )


def main():
    df = load_processed_data()

    X_with, y = prepare_features(df, include_previous_grades=True)
    X_without, y_without = prepare_features(
        df, include_previous_grades=False
    )

    evaluate_experiment(X_with, y, "With G1 and G2")
    evaluate_experiment(X_without, y_without, "Without G1 and G2")


if __name__ == "__main__":
    main()
