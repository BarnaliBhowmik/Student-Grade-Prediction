
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

from prepare_features import load_processed_data, prepare_features
from c45_tree import C45DecisionTree


def make_standard_tree(criterion, categorical_columns):
    """Encode categorical features before training a sklearn tree."""
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


def compare_experiment(X, y, experiment_name):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    categorical_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    models = {
        "Custom C4.5-style": C45DecisionTree(
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
        ),
        "Standard tree (Gini)": make_standard_tree(
            "gini", categorical_columns
        ),
        "Standard tree (Entropy)": make_standard_tree(
            "entropy", categorical_columns
        ),
    }

    print(f"\n--- {experiment_name} ---")
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy,
                "Macro F1": macro_f1,
            }
        )

        print(
            f"{name:25} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )

    return results


def main():
    df = load_processed_data()

    X_with_grades, y = prepare_features(
        df, include_previous_grades=True
    )
    X_without_grades, y_without_grades = prepare_features(
        df, include_previous_grades=False
    )

    compare_experiment(
        X_with_grades, y, "With G1 and G2"
    )
    compare_experiment(
        X_without_grades, y_without_grades, "Without G1 and G2"
    )


if __name__ == "__main__":
    main()
