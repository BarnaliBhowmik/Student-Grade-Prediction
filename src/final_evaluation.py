
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from prepare_features import load_processed_data, prepare_features


def make_model(X, max_depth, min_samples_leaf):
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

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=2,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    return make_pipeline(preprocessor, tree)


def main():
    df = load_processed_data()
    X, y = prepare_features(df, include_previous_grades=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Baseline Gini": make_model(
            X, max_depth=8, min_samples_leaf=2
        ),
        "Tuned Gini": make_model(
            X, max_depth=5, min_samples_leaf=6
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"\n--- {name} ---")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print(
            "Macro F1: "
            f"{f1_score(y_test, predictions, average='macro'):.4f}"
        )
        print("\nClassification report:")
        print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
