
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

from prepare_features import load_processed_data, prepare_features


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

    model = make_pipeline(
        preprocessor,
        DecisionTreeClassifier(
            criterion="gini",
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["Fail", "Satisfactory", "Good", "Excellent"],
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX")
    print("Order: Fail, Satisfactory, Good, Excellent")
    print(confusion_matrix(
        y_test,
        predictions,
        labels=["Fail", "Satisfactory", "Good", "Excellent"],
    ))


if __name__ == "__main__":
    main()
