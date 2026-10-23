
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from prepare_features import load_processed_data, prepare_features
from c45_tree import C45DecisionTree


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RANDOM_STATE = 42
TARGET_CLASSES = ["Fail", "Satisfactory", "Good", "Excellent"]


def evaluate_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    experiment_name: str,
) -> dict:
    """Train and evaluate one feature configuration."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = C45DecisionTree(
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\n{'=' * 60}")
    print(experiment_name)
    print("=" * 60)
    print(f"Training students: {len(X_train)}")
    print(f"Testing students: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=TARGET_CLASSES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=TARGET_CLASSES,
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(
        matrix,
        index=TARGET_CLASSES,
        columns=TARGET_CLASSES,
    ))

    return {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "model": model,
    }


def main() -> None:
    df = load_processed_data()

    # Experiment A: previous grades are available.
    X_with_grades, y = prepare_features(
        df,
        include_previous_grades=True,
    )

    # Experiment B: predict without G1 and G2.
    X_without_grades, y_without_grades = prepare_features(
        df,
        include_previous_grades=False,
    )

    result_a = evaluate_experiment(
        X_with_grades,
        y,
        "Experiment A — With G1 and G2",
    )

    result_b = evaluate_experiment(
        X_without_grades,
        y_without_grades,
        "Experiment B — Without G1 and G2",
    )

    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    print(f"With G1/G2:    {result_a['accuracy']:.4f}")
    print(f"Without G1/G2: {result_b['accuracy']:.4f}")
    print(
        "Accuracy difference: "
        f"{(result_a['accuracy'] - result_b['accuracy']) * 100:.2f} percentage points"
    )


if __name__ == "__main__":
    main()
