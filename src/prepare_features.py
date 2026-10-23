
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed_students.csv"

TARGET_COLUMN = "grade_category"
GRADE_COLUMNS = ["G1", "G2", "G3"]


def load_processed_data() -> pd.DataFrame:
    """Load the processed student dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


def prepare_features(
    df: pd.DataFrame,
    include_previous_grades: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate input features from the target category."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    columns_to_drop = [TARGET_COLUMN, "G3"]

    if not include_previous_grades:
        columns_to_drop.extend(["G1", "G2"])

    X = df.drop(columns=columns_to_drop)
    y = df[TARGET_COLUMN]

    return X, y


def main() -> None:
    df = load_processed_data()

    X_with_grades, y = prepare_features(
        df,
        include_previous_grades=True,
    )

    X_without_grades, _ = prepare_features(
        df,
        include_previous_grades=False,
    )

    print("--- DATA PREPARATION SUMMARY ---")
    print(f"Total students: {len(df)}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Target classes: {sorted(y.unique().tolist())}")

    print("\nExperiment A — Includes G1 and G2")
    print(f"Number of features: {X_with_grades.shape[1]}")
    print(f"G1 included: {'G1' in X_with_grades.columns}")
    print(f"G2 included: {'G2' in X_with_grades.columns}")
    print(f"G3 included: {'G3' in X_with_grades.columns}")

    print("\nExperiment B — Excludes G1 and G2")
    print(f"Number of features: {X_without_grades.shape[1]}")
    print(f"G1 included: {'G1' in X_without_grades.columns}")
    print(f"G2 included: {'G2' in X_without_grades.columns}")
    print(f"G3 included: {'G3' in X_without_grades.columns}")

    print("\nFeature preparation completed successfully.")


if __name__ == "__main__":
    main()
