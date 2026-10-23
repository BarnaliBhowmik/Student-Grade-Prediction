
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"

GRADE_BINS = [-1, 9, 13, 16, 20]
GRADE_LABELS = ["Fail", "Satisfactory", "Good", "Excellent"]


def load_data() -> pd.DataFrame:
    """Load the UCI Mathematics student dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH, sep=";")


def assign_grade_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add a project-defined category based on final grade G3."""
    if "G3" not in df.columns:
        raise ValueError("The dataset must contain the G3 column.")

    result = df.copy()

    result["grade_category"] = pd.cut(
        result["G3"],
        bins=GRADE_BINS,
        labels=GRADE_LABELS,
    )

    return result


def main() -> None:
    df = load_data()
    df = assign_grade_category(df)
    output_path = PROJECT_ROOT / "data" / "processed_students.csv"
    df.to_csv(output_path, index=False)
    print(f"\nProcessed dataset saved to: {output_path}")

    print("\n--- DATASET SUMMARY ---")
    print(f"Number of students: {len(df)}")
    print(f"Number of original columns: {df.shape[1] - 1}")

    print("\n--- GRADE CATEGORY COUNTS ---")
    print(
        df["grade_category"]
        .value_counts(sort=False)
        .to_string()
    )

    print("\n--- GRADE CATEGORY PERCENTAGES ---")
    print(
        (df["grade_category"].value_counts(normalize=True, sort=False) * 100)
        .round(2)
        .to_string()
    )

    print("\n--- ORIGINAL G3 SCORES ---")
    print(df["G3"].describe())

    print("\n--- ZERO FINAL GRADES ---")
    print(f"Number of students with G3 = 0: {(df['G3'] == 0).sum()}")

    print("\nPreprocessing check completed successfully.")


if __name__ == "__main__":
    main()
