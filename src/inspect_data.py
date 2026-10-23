
from pathlib import Path
import pandas as pd

# Locate the project and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "student-mat.csv"

# Check that the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Download and place student-mat.csv inside the data folder."
    )

# Load the semicolon-separated CSV
df = pd.read_csv(DATA_PATH, sep=";")

print("\n--- DATASET SHAPE ---")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum().sum())

print("\n--- FINAL GRADE DISTRIBUTION ---")
print(df["G3"].value_counts().sort_index())

print("\n--- DATA TYPES ---")
print(df.dtypes)