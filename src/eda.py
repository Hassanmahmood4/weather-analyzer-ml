import pandas as pd
from pathlib import Path

# Resolve project root
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "raw" / "weatherHistory.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Columns:")
print(df.columns.tolist())

print("\n🔹 Info:")
print(df.info())

print("\n🔹 Missing values:")
print(df.isna().sum())

print("\n🔹 Basic stats:")
print(df.describe(include="all"))