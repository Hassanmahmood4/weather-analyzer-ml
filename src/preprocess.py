import pandas as pd
from pathlib import Path

# Resolve project root
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "weatherHistory.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "weather_processed.csv"

# Load raw data
df = pd.read_csv(RAW_PATH)

# Parse datetime (use utc to avoid warnings)
df["Formatted Date"] = pd.to_datetime(df["Formatted Date"], errors="coerce", utc=True)

# Time-based features
df["hour"] = df["Formatted Date"].dt.hour
df["day"] = df["Formatted Date"].dt.day
df["month"] = df["Formatted Date"].dt.month
df["dayofweek"] = df["Formatted Date"].dt.dayofweek

# Handle missing precip type
df["Precip Type"] = df["Precip Type"].fillna("None")

# Drop very long text column (not useful for baseline ML)
df = df.drop(columns=["Daily Summary"])

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["Summary", "Precip Type"], drop_first=True)

# Create processed dir if needed
PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

# Save processed dataset
df.to_csv(PROCESSED_PATH, index=False)

print("✅ Preprocessing complete!")
print(f"Saved to: {PROCESSED_PATH}")
print("Final shape:", df.shape)