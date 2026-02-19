import pandas as pd
from pathlib import Path
import joblib

# Resolve project root
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "weather_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_temperature_model.pkl"

# Load model and data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(PROCESSED_PATH)

TARGET = "Temperature (C)"
FEATURES = [c for c in df.columns if c not in [TARGET, "Formatted Date", "Apparent Temperature (C)"]]

def predict_from_row(row: pd.Series) -> float:
    X = row[FEATURES].values.reshape(1, -1)
    return float(model.predict(X)[0])

if __name__ == "__main__":
    sample = df.sample(1).iloc[0]
    true_temp = sample[TARGET]
    pred_temp = predict_from_row(sample)

    print(f"🌡️ True Temperature: {true_temp:.2f} °C")
    print(f"🤖 Predicted Temperature: {pred_temp:.2f} °C")