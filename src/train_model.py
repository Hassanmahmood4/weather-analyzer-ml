import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import numpy as np

# Resolve project root
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "weather_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_temperature_model.pkl"

# Load processed data
df = pd.read_csv(PROCESSED_PATH)

TARGET = "Temperature (C)"
FEATURES = [c for c in df.columns if c not in [TARGET, "Formatted Date", "Apparent Temperature (C)"]]

X = df[FEATURES]
y = df[TARGET]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))

print("✅ Model trained!")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# Save model
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to: {MODEL_PATH}")