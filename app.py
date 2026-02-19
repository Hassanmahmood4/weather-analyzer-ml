import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

from src.weather_api import fetch_weather_by_city
from src.city_recommender import closest_city_from_conditions

st.set_page_config(page_title="Weather Analyzer ML", layout="centered")

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rf_temperature_model.pkl"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "weather_processed.csv"

# Load model & data
model = joblib.load(MODEL_PATH)
df = pd.read_csv(PROCESSED_PATH)

TARGET = "Temperature (C)"
FEATURES = [c for c in df.columns if c not in [TARGET, "Formatted Date", "Apparent Temperature (C)"]]

st.title("🌦️ Weather Analyzer (Two Modes)")
st.caption("ML-powered temperature prediction + city similarity search")

mode = st.radio("Choose mode", ["City → Conditions", "Conditions → Closest City"])

# -----------------------
# Mode 1: City → Conditions
# -----------------------
if mode == "City → Conditions":
    st.subheader("Enter a city")
    city = st.text_input("City name", placeholder="e.g., London, Tokyo, Karachi")

    if st.button("Analyze City"):
        if not city:
            st.warning("Please enter a city name.")
        else:
            try:
                live = fetch_weather_by_city(city)

                # Build ML input with dataset means
                input_data = {col: df[col].mean() for col in FEATURES}
                input_data["Humidity"] = live["humidity"]
                input_data["Wind Speed (km/h)"] = live["wind_speed"]
                input_data["Pressure (millibars)"] = live["pressure"]
                input_data["Visibility (km)"] = live["visibility"]

                X = pd.DataFrame([input_data])[FEATURES]
                pred_temp = model.predict(X)[0]

                st.success(f"🌡️ Predicted Temperature (ML): **{pred_temp:.2f} °C**")

                st.markdown("### 🌤️ Current Conditions")
                st.write(f"☁️ Summary: **{live['summary'].title()}**")
                st.write(f"🌧️ Precipitation: **{'Rain' if 'rain' in live['summary'] else 'None'}**")
                st.write(f"💨 Wind: **{'Windy' if live['wind_speed'] > 15 else 'Breezy'}**")

            except Exception as e:
                st.error("❌ Could not fetch live weather for this city.")
                st.info("Check your OpenWeather API key in .env or your internet connection.")

# -----------------------
# Mode 2: Conditions → Closest City
# -----------------------
else:
    st.subheader("Enter weather conditions")

    temp = st.slider("Temperature (°C)", -30.0, 40.0, 10.0)
    humidity = st.slider("Humidity (0–1)", 0.0, 1.0, 0.5)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 60.0, 10.0)
    pressure = st.slider("Pressure (millibars)", 950.0, 1050.0, 1013.0)
    visibility = st.slider("Visibility (km)", 0.0, 20.0, 10.0)

    if st.button("Find Closest City"):
        try:
            city, live = closest_city_from_conditions(
                temp, humidity, wind_speed, pressure, visibility
            )

            if city:
                st.success(f"🏙️ Closest matching city right now: **{city}**")
                st.write("Based on similarity with current live weather data.")
                st.json(live)
            else:
                st.warning("Could not determine a closest city. Try again.")

        except Exception:
            st.error("❌ Could not compute closest city (API issue).")
            st.info("Make sure OpenWeather API is working.")