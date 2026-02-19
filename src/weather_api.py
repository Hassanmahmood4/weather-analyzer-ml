import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_weather_by_city(city: str):
    if not API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY not found in .env")

    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    r = requests.get(BASE_URL, params=params, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"OpenWeather error {r.status_code}: {r.text}")

    data = r.json()

    return {
        "temp": float(data["main"]["temp"]),
        "humidity": float(data["main"]["humidity"]) / 100.0,
        "pressure": float(data["main"]["pressure"]),
        "wind_speed": float(data["wind"]["speed"]) * 3.6,  # m/s -> km/h
        "summary": data["weather"][0]["main"].lower(),
        "visibility": float(data.get("visibility", 10000)) / 1000.0
    }