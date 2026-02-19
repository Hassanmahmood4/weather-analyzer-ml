import numpy as np
from src.weather_api import fetch_weather_by_city

# Small curated list of cities (you can expand this)
CITIES = ["London", "New York", "Tokyo", "Paris", "Berlin", "Karachi", "Delhi", "Sydney"]

def closest_city_from_conditions(temp, humidity, wind_speed, pressure, visibility):
    best_city = None
    best_dist = float("inf")
    best_live = None

    for city in CITIES:
        try:
            live = fetch_weather_by_city(city)
            vec_city = np.array([
                live["temp"],
                live["humidity"],
                live["wind_speed"],
                live["pressure"],
                live["visibility"],
            ])
            vec_query = np.array([temp, humidity, wind_speed, pressure, visibility])

            dist = np.linalg.norm(vec_city - vec_query)

            if dist < best_dist:
                best_dist = dist
                best_city = city
                best_live = live
        except Exception:
            continue

    return best_city, best_live