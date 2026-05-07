"""
Weather tool using Open-Meteo API (free, no API key required).

Provides current weather lookup by city name.
"""

import requests
from langchain.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def _get_coordinates(city: str) -> tuple[float, float] | None:
    """Resolve a city name to latitude/longitude."""
    # Sanitize input — agents often pass quoted or padded strings
    city = city.strip().strip('"').strip("'")
    if not city:
        return None
    try:
        resp = requests.get(
            GEOCODING_URL,
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        return results[0]["latitude"], results[0]["longitude"]
    except Exception:
        return None


def _fetch_weather(lat: float, lon: float) -> dict | None:
    """Fetch current weather from Open-Meteo."""
    try:
        resp = requests.get(
            WEATHER_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("current_weather", {})
    except Exception:
        return None


@tool
def get_current_weather(city: str) -> str:
    """
    Get the current weather for a city.
    
    Args:
        city: The name of the city (e.g., "Beijing", "New York", "London").
    
    Returns:
        A short description of the current weather and temperature.
    """
    coords = _get_coordinates(city)
    if coords is None:
        return f"Sorry, I couldn't find the location '{city}'. Please check the city name and try again."

    lat, lon = coords
    weather = _fetch_weather(lat, lon)
    if weather is None:
        return "Sorry, the weather service is currently unavailable. Please try again later."

    temp = weather.get("temperature", "unknown")
    wind = weather.get("windspeed", "unknown")
    code = weather.get("weathercode", 0)
    description = _weather_code_to_description(code)

    return (
        f"Current weather in {city}: {description}, "
        f"temperature {temp}°C, wind speed {wind} km/h."
    )


def _weather_code_to_description(code: int) -> str:
    """Map WMO weather codes to human-readable descriptions."""
    mapping = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return mapping.get(code, "Unknown conditions")


if __name__ == "__main__":
    print(get_current_weather.run("Beijing"))
    print(get_current_weather.run("London"))
