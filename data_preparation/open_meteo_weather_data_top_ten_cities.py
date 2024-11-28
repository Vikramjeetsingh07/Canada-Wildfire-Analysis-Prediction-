# -*- coding: utf-8 -*-

"""
Dependencies, that you might need before running:-
!pip install openmeteo-requests
!pip install requests-cache retry-requests numpy pandas
# Getting data using openmeteo API for top ten cities with highest wildfire's weather from 2006-2023 using pandas, requests_cache, retry-requests, and numpy.

# I have to add 60 seconds delay as it has limit per minute and hour at openmeteo. so , there are certain measure to use it for free
"""

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from pathlib import Path
import time

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Locations with latitude and longitude
locations = [
    {"name": "Fort McMurray", "latitude": 56.7265, "longitude": -111.379},
    {"name": "Kelowna", "latitude": 49.888, "longitude": -119.496},
    {"name": "Kamloops", "latitude": 50.6745, "longitude": -120.3273},
    {"name": "Prince George", "latitude": 53.9171, "longitude": -122.7497},
    {"name": "Vancouver Island", "latitude": 49.6508, "longitude": -125.4492},
    {"name": "Lytton", "latitude": 50.2316, "longitude": -121.5824},
    {"name": "Penticton", "latitude": 49.4991, "longitude": -119.5937},
    {"name": "Williams Lake", "latitude": 52.141, "longitude": -122.141},
    {"name": "Grande Prairie", "latitude": 55.1707, "longitude": -118.7884},
    {"name": "Edson", "latitude": 53.581, "longitude": -116.439}
]

# Open-Meteo API URL and parameters
url = "https://archive-api.open-meteo.com/v1/archive"
params_template = {
    "start_date": "2006-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
        "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum",
        "snowfall_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
        "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
    ],
    "timezone": "America/Los_Angeles"
}

# Function to fetch data for a single location
def fetch_weather_data(location):
    params = params_template.copy()
    params["latitude"] = location["latitude"]
    params["longitude"] = location["longitude"]
    try:
        response = openmeteo.weather_api(url, params=params)[0]
        daily = response.Daily()
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left"
            ),
            "city": location["name"],
            "latitude": location["latitude"],
            "longitude": location["longitude"],
        }
        # Map weather variables to columns
        for i, var_name in enumerate(params["daily"]):
            daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()
        return pd.DataFrame(daily_data)
    except Exception as e:
        print(f"Failed to fetch data for {location['name']}: {e}")
        return pd.DataFrame()  # Return empty DataFrame if the request fails

# File to save the consolidated data
output_file = Path("wildfire_prone_areas_weather_data.csv")

# Load existing data if the file exists
if output_file.exists():
    all_data = pd.read_csv(output_file)
else:
    all_data = pd.DataFrame()

# Fetch and append data for all locations
for location in locations:
    if "city" in all_data.columns and location["name"] in all_data["city"].unique():
        print(f"Data for {location['name']} already exists. Skipping...")
        continue
    print(f"Fetching data for {location['name']}...")
    location_data = fetch_weather_data(location)
    all_data = pd.concat([all_data, location_data], ignore_index=True)
    all_data.to_csv(output_file, index=False)  # Save after every city
    print(f"Data for {location['name']} saved. Waiting 60 seconds...")
    time.sleep(30)  # Respect API rate limits

print(f"Consolidated data saved to {output_file}")
