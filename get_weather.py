import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import date
import holidays

def weather(start_date, end_date):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://archive-api.open-meteo.com/v1/archive"
	params = {
		"latitude": 38.04,
		"longitude": 84.5,
		"start_date": start_date,
		"end_date": end_date,
		"daily": ["temperature_2m_mean", "apparent_temperature_mean", "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum", "wind_speed_10m_max"],
		"temperature_unit": "fahrenheit",
		"wind_speed_unit": "mph",
		"precipitation_unit": "inch",
		"timezone": "America/New_York"
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
	daily_apparent_temperature_mean = daily.Variables(1).ValuesAsNumpy()
	daily_daylight_duration = daily.Variables(2).ValuesAsNumpy()
	daily_sunshine_duration = daily.Variables(3).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(4).ValuesAsNumpy()
	daily_rain_sum = daily.Variables(5).ValuesAsNumpy()
	daily_snowfall_sum = daily.Variables(6).ValuesAsNumpy()
	daily_wind_speed_10m_max = daily.Variables(7).ValuesAsNumpy()

	daily_data = {"date": pd.date_range(
		start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
		end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = daily.Interval()),
		inclusive = "left"
	)}
	daily_data["temperature"] = daily_temperature_2m_mean
	daily_data["apparent_temperature"] = daily_apparent_temperature_mean
	daily_data["daylight"] = daily_daylight_duration
	daily_data["sunshine"] = daily_sunshine_duration
	daily_data["precipitation"] = daily_precipitation_sum
	daily_data["rain"] = daily_rain_sum
	daily_data["snow"] = daily_snowfall_sum
	daily_data["wind_speed"] = daily_wind_speed_10m_max

	daily_dataframe = pd.DataFrame(data = daily_data)
	daily_dataframe = daily_dataframe.dropna()
	holidays_df = pd.DataFrame(sorted(holidays.US(years=[2022, 2023, 2024]).items()),
							   columns=['date', 'holiday'])
	holidays_df['holiday'] = 'Holiday'
	holidays_df['date'] = pd.to_datetime(holidays_df['date'], utc=True)
	daily_dataframe['date'] = daily_dataframe['date'].dt.normalize()
	df = pd.merge(daily_dataframe, holidays_df, on='date', how='left')
	df['holiday'] = df['holiday'].fillna('Normal')
	df['day_of_week'] = df['date'].dt.day_name()
	df['month'] = df['date'].dt.month_name()
	return df

def weather_forecast():
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
	retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
	openmeteo = openmeteo_requests.Client(session=retry_session)

	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": 38.04,
		"longitude": -84.5, #For some reason this one defaults to E, so have to put negative sign for W
		"daily": ["temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "apparent_temperature_min",
				  "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum", "snowfall_sum",
				  "wind_speed_10m_max"],
		"temperature_unit": "fahrenheit",
		"wind_speed_unit": "mph",
		"precipitation_unit": "inch",
		"timezone": "America/New_York",
		"past_days": 5
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]

	# Process daily data. The order of variables needs to be the same as requested.
	daily = response.Daily()
	daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
	daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
	daily_apparent_temperature_max = daily.Variables(2).ValuesAsNumpy()
	daily_apparent_temperature_min = daily.Variables(3).ValuesAsNumpy()
	daily_daylight_duration = daily.Variables(4).ValuesAsNumpy()
	daily_sunshine_duration = daily.Variables(5).ValuesAsNumpy()
	daily_precipitation_sum = daily.Variables(6).ValuesAsNumpy()
	daily_rain_sum = daily.Variables(7).ValuesAsNumpy()
	daily_snowfall_sum = daily.Variables(8).ValuesAsNumpy()
	daily_wind_speed_10m_max = daily.Variables(9).ValuesAsNumpy()

	daily_data = {"date": pd.date_range(
		start=pd.to_datetime(daily.Time(), unit="s", utc=True),
		end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
		freq=pd.Timedelta(seconds=daily.Interval()),
		inclusive="left"
	)}
	daily_data["temperature_max"] = daily_temperature_2m_max
	daily_data["temperature_min"] = daily_temperature_2m_min
	daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
	daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
	daily_data["daylight"] = daily_daylight_duration
	daily_data["sunshine"] = daily_sunshine_duration
	daily_data["precipitation"] = daily_precipitation_sum
	daily_data["rain"] = daily_rain_sum
	daily_data["snow"] = daily_snowfall_sum
	daily_data["wind_speed"] = daily_wind_speed_10m_max

	daily_data['temperature'] = (daily_data["temperature_max"] + daily_data["temperature_min"]) / 2
	daily_data['apparent_temperature'] = (daily_data["apparent_temperature_max"] + daily_data["apparent_temperature_min"]) / 2

	daily_dataframe = pd.DataFrame(data=daily_data)
	holidays_df = pd.DataFrame(sorted(holidays.US(years=[2022, 2023, 2024]).items()),
							   columns=['date', 'holiday'])
	holidays_df['date'] = pd.to_datetime(holidays_df['date'], utc=True)
	df = pd.merge(daily_dataframe, holidays_df, on='date', how='left')
	df['holiday'] = df['holiday'].fillna('Normal')
	df['day_of_week'] = df['date'].dt.day_name()
	df['month'] = df['date'].dt.month_name()
	return df
