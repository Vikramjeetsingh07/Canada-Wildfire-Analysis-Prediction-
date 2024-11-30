# Forecast 🚀 - Project on predicting Fire Forest for the top ten cities with most wildfires in Canada
## Locations for this project with Latitude and Longitude

The following table lists various locations with their respective latitude and longitude:

| Name              | Latitude  | Longitude   |
|-------------------|-----------|-------------|
| Fort McMurray     | 56.7265   | -111.379    |
| Kelowna           | 49.888    | -119.496    |
| Kamloops          | 50.6745   | -120.3273   |
| Prince George     | 53.9171   | -122.7497   |
| Vancouver Island  | 49.6508   | -125.4492   |
| Lytton            | 50.2316   | -121.5824   |
| Penticton         | 49.4991   | -119.5937   |
| Williams Lake     | 52.141    | -122.141    |
| Grande Prairie    | 55.1707   | -118.7884   |
| Edson             | 53.581    | -116.439    |


### Overview
This project explores predication on wildfires in Canada from 2006 to 2023, utilizing weather data and NASA FIRMS satellite scan data.

The whole procedue can be mainly divided to three parts:
- Data Processing
- Analysis
- Machine Learning Training and Testing

### Data Processing
We collected the weather data from [open-meteo](https://open-meteo.com/) through [Python API](https://github.com/Vikramjeetsingh07/Canada-Wildfire-Analysis-Prediction-/blob/main/data-prep/multiple_cities_open_meteo_weather_data_code.py). Then, we joined the weather data with Modis data on date & city on [data_pipeline_top_ten_cities.py](https://github.com/Vikramjeetsingh07/Canada-Wildfire-Analysis-Prediction-/blob/main/data-prep/data_pipeline_top_ten_cities.py).

With `top_ten_wildfire_weather_data_complete.csv`, we continued to add some new features to the existing ones because it could expand the meaning of the dataset. Through add_newFeatures.ipynb, we managed to add some new features (new_fire_intensity, new_daylight_fraction, new_fire_risk,new_relative_humidity, etc.). For example, the new_relative_humidity was computed as:

$$
\text{RH} = 100 \times \frac{\exp\left(\frac{17.625 \cdot T_{\text{min}}}{T_{\text{min}} + 243.04}\right)}{\exp\left(\frac{17.625 \cdot T_{\text{max}}}{T_{\text{max}} + 243.04}\right)}
$$

### Analysis

TBD..

### Machine Learning Training and Testing
We divided this ML to two parts: Classification and Regression.

#### Classification
TBD..

#### Regression
In [regression-model](https://github.com/Vikramjeetsingh07/Canada-Wildfire-Analysis-Prediction-/tree/main/regression-model), we train different linear regression models, tree models, and neural network models on `combined_final.csv` and achieved 15 as RMSE and 0.99 as R2.
