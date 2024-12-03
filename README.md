# Forecast 🚀 - Project on predicting Forest Fires for the top ten cities with most wildfires in Canada

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
This project explores prediction on wildfires in Canada from 2006 to 2023, utilizing weather data and NASA FIRMS satellite scan data. The whole procedure can be mainly divided into six parts: Data collection, Data ETL, Feature Addition, Feature Extraction, Analysis of dataset, and Machine Learning for predicting Fire happens or not (classification) and brightness and confidence value in MODIS (regression).

### Data Processing
We collected the weather data from [open-meteo](https://open-meteo.com/) through [Python API](https://github.com/Vikramjeetsingh07/Canada-Wildfire-Analysis-Prediction-/blob/main/data-prep/multiple_cities_open_meteo_weather_data_code.py). Then, we joined the weather data with MODIS data on date & city on [data_pipeline_top_ten_cities.py](https://github.com/Vikramjeetsingh07/Canada-Wildfire-Analysis-Prediction-/blob/main/data-prep/data_pipeline_top_ten_cities.py).

With `top_ten_wildfire_weather_data_complete.csv`, we continued to add some new features to the existing ones to expand the meaning of the dataset. Through add_newFeatures.ipynb, we managed to add features like new_fire_intensity, new_daylight_fraction, new_fire_risk, new_relative_humidity, etc. For example, the new_relative_humidity was computed as:

$$
\text{RH} = 100 \times \frac{\exp\left(\frac{17.625 \cdot T_{\text{min}}}{T_{\text{min}} + 243.04}\right)}{\exp\left(\frac{17.625 \cdot T_{\text{max}}}{T_{\text{max}} + 243.04}\right)}
$$These enhancements have helped in refining our analysis and improving model accuracy.

### Analysis
Our comprehensive analysis focused on correlating various environmental factors such as wind, temperature, and precipitation with wildfire occurrences. The feature correlation insights and the identification of significant predictors such as brightness and fire intensity based on MODIS data have informed the model development phase. Conclusions from the analysis demonstrated the critical impact of temperature increases and precipitation decreases on fire risk, emphasizing the need for predictive modeling to incorporate these variables effectively.

### Machine Learning Training and Testing
The machine learning component was divided into two parts: Classification and Regression.

#### Classification
We employed RandomForest and Gradient Boosting classifiers to predict whether a wildfire will occur based on environmental conditions reported on specific days. Our initial models achieved an accuracy of 87%, which improved to nearly 95% after hyperparameter tuning. This tuning involved adjusting parameters such as the number of decision trees in the RandomForest model and the learning rate in the Gradient Boosting model. The high accuracy of these models underscores their effectiveness in distinguishing days with high wildfire risk from those with low risk.

#### Regression
We have applied various regression models to predict the brightness and confidence levels reported by MODIS. Our neural network regressor outperformed other models, achieving an RMSE of 15.38 and R² of 0.991 for brightness prediction. The regression models have been critical in accurately predicting fire characteristics from satellite data.

This enhanced README preserves the original structure while incorporating detailed insights from the extended analysis and results sections to provide a comprehensive view of the project's scope and its impact.
