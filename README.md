# Predicting and Historical analysis of Forest Fires for the top ten cities with the most wildfires in Canada

## General Project Guide:

## In folder data:-
#### complete_modis_data.csv : contains modis data for entire canada.
#### top_ten_modis.csv : it contains modis dataset for top 10 target cities.
#### top_ten_wildfire_weather_data_complete.csv : it contains all the weather data for top ten cities 
#### combined_final.csv : it is the final dataset used for all the analysis and ML Models (combined from top_ten_modis.csv and top_ten_wildfire_weather_data_complete.csv)


## Reports/ visualizations:
##### Data Analysis Report: it contains every important analysis with description and summary.
#### Report.pdf (Required ): it contains report of the project + Data Analysis Report (which is added as we felt it would provide valuable insights)
#### ML results.pdf : contains all the results from all the algorithms performed
#### video presentation slides.pptx : contains slides used during project video for the flow of presentation


## Target Locations relevant for this project with their respective Latitude and Longitude

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

# Wildfire Analysis and Prediction

This project explores wildfire patterns and predictive modeling in Canada from 2006 to 2023. By combining data from NASA's FIRMS (Fire Information for Resource Management System) and Open-Meteo, the project aims to analyze wildfire occurrences and develop machine learning models for prediction based on environmental and climatic factors.

---

## Problem Statement

Wildfires are a growing concern in Canada, driven by climate change. This project addresses challenges like:
- Analyzing large, multivariate datasets from NASA FIRMS and Open-Meteo.
- Preprocessing data to align spatial and temporal resolutions.
- Developing predictive models to understand wildfire risks.

---

## Methodology

### Data Sources
- **NASA FIRMS**: Satellite data on wildfire brightness, fire radiative power (FRP), and confidence.
- **Open-Meteo**: Weather data for Canadian cities prone to wildfires, including temperature, precipitation, and wind metrics.

### Data Preparation
- Preprocessing involved aligning datasets by location and date, imputing missing values, and removing outliers.
- Feature engineering created derived metrics like dryness index, cumulative precipitation, and fire intensity.

### Data Analysis
- Analyzed feature correlations, trends over time, and geographic variability.
- Conducted city-specific comparisons to identify high-risk areas.

---

## Machine Learning Models

### Classification
- **Algorithms**: Random Forest, Gradient Boosted Trees (GBT), Neural Networks, SVM, Logistic classification.
- **Results**: Accuracy up to **99.9%** for predicting wildfire occurrences.

### Regression
- **Algorithms**: Neural Networks, XGBoost, LightGBM, Random Forest, Linear Regression.
- **Results**: Neural Networks achieved **R² = 0.991** and RMSE = **15.38** for predicting wildfire brightness.

---


## Technologies

- **Distributed Computing**: Apache Spark for scalable data processing and machine learning.
- **Visualization**: Matplotlib and Seaborn for trend analysis and feature importance.
- **Machine Learning Frameworks**: PySpark ML, XGBoost, LightGBM, and TensorFlow.

---
###  Scalability
- Implemented distributed computing using PySpark for efficient ETL and model training on large datasets.

---
## Results and Visualizations

### Key Findings
- **Wildfire Predictors**: High brightness, dryness index, and elevated temperatures are strong indicators of wildfire activity.
- **Geographic Trends**: Kamloops, Penticton, and Kelowna consistently show high wildfire intensity and frequency.
- **Temporal Trends**: Significant wildfire activity spikes in 2016 and 2021 due to extreme heat and dryness.

### Visualizations
1. **Heatmaps**: Show correlations between wind, soil moisture, and fire intensity.
2. **City Trends**: Highlight yearly patterns in dryness, temperature, and fire risk.
3. **Wildfire Counts**: Geographic variability of wildfire occurrences from 2006–2023.
4. **Prediction Results**: Charts comparing model accuracy and performance metrics.

---

## Repository Contents

- **Data Pipeline Scripts**: ETL scripts for data preprocessing and feature engineering.
- **Machine Learning Models**: Code for training, validation, and prediction.
- **Visualizations**: Graphs and heatmaps for analysis and insights.
- **Reports**: Detailed analysis and model results.

---

## Future Work

- Incorporating additional datasets like soil moisture and vegetation indices.
- Extending the analysis to other regions or smaller geographic scales.
- Developing a user-friendly interface for real-time wildfire risk prediction.

---

## Contributors

- **Vikramjeet Singh**
- **Yiming**
- **Harman**
- **Wenhao**

---
