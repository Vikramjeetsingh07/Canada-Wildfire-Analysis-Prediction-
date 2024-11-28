from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, exp, radians, cos, sin, mean, avg, max as spark_max, min as spark_min,
    sum as spark_sum, lag
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import year
from pyspark.sql.functions import count, desc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from pyspark.sql.functions import spark_partition_id

spark = SparkSession.builder.appName("wildfire project").getOrCreate()

data_dir = "/data/combined_final"

df = spark.read.csv(data_dir, header=True, inferSchema=True).cache()

# df.show(10)

df = df.withColumn("wildfire", when(col("instrument") == "MODIS", 1).otherwise(0))

from pyspark.sql.functions import countDistinct

# Filter for wildfire occurrences (wildfire == 1)
wildfires = df.filter(col("wildfire") == 1)

# Group by city and count distinct dates
wildfire_days_per_city = wildfires.groupBy("city") \
    .agg(countDistinct("date").alias("wildfire_days")) \
    .orderBy(desc("wildfire_days"))  # Order by descending wildfire days

# Convert to Pandas DataFrame for further analysis or plotting (if needed)
wildfire_days_per_city_pd = wildfire_days_per_city.toPandas()

# Show the result (or proceed with plotting)
wildfire_days_per_city.show()

# Create the bar plot
plt.figure(figsize=(12, 6))  # Adjust figure size as needed
plt.bar(wildfire_days_per_city_pd['city'], wildfire_days_per_city_pd['wildfire_days'],
        color='orange', alpha=0.7)  # Customize color and alpha

plt.xlabel('City')
plt.ylabel('Number of Wildfire Days')
plt.title('Wildfire Days per City, 2006 - 2023')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability

plt.tight_layout()  # Adjust layout to prevent overlapping labels
plt.show()


from pyspark.sql.functions import year, avg, when, sum as spark_sum

# Filter for the specified cities and years
filtered_df = df.filter(
    (col("city").isin(["Penticton", "Lytton", "Kelowna", "Kamloops"])) &
    (year(col("date")).isin([2020, 2021]))
)

# Group by city, year, and calculate average values for features
city_year_features = filtered_df.groupBy("city", year(col("date")).alias("year")) \
    .agg(
        avg("temperature_2m_mean").alias("avg_temperature"),
        avg("new_dryness_index").alias("avg_dryness_index"),
        avg("precipitation_sum").alias("avg_precipitation"),
        avg("sunshine_duration").alias("avg_sunshine"),  # Corrected column name
        avg("daylight_duration").alias("avg_daylight"),
        avg("et0_fao_evapotranspiration").alias("avg_et0_fao_evapotranspiration"),
        avg("new_relative_humidity").alias("avg_relative_humidity_2m"),
        # Add other features you want to compare here
    ) \
    .orderBy("city", "year")

# Pivot the table for easier comparison
features_pivot = city_year_features.groupBy("city") \
    .pivot("year", [2020, 2021]) \
    .agg(
        # Select the average values for each feature
        # Use the appropriate aggregation function if needed (e.g., spark_sum, spark_max)
        avg("avg_temperature").alias("avg_temperature"),
        avg("avg_dryness_index").alias("avg_dryness_index"),
        avg("avg_precipitation").alias("avg_precipitation"),
        avg("avg_sunshine").alias("avg_sunshine"),  # Corrected column name
        avg("avg_daylight").alias("avg_daylight"),
        avg("avg_et0_fao_evapotranspiration").alias("avg_et0_fao_evapotranspiration"),
        avg("avg_relative_humidity_2m").alias("avg_relative_humidity_2m"),
        # Add other features and aggregation functions here
    ) \
    .orderBy("city")

# Show the comparison table
features_pivot.show()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# List of cities
cities = ["Penticton", "Lytton", "Kelowna", "Kamloops"]

# List of features to compare (without year prefix)
features = ['avg_temperature', 'avg_dryness_index', 'avg_precipitation', 'avg_sunshine',
            'avg_daylight', 'avg_et0_fao_evapotranspiration', 'avg_relative_humidity_2m']

features_pivot_pd = features_pivot.toPandas()

# Rename columns for better clarity and plotting
features_pivot_pd.columns = [
    col.replace("2020_", "2020_").replace("2021_", "2021_") for col in features_pivot_pd.columns
]

# Create a figure and axes for each city
for city in cities:
    city_data = features_pivot_pd[features_pivot_pd['city'] == city]

    # Extract data for 2020 and 2021
    values_2020 = city_data[[f'2020_{feature}' for feature in features]].values.flatten()
    values_2021 = city_data[[f'2021_{feature}' for feature in features]].values.flatten()

    # Apply log transformation to sunshine, daylight, and relative humidity
    sunshine_index = features.index('avg_sunshine')
    daylight_index = features.index('avg_daylight')
    humidity_index = features.index('avg_relative_humidity_2m')  # Get index of humidity


    # Regularizing features
    values_2020[sunshine_index] = np.log1p(values_2020[sunshine_index])
    values_2020[daylight_index] = np.log1p(values_2020[daylight_index])
    values_2020[humidity_index] = np.log1p(values_2020[humidity_index])

    values_2021[sunshine_index] = np.log1p(values_2021[sunshine_index])
    values_2021[daylight_index] = np.log1p(values_2021[daylight_index])
    values_2021[humidity_index] = np.log1p(values_2021[humidity_index])

    # print(values_2020)
    # print(values_2021)

    # Set up the bar positions
    x = np.arange(len(features))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, values_2020, width, label='2020')
    rects2 = ax.bar(x + width/2, values_2021, width, label='2021')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Values')
    ax.set_title(f'Feature Comparison for {city}')
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha='right')  # Rotate x-axis labels
    ax.legend()

    fig.tight_layout()
    plt.show()