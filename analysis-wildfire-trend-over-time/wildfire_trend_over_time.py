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


"""# Wildfires Trend Over Time"""

wildfires = df.filter(col("wildfire") == 1)
wildfires_with_year = wildfires.withColumn("year", year(col("date")))
wildfire_trend = wildfires_with_year.groupBy("year", "date", "city") \
    .agg(count("*").alias("daily_wildfire_count")) \
    .groupBy("year") \
    .agg(countDistinct("date", "city").alias("unique_wildfire_days")) \
    .orderBy("year")

wildfire_trend_pd = wildfire_trend.toPandas()

# Calculate average temperature per year using 'temperature_2m_mean'
avg_temp_per_year = df.groupBy(year(col("date")).alias("year")) \
    .agg(avg("temperature_2m_mean").alias("avg_temperature")) \
    .orderBy("year")

# Calculate average precipitation per year
avg_precip_per_year = df.groupBy(year(col("date")).alias("year")) \
    .agg(avg("precipitation_sum").alias("avg_precipitation")) \
    .orderBy("year")


# Convert all DataFrames to Pandas for plotting
wildfire_trend_pd = wildfire_trend.toPandas()
avg_temp_pd = avg_temp_per_year.toPandas()
avg_precip_pd = avg_precip_per_year.toPandas()  # Convert to Pandas

fig, ax1 = plt.subplots(figsize=(12, 6))  # Adjust figure size

# Plot wildfire trend (primary y-axis)
ax1.plot(wildfire_trend_pd["year"], wildfire_trend_pd["unique_wildfire_days"], \
         color="blue", label="Unique Wildfire Days")
ax1.set_xlabel("Year", fontsize=12)  # Increase font size
ax1.set_ylabel("Number of Wildfires", color="blue", fontsize=12)
ax1.tick_params(axis="y", labelcolor="blue")

# Average temperature (secondary y-axis)
ax2 = ax1.twinx()
ax2.plot(avg_temp_pd["year"], avg_temp_pd["avg_temperature"], color="red", label="Average Temperature")
ax2.set_ylabel("Average Temperature (°C)", color="red", fontsize=12)  # Add units
ax2.tick_params(axis="y", labelcolor="red")

# Average precipitation (tertiary y-axis)
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis spine
ax3.plot(avg_precip_pd["year"], avg_precip_pd["avg_precipitation"], color="green", label="Average Precipitation")
ax3.set_ylabel("Average Precipitation (mm)", color="green", fontsize=12)  # Add units
ax3.tick_params(axis="y", labelcolor="green")

# Improved legend placement
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=10)

# Title and grid
plt.title("Wildfire Trend, Average Temperature, and Precipitation Over Time", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)  # Customize grid

plt.tight_layout()  # Adjust layout
plt.show()

"""This chart visualizes the relationship between wildfire count, average temperature, and average precipitation over time. Key observations include the correlation between temperature, precipitation, and wildfire activity. The blue line represents wildfire count, the red line indicates average temperature, and the green line corresponds to average precipitation.

In years with higher average temperatures, such as 2016 and 2021, there is a noticeable increase in wildfire count, suggesting a potential link between temperature and wildfire activity. Conversely, years with higher average precipitation generally show lower wildfire counts, indicating precipitation may act as a mitigating factor for wildfires. The interplay between these variables highlights how climatic factors like temperature and precipitation contribute to wildfire dynamics, underscoring the importance of understanding environmental influences for better wildfire management.
"""

# Calculate unique wildfire days per city per year
wildfires = df.filter(col("wildfire") == 1)
wildfires_with_year_city = wildfires.withColumn("year", year(col("date")))
wildfire_counts = wildfires_with_year_city.groupBy("year", "date", "city") \
    .agg(count("*").alias("daily_wildfire_count")) \
    .groupBy("year", "city") \
    .agg(countDistinct("date").alias("unique_wildfire_days")) \
    .orderBy("year", "city")

wildfire_counts_pd = wildfire_counts.toPandas()

# Pivot the DataFrame to create a suitable format for stacked histogram
wildfire_counts_pivot = wildfire_counts_pd.pivot(index='year', columns='city', values='unique_wildfire_days')

# Create the stacked histogram
wildfire_counts_pivot.plot(kind='bar', stacked=True, figsize=(12, 6))

# Add labels and title
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
           fancybox=True, shadow=True, ncol=5)
plt.xlabel('Year')
plt.ylabel('Numbers of Days of Wildfires')
plt.title('Wildfire Trend with City Contributions (Stacked Histogram)')

# Display the plot
plt.show()

"""This stacked histogram visualizes the annual trend of wildfires across various cities from 2006 to 2023, with contributions from specific cities represented by different colors. The y-axis shows the total number of wildfires, while the x-axis represents the years. Each segment in a bar corresponds to the number of wildfires reported in a particular city, allowing for a clear comparison of contributions by location over time.

Key observations include a noticeable spike in wildfire occurrences in 2015 and 2023, indicating potentially extreme conditions during those years. Conversely, years like 2012 and 2020 exhibit relatively lower wildfire counts. Among the cities, locations such as Prince George, Kamloops, and Penticton seem to consistently contribute a significant portion to the total wildfire count, reflecting areas of high fire activity. This trend suggests that wildfire management efforts might need to prioritize these cities due to their recurrent vulnerability.

From 2020 to 2021, there is a significant leap in the total number of wildfires, as illustrated by the marked increase in the bar height. This sharp rise suggests a drastic shift in environmental conditions or other contributing factors, such as prolonged droughts, higher temperatures, or increased human activities during this period. The cities of **Kamloops, Penticton, Kelowna and Lytton** appear to play a prominent role in this increase, with larger segments in 2021 compared to 2020. This jump highlights the need to investigate and address the underlying causes of such spikes to mitigate future wildfire risks.

"""

