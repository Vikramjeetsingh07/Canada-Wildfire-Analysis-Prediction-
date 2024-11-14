from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, to_date, abs, when

# Initialize Spark session
spark = SparkSession.builder.appName("WildfirePredictionPipeline").getOrCreate()

# Load datasets
modis_df = spark.read.csv("/content/fire_archive_M-C61_537264.csv", header=True, inferSchema=True)
wildfire_df = spark.read.csv("/content/fp-historical-wildfire-data-2006-2023.csv", header=True, inferSchema=True)
weather_df = spark.read.csv("/content/weather_data.csv", header=True, inferSchema=True)

# Data Cleaning and Type Casting
modis_df = modis_df.dropna(subset=["acq_date", "latitude", "longitude"])
wildfire_df = wildfire_df.dropna(subset=["fire_start_date", "fire_location_latitude", "fire_location_longitude"])
weather_df = weather_df.dropna(subset=["date", "latitude", "longitude"])

# Convert date columns to DateType
modis_df = modis_df.withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))
wildfire_df = wildfire_df.withColumn("fire_start_date", to_date(col("fire_start_date"), "yyyy-MM-dd"))
weather_df = weather_df.withColumn("weather_date", to_date(col("date"), "yyyy-MM-dd"))

# Round coordinates to 3 decimal places for spatial join accuracy
modis_df = modis_df.withColumn("latitude_rounded", round(col("latitude"), 3))\
                   .withColumn("longitude_rounded", round(col("longitude"), 3))
wildfire_df = wildfire_df.withColumn("latitude_rounded", round(col("fire_location_latitude"), 3))\
                         .withColumn("longitude_rounded", round(col("fire_location_longitude"), 3))
weather_df = weather_df.withColumn("latitude_rounded", round(col("latitude"), 3))\
                       .withColumn("longitude_rounded", round(col("longitude"), 3))

# Join wildfire data with weather data to get fire indicator
# Use a left join from weather to wildfire to retain all weather records, marking fire events
combined_df = weather_df.join(
    wildfire_df,
    (abs(weather_df.latitude_rounded - wildfire_df.latitude_rounded) <= 0.001) &
    (abs(weather_df.longitude_rounded - wildfire_df.longitude_rounded) <= 0.001) &
    (weather_df.weather_date == wildfire_df.fire_start_date),
    "left"
)

# Create a column "fire" which is True if there was a fire on that date, otherwise False
combined_df = combined_df.withColumn(
    "fire label",
    when(col("fire_start_date").isNotNull(), True).otherwise(False)
)

# Join with MODIS data to enrich with additional information
combined_df = combined_df.join(
    modis_df,
    (abs(combined_df.latitude_rounded - modis_df.latitude_rounded) <= 0.001) &
    (abs(combined_df.longitude_rounded - modis_df.longitude_rounded) <= 0.001) &
    (combined_df.weather_date == modis_df.acq_date),
    "left"
)

# Drop unnecessary columns
combined_df = combined_df.drop(
    "latitude_rounded", "longitude_rounded", "fire_location_latitude", "fire_location_longitude",
    "latitude", "longitude", "fire_start_date", "acq_date"
)

# Save the final DataFrame for downstream processing or modeling
output_path = "/content/combined_weather_fire_data"
combined_df.write.mode("overwrite").csv(output_path)

# Show the resulting DataFrame
combined_df.show()
