from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round, to_date, year, month, dayofmonth, hour

# Initialize Spark session
spark = SparkSession.builder.appName("WildfirePredictionPipeline").getOrCreate()

# Load datasets
modis_df = spark.read.csv("/content/fire_archive_M-C61_537264.csv", header=True, inferSchema=True)
wildfire_df = spark.read.csv("/content/fp-historical-wildfire-data-2006-2023.csv", header=True, inferSchema=True)

# Data Cleaning and Type Casting
# Remove rows with missing values or outliers (adjust thresholds as necessary)
modis_df = modis_df.dropna(subset=["acq_date", "latitude", "longitude"])
wildfire_df = wildfire_df.dropna(subset=["fire_start_date", "fire_location_latitude", "fire_location_longitude"])

# Convert date columns to DateType
modis_df = modis_df.withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))
wildfire_df = wildfire_df.withColumn("fire_start_date", to_date(col("fire_start_date"), "yyyy-MM-dd"))

# Round coordinates to enable spatial join
modis_df = modis_df.withColumn("latitude_rounded", round(col("latitude"), 2))\
                   .withColumn("longitude_rounded", round(col("longitude"), 2))
wildfire_df = wildfire_df.withColumn("latitude_rounded", round(col("fire_location_latitude"), 2))\
                         .withColumn("longitude_rounded", round(col("fire_location_longitude"), 2))

# Feature Engineering
# Extract additional temporal features for analysis
modis_df = modis_df.withColumn("year", year(col("acq_date")))\
                   .withColumn("month", month(col("acq_date")))\
                   .withColumn("day", dayofmonth(col("acq_date")))\
                   .withColumn("hour", hour(col("acq_time")))

wildfire_df = wildfire_df.withColumn("year", year(col("fire_start_date")))\
                         .withColumn("month", month(col("fire_start_date")))\
                         .withColumn("day", dayofmonth(col("fire_start_date")))

# Perform the Join
combined_df = modis_df.join(
    wildfire_df,
    (modis_df.latitude_rounded == wildfire_df.latitude_rounded) &
    (modis_df.longitude_rounded == wildfire_df.longitude_rounded) &
    (modis_df.acq_date == wildfire_df.fire_start_date),
    "inner"
)

# Drop redundant columns post-join
combined_df = combined_df.drop("latitude_rounded", "longitude_rounded", "fire_location_latitude", "fire_location_longitude")

# Save the Combined DataFrame for Downstream Processing or Modeling
output_path = "/content/combined_wildfire_data"
combined_df.write.mode("overwrite").parquet(output_path)

# Show the resulting DataFrame
combined_df.show()
