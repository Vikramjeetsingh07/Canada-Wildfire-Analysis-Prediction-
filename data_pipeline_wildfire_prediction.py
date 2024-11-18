from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("MODIS and Weather Data Join") \
    .getOrCreate()

# File paths
modis_file = "vancouver_modis.csv"  # Replace with the actual path to the MODIS data
weather_file = "vancouver_weather_2006_2023.csv"  # Replace with the actual path to the weather data

# Load MODIS data
modis_df = spark.read.csv(modis_file, header=True, inferSchema=True)
# Convert acq_date to DateType
modis_df = modis_df.withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))

# Load Weather data
weather_df = spark.read.csv(weather_file, header=True, inferSchema=True)
# Convert date to DateType (remove timestamp for matching)
weather_df = weather_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

# Perform the join (left join to keep all MODIS data)
combined_df = modis_df.join(weather_df, modis_df.acq_date == weather_df.date, "left").drop(weather_df.date)

# Show the combined DataFrame
combined_df.show(truncate=False)

# Save the result to a CSV file
output_path = "wildfire_data.csv"
combined_df.write.csv(output_path, header=True, mode="overwrite")

# Stop the SparkSession
spark.stop()

