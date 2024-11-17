from pyspark.sql.functions import col
from pyspark.sql import functions as F

from pyspark.sql import SparkSession, types
import sys
import os
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("GHCN Observation Analysis") \
    .getOrCreate()

# Set the directory path
dir_path = sys.argv[1]


observation_schema = types.StructType([
    types.StructField('station', types.StringType(), False),      # Station ID
    # Date in yyyyMMdd format
    types.StructField('date', types.StringType(), False),
    types.StructField('latitude', types.DoubleType(),
                      True),      # Latitude of the station
    types.StructField('longitude', types.DoubleType(),
                      True),     # Longitude of the station
    types.StructField('elevation', types.DoubleType(),
                      True),     # Elevation of the station
    types.StructField('observation', types.StringType(),
                      False),  # Observation type
    # Value of the observation
    types.StructField('value', types.IntegerType(), False),
    types.StructField('mflag', types.StringType(),
                      True),         # Measurement flag
    types.StructField('qflag', types.StringType(),
                      True),         # Quality flag
    types.StructField('sflag', types.StringType(), True),         # Source flag
    types.StructField('obstime', types.StringType(),
                      True),       # Observation time
])
df = spark.read.json(dir_path, schema=observation_schema)
# List of relevant observation types (from the second image)
attributes = [
    "TMIN", "TMAX", "TAVG", "PRCP", "AWND", "WSFG", "WDFG", "WSF2", "WSF5",
    "SNOW", "SNWD", "WESD", "WESF", "TSUN", "PSUN", "WT01", "WT02", "WT03",
    "WT04", "WT05", "WT06", "WT07", "WT08", "WT09", "WT10", "WT11", "WT13"
]

# Filter rows where the observation is in the attributes list
filtered_df = df.filter(col("observation").isin(attributes))

# Pivot the observation column to create separate columns for each observation type
pivoted_df = filtered_df.groupBy("station", "date", "latitude", "longitude", "elevation") \
    .pivot("observation", attributes) \
    .agg(F.first("value"))

# Show the pivoted DataFrame
pivoted_df.show(5)

# Print the schema to confirm the transformation
pivoted_df.printSchema()

output_path = "ghcn_latitude_longitude_data1"

# Write the result to CSV
pivoted_df.write.csv(output_path, mode="overwrite", header=True)

print(f"Data has been successfully written to {output_path}")

spark.stop()
