from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MODIS Vancouver Data") \
    .getOrCreate()

# Load the data
data_schema = """
    latitude DOUBLE,
    longitude DOUBLE,
    brightness DOUBLE,
    scan DOUBLE,
    track DOUBLE,
    acq_date STRING,
    acq_time STRING,
    satellite STRING,
    instrument STRING,
    confidence INT,
    version STRING,
    bright_t31 DOUBLE,
    frp DOUBLE,
    daynight STRING,
    type INT
"""

# Load the data into a DataFrame (adjust the file path as needed)
data_path = "modis_canada.csv"  # Replace with the actual file path
modis_df = spark.read.csv(data_path, schema=data_schema, header=True)

# Define a bounding box around Vancouver
# Adjust as needed based on the acceptable range
vancouver_lat_min, vancouver_lat_max = 48.0, 50.0
vancouver_lon_min, vancouver_lon_max = -124.0, -122.0

# Filter the data for the Vancouver region
vancouver_df = modis_df.filter(
    (col("latitude") >= vancouver_lat_min) & (col("latitude") <= vancouver_lat_max) &
    (col("longitude") >= vancouver_lon_min) & (col("longitude") <= vancouver_lon_max)
)

# Show the results
vancouver_df.show()

# Write the results to a file (optional, scalable output)
output_path = "vancouver_modis"
vancouver_df.write.csv(output_path, header=True)

# Stop the Spark session
spark.stop()
