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

# Load the data into a DataFrame 
data_path = "modis_canada.csv" 
modis_df = spark.read.csv(data_path, schema=data_schema, header=True)

# Define a bounding box around Vancouver
vancouver_lat_min, vancouver_lat_max = 48.0, 50.0
vancouver_lon_min, vancouver_lon_max = -124.0, -122.0

# Filter the data for the Vancouver region
vancouver_df = modis_df.filter(
    (col("latitude") >= vancouver_lat_min) & (col("latitude") <= vancouver_lat_max) &
    (col("longitude") >= vancouver_lon_min) & (col("longitude") <= vancouver_lon_max)
)

# Sort the filtered data by date and time
sorted_vancouver_df = vancouver_df.orderBy(col("acq_date").asc(), col("acq_time").asc())

# Write the results to CSV file in the output folder
output_path = "vancouver_modis_data_2006_2023"  
sorted_vancouver_df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

# Stop the Spark session
spark.stop()
