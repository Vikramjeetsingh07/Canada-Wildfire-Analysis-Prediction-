from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit


spark = SparkSession.builder \
    .appName("MODIS Multiple Locations Data") \
    .getOrCreate()


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


data_path = "modis_canada.csv"
modis_df = spark.read.csv(data_path, schema=data_schema, header=True)


locations = [
    {"name": "Fort McMurray", "latitude": 56.7265, "longitude": -111.379, "range": 0.5},
    {"name": "Kelowna", "latitude": 49.888, "longitude": -119.496, "range": 0.5},
    {"name": "Kamloops", "latitude": 50.6745, "longitude": -120.3273, "range": 0.5},
    {"name": "Prince George", "latitude": 53.9171, "longitude": -122.7497, "range": 0.5},
    {"name": "Vancouver Island", "latitude": 49.6508, "longitude": -125.4492, "range": 0.5},
    {"name": "Lytton", "latitude": 50.2316, "longitude": -121.5824, "range": 0.5},
    {"name": "Penticton", "latitude": 49.4991, "longitude": -119.5937, "range": 0.5},
    {"name": "Williams Lake", "latitude": 52.141, "longitude": -122.141, "range": 0.5},
    {"name": "Grande Prairie", "latitude": 55.1707, "longitude": -118.7884, "range": 0.5},
    {"name": "Edson", "latitude": 53.581, "longitude": -116.439, "range": 0.5}
]


filtered_df = None


for location in locations:
    lat_min = location["latitude"] - location["range"]
    lat_max = location["latitude"] + location["range"]
    lon_min = location["longitude"] - location["range"]
    lon_max = location["longitude"] + location["range"]
    

    city_df = modis_df.filter(
        (col("latitude") >= lat_min) & (col("latitude") <= lat_max) &
        (col("longitude") >= lon_min) & (col("longitude") <= lon_max)
    ).withColumn("city", lit(location["name"]))
    

    if filtered_df is None:
        filtered_df = city_df
    else:
        filtered_df = filtered_df.union(city_df)


sorted_filtered_df = filtered_df.orderBy(col("city").asc(), col("acq_date").asc(), col("acq_time").asc())


output_path = "modis_multiple_locations_data"
sorted_filtered_df.coalesce(1).write.csv(output_path, header=True, mode="overwrite")


spark.stop()
