from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, lit, coalesce, when


spark = SparkSession.builder \
    .appName("Top Ten MODIS and Weather Data Join") \
    .getOrCreate()


modis_file = "top_ten_modis.csv"
weather_file = "weather_top_ten_wildfire_data.csv"


modis_df = spark.read.csv(modis_file, header=True, inferSchema=True)


modis_df = modis_df.withColumn("acq_date", to_date(col("acq_date"), "yyyy-MM-dd"))


weather_df = spark.read.csv(weather_file, header=True, inferSchema=True)


weather_df = weather_df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))

weather_df = weather_df.withColumnRenamed("city", "weather_city") \
                       .withColumnRenamed("latitude", "weather_latitude") \
                       .withColumnRenamed("longitude", "weather_longitude")





combined_df = modis_df.join(
    weather_df,
    (modis_df["acq_date"] == weather_df["date"]) & (modis_df["city"] == weather_df["weather_city"]),
    "full_outer"
)


combined_df = combined_df.select(
    coalesce(modis_df["acq_date"], weather_df["date"]).alias("date"),
    coalesce(modis_df["city"], weather_df["weather_city"]).alias("city"),
    weather_df["weather_latitude"],
    weather_df["weather_longitude"],
    weather_df["temperature_2m_max"],
    weather_df["temperature_2m_min"],
    weather_df["temperature_2m_mean"],
    weather_df["apparent_temperature_max"],
    weather_df["apparent_temperature_min"],
    weather_df["apparent_temperature_mean"],
    weather_df["daylight_duration"],
    weather_df["sunshine_duration"],
    weather_df["precipitation_sum"],
    weather_df["rain_sum"],
    weather_df["snowfall_sum"],
    weather_df["precipitation_hours"],
    weather_df["wind_speed_10m_max"],
    weather_df["wind_gusts_10m_max"],
    weather_df["wind_direction_10m_dominant"],
    weather_df["shortwave_radiation_sum"],
    weather_df["et0_fao_evapotranspiration"],
    modis_df["latitude"],
    modis_df["longitude"],
    modis_df["brightness"],
    modis_df["scan"],
    modis_df["track"],
    modis_df["satellite"],
    modis_df["instrument"],
    modis_df["confidence"],
    modis_df["version"],
    modis_df["bright_t31"],
    modis_df["frp"],
    modis_df["daynight"],
    modis_df["type"]
)


default_values = {
    "latitude": 0.0,
    "longitude": 0.0,
    "brightness": 0.0,
    "scan": 0.0,
    "track": 0.0,
    "satellite": "Unknown",
    "instrument": "Unknown",
    "confidence": 0,
    "version": 0.0,
    "bright_t31": 0.0,
    "frp": 0.0,
    "daynight": "Unknown",
    "type": 0
}
combined_df = combined_df.fillna(default_values)

# addinng labeled in_modis True or False
combined_df = combined_df.withColumn(
    "in_modis",
    when(
        (combined_df["latitude"] != 0.0) &
        (combined_df["longitude"] != 0.0) &
        (combined_df["satellite"] != "Unknown") &
        (combined_df["daynight"] != "Unknown"),
        lit(True)
    ).otherwise(lit(False))
)



sorted_df = combined_df.orderBy("date", "city")

output_path = "new_top_ten_wildfire_weather_data.csv"
sorted_df.write.csv(output_path, header=True, mode="overwrite")




spark.stop()
