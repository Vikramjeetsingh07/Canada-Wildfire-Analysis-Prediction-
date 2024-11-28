from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

#This pipeline is for all True labels meaning all incidents when fire happened with weather on that date
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
    (modis_df.acq_date == weather_df.date) & (modis_df.city == weather_df.weather_city),
    "left"
).select(
    modis_df["*"], 
    weather_df["weather_latitude"],
    weather_df["weather_longitude"],
    weather_df["temperature_2m_max"],
    weather_df["temperature_2m_min"],
    weather_df["temperature_2m_mean"],
    weather_df["weather_code"],
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
    weather_df["et0_fao_evapotranspiration"]

)


combined_df = combined_df.orderBy(col("city").asc(), col("acq_date").asc())


output_path = "combined_top_ten_wildfire_weather_data.csv"
combined_df.write.csv(output_path, header=True, mode="overwrite")


spark.stop()



