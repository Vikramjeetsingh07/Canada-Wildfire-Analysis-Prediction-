import sys
assert sys.version_info >= (3, 5) 

from pyspark.sql import SparkSession, functions as f, types
spark = SparkSession.builder.appName('weather_tomorrow').getOrCreate()
assert spark.version >= '2.3' 
spark.sparkContext.setLogLevel('WARN')

from datetime import date, timedelta

from pyspark.ml import PipelineModel

data = spark.read.csv('./processed_data', inferSchema=True, header=True)
data = data.na.fill(0)
data = data.withColumn("in_modis", f.col("in_modis").cast("integer"))
data = data.withColumn('day-of-year', f.dayofyear(f.col('date')))
daily_averages_year = data.groupBy('day-of-year').agg(f.avg('temperature_2m_mean').alias('temperature_2m_mean'), f.avg('new_cumulative_precipitation').alias('new_cumulative_precipitation'), 
                                                      f.avg('et0_fao_evapotranspiration').alias('et0_fao_evapotranspiration'), f.avg('temperature_2m_max').alias('temperature_2m_max'),
                                                      f.avg('temperature_2m_min').alias('temperature_2m_min'), f.avg('new_dryness_index').alias('new_dryness_index'), 
                                                      f.avg('daylight_duration').alias('daylight_duration'), f.avg('sunshine_duration').alias('sunshine_duration'),
                                                      f.avg('new_relative_humidity').alias('new_relative_humidity'), f.avg('new_soil_moisture').alias('new_soil_moisture'),
                                                      f.avg('weather_latitude').alias('weather_latitude'), f.avg('weather_longitude').alias('weather_longitude'),
                                                      f.avg('apparent_temperature_max').alias('apparent_temperature_max'), f.avg('apparent_temperature_min').alias('apparent_temperature_min'),
                                                      f.avg('apparent_temperature_mean').alias('apparent_temperature_mean'), f.avg('precipitation_sum').alias('precipitation_sum'),
                                                      f.avg('rain_sum').alias('rain_sum'), f.avg('snowfall_sum').alias('snowfall_sum'), f.avg('precipitation_hours').alias('precipitation_hours'),
                                                      f.avg('wind_speed_10m_max').alias('wind_speed_10m_max'), f.avg('wind_gusts_10m_max').alias('wind_gusts_10m_max'),
                                                      f.avg('wind_direction_10m_dominant').alias('wind_direction_10m_dominant'), f.avg('shortwave_radiation_sum').alias('shortwave_radiation_sum'),
                                                      f.avg('new_temperature_range').alias('new_temperature_range'), f.avg('new_daylight_fraction').alias('new_daylight_fraction'),
                                                      f.avg('new_wind_x').alias('new_wind_x'), f.avg('new_wind_y').alias('new_wind_y'),
                                                      f.avg('new_precip_radiation_ratio').alias('new_precip_radiation_ratio'))

daily_averages_year = daily_averages_year.select(f.col('day-of-year'), f.col('temperature_2m_mean'), f.col('new_cumulative_precipitation'), f.col('et0_fao_evapotranspiration'), f.col('temperature_2m_max'),
                                        f.col('temperature_2m_min'), f.col('new_dryness_index'), f.col('daylight_duration'), f.col('sunshine_duration'), f.col('new_relative_humidity'),
                                        f.col('new_soil_moisture'), f.col('weather_latitude'), f.col('weather_longitude'), f.col('apparent_temperature_max'), f.col('apparent_temperature_min'),
                                        f.col('apparent_temperature_mean'), f.col('precipitation_sum'), f.col('rain_sum'), f.col('snowfall_sum'), f.col('precipitation_hours'),
                                        f.col('wind_speed_10m_max'), f.col('wind_gusts_10m_max'), f.col('wind_direction_10m_dominant'), f.col('shortwave_radiation_sum'),
                                        f.col('new_temperature_range'), f.col('new_daylight_fraction'), f.col('new_wind_x'), f.col('new_wind_y'), f.col('new_precip_radiation_ratio'))

start_date_2025 = date(2025, 1, 1)
end_date_2025 = date(2025, 12, 31)

delta = timedelta(days=1)

dates_2025 = []

while start_date_2025 <= end_date_2025:
    dates_2025.append({
        'date': start_date_2025.isoformat(),
        'day-of-year': start_date_2025.timetuple().tm_yday
    })
    start_date_2025 += delta
    
data_2025 = spark.createDataFrame(dates_2025)

final_data = data_2025.join(daily_averages_year, 'day-of-year').drop('day-of-year')

model = PipelineModel.load('wildfire_model_gradient')

predictions = model.transform(final_data)

prediction = predictions.filter(f.col('prediction') == 1.0).select('date').count()
print('Prediction for the date: ', prediction)