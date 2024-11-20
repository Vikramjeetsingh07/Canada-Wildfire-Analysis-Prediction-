from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
import pandas as pd

spark = SparkSession.builder \
    .appName("Fire Risk") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
city_mean_intensity = data.groupBy('city').agg(f.avg(f.col('new_fire_intensity')).alias('avg_city_intensity'))
avg_city_wind_speed = data.groupBy('city').agg(f.avg(f.col('wind_speed_10m_max')).alias('avg_city_wind_speed'))
final_data = city_mean_intensity.join(avg_city_wind_speed, 'city', 'inner')

data_pd = final_data.toPandas()
plt.figure(figsize=(12,8))
sns.scatterplot(x='avg_city_intensity', y='avg_city_wind_speed', hue='city', data=data_pd)
plt.savefig('./analysis-hd/city_compare_speed_intensity.png')