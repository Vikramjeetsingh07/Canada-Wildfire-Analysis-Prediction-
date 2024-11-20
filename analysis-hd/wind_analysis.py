from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
import pandas as pd

spark = SparkSession.builder \
    .appName("Wind Analysis") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
wind_speed_category = data.withColumn('wind_speed_category', f.when(f.col('wind_speed_10m_max') >= 10, 'High Wind Speed').when(f.col('wind_speed_10m_max') >= 5, 'Medium Wind Speed').when(f.col('wind_speed_10m_max') > 0, 'Low Wind Speed'))
wind_gust_category = wind_speed_category.withColumn('wind_gust_category', f.when(f.col('wind_gusts_10m_max') >= 50, 'High Gust Speed').when(f.col('wind_gusts_10m_max') >= 25, 'Medium Gust Speed').when(f.col('wind_gusts_10m_max') > 0, 'Low Gust Speed'))
wind_speed_impact = wind_gust_category.groupBy('wind_speed_category', 'wind_gust_category').agg(f.max(f.col('new_fire_intensity')).alias('max_intensity'))

data_pd = wind_speed_impact.toPandas()
heatmap_pivot = data_pd.pivot('wind_speed_category', 'wind_gust_category', 'max_intensity')
plt.figure(figsize=(12,8))
sns.heatmap(heatmap_pivot, annot=True, cmap='Reds', fmt='.2f', cbar_kws={'label': 'Intensity'})
plt.savefig('./analysis-hd/wind_analysis.png')