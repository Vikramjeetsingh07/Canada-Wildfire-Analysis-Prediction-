from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
import pandas as pd

spark = SparkSession.builder \
    .appName("Soil Analysis") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
soil_categories = data.withColumn('soil_category', f.when(f.col('new_soil_moisture') < 0, 'Low Soil Moisture').when(f.col('new_soil_moisture') > 0, 'High Soil Moisture'))
humidity_categories = soil_categories.withColumn('humidity_category', f.when(f.col('new_relative_humidity') >= 60, 'High Humidity').when(f.col('new_relative_humidity') > 30, 'Medium Humidity').when(f.col('new_relative_humidity') > 0, 'Low Humidity'))
final_data = humidity_categories.groupby('soil_category', 'humidity_category').agg(f.avg(f.col('brightness')).alias('avg_brightness'))


data_pd = final_data.toPandas()
plt.figure(figsize=(12,8))
sns.barplot(x='soil_category', y='avg_brightness', hue='humidity_category', data=data_pd)
plt.savefig('./analysis-hd/soil_analysis.png')