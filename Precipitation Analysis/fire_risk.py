from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
import pandas as pd

spark = SparkSession.builder \
    .appName("Fire Risk") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
dryness_median = data.approxQuantile('new_dryness_index', [0.5], 0.01)[0]
dryness_categories = data.withColumn('dryness_c', f.when(f.col('new_dryness_index') >= dryness_median, 'High Dryness').otherwise('Low Dryness'))
cumulative_precipitation_categories = dryness_categories.withColumn('cp_c', f.when(f.col('new_cumulative_precipitation') >= 50, 'High Precipitation').when(f.col('new_cumulative_precipitation') < 50, 'Low Precipitation'))
final_data = cumulative_precipitation_categories.groupBy('dryness_c', 'cp_c').agg(f.avg(f.col('new_fire_risk')).alias('avg_risk')).orderBy('dryness_c', 'cp_c')


data_pd = final_data.toPandas()

plt.figure(figsize=(12,8))
sns.barplot(x='dryness_c', y='avg_risk', hue='cp_c', data=data_pd)
plt.savefig('./analysis-hd/fire_risk.png')