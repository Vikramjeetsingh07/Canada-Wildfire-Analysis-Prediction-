from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
import pandas as pd

spark = SparkSession.builder \
    .appName("Fire Risk") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
city_mean_temp = data.groupBy('city').agg(f.avg(f.col('temperature_2m_mean')).alias('avg_city_temp'))
avg_fire_risk_city = data.groupBy('city').agg(f.avg(f.col('new_fire_risk')).alias('avg_city_risk'))
final_data = city_mean_temp.join(avg_fire_risk_city, 'city', 'inner')


data_pd = final_data.toPandas()

plt.figure(figsize=(12,8))
sns.scatterplot(x='avg_city_temp', y='avg_city_risk', hue='city', data=data_pd)
plt.savefig('./analysis-hd/risk_per_city.png')