from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f
import pandas as pd

spark = SparkSession.builder \
    .appName("Feature Correlation") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
data = data.withColumn('fire_occurrence', f.when(f.col('new_fire_intensity') > 0, 1).otherwise(0))


data_pd = data.toPandas()
plt.figure(figsize=(12,8))
sns.pairplot(data_pd, hue='fire_occurrence', vars=['sunshine_duration', 'daylight_duration', 'new_dryness_index', 'temperature_2m_mean'])
plt.savefig('./analysis-hd/feature_correlation_analysis.png')