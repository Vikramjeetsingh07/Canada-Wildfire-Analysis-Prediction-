from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f

spark = SparkSession.builder \
    .appName("Trends") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
data_year = data.withColumn('year', f.year(f.col('date')))
final_data = data_year.groupBy('year', 'city').agg(f.avg(f.col('new_dryness_index')).alias('avg_dryness'), f.avg(f.col('new_cumulative_precipitation')).alias('avg_precipitation'), f.avg(f.col('new_relative_humidity')).alias('avg_humidity'), f.avg(f.col('new_fire_risk')).alias('avg_risk'))

data_pd = final_data.toPandas()
melted_data = data_pd.melt(
    id_vars=['year', 'city'],
    value_vars=['avg_dryness', 'avg_precipitation', 'avg_humidity', 'avg_risk'],
    var_name='Metric',
    value_name='Value'
)
plt.figure(figsize=(12,8))
g = sns.FacetGrid(melted_data, col="Metric", hue="city", col_wrap=2, height=5, sharey=False)
g.map(sns.lineplot, "year", "Value", marker="o")
g.add_legend(title="City")
plt.savefig('./analysis-hd/yearly_city_trends.png')