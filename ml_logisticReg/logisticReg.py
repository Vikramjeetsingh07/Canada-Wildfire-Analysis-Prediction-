from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lit, when, exp, radians, cos, sin, mean, avg, max as spark_max, min as spark_min,
    sum as spark_sum, lag
)
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import year
import matplotlib.pyplot as plt
from pyspark.sql.functions import count, desc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

spark = SparkSession.builder.appName("wildfire project").getOrCreate()

#inside data is all processed_data file, i.e. part-00000-d930...
data_dir = "/data/combined_final"

df = spark.read.csv(data_dir, header=True, inferSchema=True).cache()

# List of columns to keep
columns_to_keep = [
    "new_cumulative_precipitation", "et0_fao_evapotranspiration", "temperature_2m_mean", "shortwave_radiation_sum",
    "temperature_2m_min", "temperature_2m_max", "new_dryness_index", "daylight_duration",
    "new_relative_humidity", "new_precip_radiation_ratio", "wind_gusts_10m_max",
    "sunshine_duration", "weather_latitude", "weather_longitude","wind_speed_10m_max","in_modis"
]

# Select only the desired columns
df = df.select([col for col in columns_to_keep if col in df.columns])

df.show(10)

df = df.withColumn("in_modis", when(col("in_modis"), 1).otherwise(0))

# Check for null values in each column
for col_name in df.columns:
    num_nulls = df.filter(F.col(col_name).isNull()).count()
    if num_nulls > 0:
        print(f"Column '{col_name}' has {num_nulls} null values.")

#omit na row
df = df.dropna()

print("na has been removed")

# Checking for class imbalance
in_modis_count = df.filter(col("in_modis") == 1).count()
non_in_modis_count = df.filter(col("in_modis") == 0).count()

# calculate the ratio
imbalance_ratio = in_modis_count / non_in_modis_count

print(f"in_modis count: {in_modis_count}")
print(f"Non-in_modis count: {non_in_modis_count}")
print(f"Imbalance ratio: {imbalance_ratio}")

train_data, validation_data = df.randomSplit([0.8, 0.2], seed=123)

from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

# Exclude 'in_modis' from inputCols
input_cols = [col for col in columns_to_keep if col != 'in_modis']

assembler = VectorAssembler(inputCols=input_cols, outputCol="features")

train_data = assembler.transform(train_data)  
validation_data = assembler.transform(validation_data)

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="in_modis")

paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10, 20, 50]) \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol="in_modis", rawPredictionCol="rawPrediction")
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

cvModel = cv.fit(train_data)

lr_model = cvModel.bestModel

lr_best_params = lr_model.extractParamMap()
print("\nBest parameters for Logistic Regression:")
for param, value in lr_best_params.items():
    print(f"{param.name}: {value}")

# Logistic Regression model
lr_predictions = lr_model.transform(validation_data)
evaluator = BinaryClassificationEvaluator(labelCol='in_modis', rawPredictionCol='rawPrediction', metricName='areaUnderROC')
lr_auc = evaluator.evaluate(lr_predictions)
evaluator = MulticlassClassificationEvaluator(labelCol='in_modis', predictionCol='prediction', metricName='accuracy')
lr_accuracy = evaluator.evaluate(lr_predictions)

print(f"Tuned Logistic Regression AUC: {lr_auc}")
print(f"Tuned Logistic Regression Accuracy: {lr_accuracy}")