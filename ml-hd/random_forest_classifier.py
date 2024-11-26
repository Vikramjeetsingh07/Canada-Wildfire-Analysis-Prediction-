from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.sql.functions as f

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("Feature Importance") \
    .getOrCreate()
    
data = spark.read.csv('./processed_data', inferSchema=True, header=True)
data = data.na.fill(0)
data = data.withColumn("in_modis", f.col("in_modis").cast("integer"))

train, validation = data.randomSplit([0.75, 0.25])

inputsCols = ['temperature_2m_mean', 'new_dryness_index', 'new_relative_humidity', 'new_cumulative_precipitation', 'new_soil_moisture', 'weather_latitude', 'weather_longitude',
              'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max', 'apparent_temperature_min', 'apparent_temperature_mean', 'daylight_duration', 'sunshine_duration', 'precipitation_sum'
              , 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant', 'shortwave_radiation_sum', 'et0_fao_evapotranspiration', 
              'new_temperature_range', 'new_daylight_fraction', 'new_wind_x', 'new_wind_y', 'new_precip_radiation_ratio']


vecAssembler = VectorAssembler(inputCols=inputsCols, outputCol='features')

estimator = RandomForestClassifier(featuresCol='features', labelCol='in_modis')

wildfire_pipeline = Pipeline(stages=[vecAssembler, estimator])
wildfire_model = wildfire_pipeline.fit(train)

wildfire_model.write().overwrite().save('wildfire_model')

wildfire_model_2 = wildfire_model.transform(validation)

accuracy_evaluator = MulticlassClassificationEvaluator(predictionCol='prediction', labelCol='in_modis',
            metricName='accuracy')
accuracy = accuracy_evaluator.evaluate(wildfire_model_2)


print('Accuracy =', accuracy)



