from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, Imputer
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import unix_timestamp


spark = SparkSession.builder \
    .appName("FirePredictionDL") \
    .getOrCreate()


file_path = "/Users/vikramjeetsingh/combined_final.csv"  
data = spark.read.csv(file_path, header=True, inferSchema=True)


exclude_cols = [
    "brightness", "scan", "track", "satellite", "instrument", "confidence",
    "version", "bright_t31", "frp", "daynight", "type", "new_fire_intensity", "new_fire_risk"
]
data = data.drop(*exclude_cols)


data = data.withColumn("date_timestamp", unix_timestamp(data["date"], format="yyyy-MM-dd"))
data = data.drop("date")


indexer = StringIndexer(inputCol="city", outputCol="city_index", handleInvalid="skip")
data = indexer.fit(data).transform(data).drop("city")


feature_cols = [col for col in data.columns if col not in ["in_modis", "city_index"]]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")


imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols)


scaler = StandardScaler(inputCol="features", outputCol="scaled_features")


data = data.withColumn("label", data["in_modis"].cast("double"))


data.groupBy("label").count().show()


positive = data.filter(data.label == 1)
negative = data.filter(data.label == 0)
train_pos, test_pos = positive.randomSplit([0.8, 0.2], seed=42)
train_neg, test_neg = negative.randomSplit([0.8, 0.2], seed=42)
train_data = train_pos.union(train_neg)
test_data = test_pos.union(test_neg)

# defining Neural Network Model
layers = [len(feature_cols), 128, 64, 32, 2]
mlp = MultilayerPerceptronClassifier(
    featuresCol="scaled_features",
    labelCol="label",
    maxIter=100,
    layers=layers,
    blockSize=128,
    seed=42
)

# pipeline
pipeline = Pipeline(stages=[assembler, imputer, scaler, mlp])

# training Model
model = pipeline.fit(train_data)

# evaluating
predictions = model.transform(test_data)

# debugging
predictions.select("label", "rawPrediction", "probability", "prediction").show(10, truncate=False)


evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

# printing Results
#print(f"Train Count: {train_data.count()}, Test Count: {test_data.count()}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R2: {r2}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1_score}")
#print(predictions.show(1))