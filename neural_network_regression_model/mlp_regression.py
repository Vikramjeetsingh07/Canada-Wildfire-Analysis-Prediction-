from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, Imputer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import unix_timestamp


spark = SparkSession.builder \
    .appName("FirePredictionRegression") \
    .getOrCreate()


file_path = "/Users/vikramjeetsingh/combined_final.csv"  
data = spark.read.csv(file_path, header=True, inferSchema=True)

# exclude Unnecessary Columns and in_modis this time as should not use that for regression
exclude_cols = [
    "scan", "track", "satellite", "instrument", "version",
    "daynight", "type", "new_fire_intensity", "new_fire_risk", "in_modis"  
]
data = data.drop(*exclude_cols)

data = data.withColumn("date_timestamp", unix_timestamp(data["date"], format="yyyy-MM-dd"))
data = data.drop("date")


indexer = StringIndexer(inputCol="city", outputCol="city_index", handleInvalid="skip")
data = indexer.fit(data).transform(data).drop("city")


target_cols = ["frp", "confidence", "brightness"]  
feature_cols = [col for col in data.columns if col not in target_cols + ["city_index"]]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

imputer = Imputer(inputCols=feature_cols, outputCols=feature_cols)


scaler = StandardScaler(inputCol="features", outputCol="scaled_features")


train_test_split = {}
for target in target_cols:
    data = data.withColumn(f"{target}_label", data[target].cast("double"))
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    train_test_split[target] = (train_data, test_data)

# training Regression Models
for target in target_cols:
    print(f"\nTraining model to predict {target}...\n")

    
    train_data, test_data = train_test_split[target]
    train_data = train_data.withColumn("label", train_data[f"{target}_label"])
    test_data = test_data.withColumn("label", test_data[f"{target}_label"])

    
    regressor = LinearRegression(featuresCol="scaled_features", labelCol="label", maxIter=100, regParam=0.1, elasticNetParam=0.8)

    # pipeline
    pipeline = Pipeline(stages=[assembler, imputer, scaler, regressor])

    # training
    model = pipeline.fit(train_data)

    # model
    predictions = model.transform(test_data)
    evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    # metrics
    print(f"Results for {target} prediction:")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R2 Score: {r2}")

    # checking Predictions
    #predictions.select("label", "prediction", "scaled_features").show(10, truncate=False)
   

 