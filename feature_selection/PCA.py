from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, BooleanType
from pyspark.ml.feature import VectorAssembler, PCA
import numpy as np


spark = SparkSession.builder.appName("FeatureExtractionPCA").getOrCreate()


df = spark.read.csv("/content/combined.csv", header=True, inferSchema=True)


numeric_columns = [field.name for field in df.schema.fields if isinstance(field.dataType, (DoubleType, IntegerType, BooleanType))]

# excluding modis data
excluded_columns = ["brightness", "scan", "track", "satellite", "instrument",
                    "confidence", "version", "bright_t31", "frp", "daynight",
                    "type", "in_modis"]


numeric_columns = [col for col in numeric_columns if col not in excluded_columns]

df = df.fillna(0, subset=numeric_columns)


assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")
df_assembled = assembler.transform(df)


pca = PCA(k=len(numeric_columns), inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df_assembled)


coefficients = pca_model.pc.toArray()


feature_importances = np.abs(coefficients)

importances_avg = np.mean(feature_importances, axis=0)

feature_importance_list = list(zip(numeric_columns, importances_avg))


feature_importance_list_sorted = sorted(feature_importance_list, key=lambda x: x[1], reverse=True)

for feature, importance in feature_importance_list_sorted:
    print(f"Feature: {feature}, Importance: {importance*100}")