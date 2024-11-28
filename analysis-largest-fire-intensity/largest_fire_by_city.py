from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from shapely.geometry import MultiPoint
import pyspark.sql.types as T
import matplotlib.pyplot as plt
import pandas as pd
# Initialize Spark session
spark = SparkSession.builder.appName(
    "ForestFireFeatureEngineering").getOrCreate()

# Step 1: Load the Data
# Load the dataset from the directory (assuming the CSV files are inside 'new_data/' directory)
data_dir = "../../new_data/processed_data/"
df = spark.read.csv(data_dir, header=True, inferSchema=True)

# Extract year from the 'date' column
df = df.withColumn("year", F.year(F.to_date("date", "yyyy-MM-dd")))

# Find the largest fire intensity for each year across all cities
# Group by year and find the maximum new_fire_intensity
largest_fire_intensity_per_year = df.groupBy("year").agg(
    F.max("new_fire_intensity").alias("largest_fire_intensity")
)

# Use aliases to join and avoid ambiguity
df_alias = df.select("year", "city", "new_fire_intensity").alias(
    "df")  # Select only necessary columns to avoid ambiguity
largest_fire_intensity_alias = largest_fire_intensity_per_year.alias("largest")

# Join back to the original DataFrame to get the corresponding city for each maximum intensity
largest_fire_intensity_with_city = largest_fire_intensity_alias.join(
    df_alias,
    (largest_fire_intensity_alias["year"] == df_alias["year"]) &
    (largest_fire_intensity_alias["largest_fire_intensity"]
     == df_alias["new_fire_intensity"]),
    how="left"
)

# largest_fire_intensity_with_city.show()
# Step 1: Calculate the Largest Fire Intensity Per Year for Each City
# Group by year and city to find the maximum new_fire_intensity per year for each city
largest_fire_intensity_per_year = df.groupBy("city", "year").agg(
    F.max("new_fire_intensity").alias("largest_fire_intensity")
)

# Step 2: Calculate the Average of the Largest Fire Intensities Across All Years for Each City
# Group by city and calculate the average of the largest fire intensities
average_largest_fire_intensity_per_city = largest_fire_intensity_per_year.groupBy("city").agg(
    F.avg("largest_fire_intensity").alias("average_largest_fire_intensity")
)

# Display the average largest fire intensity for each city
average_largest_fire_intensity_per_city.show()

# TODO: Largest fire intensity with only one day, not consecutive

# # Define a window specification for each city and date
# window_spec_location = Window.partitionBy("city", "date")
#
# # Calculate the minimum and maximum latitude and longitude for each city on the same day
# df = df.withColumn("min_latitude", F.min("latitude").over(window_spec_location)) \
#        .withColumn("max_latitude", F.max("latitude").over(window_spec_location)) \
#        .withColumn("min_longitude", F.min("longitude").over(window_spec_location)) \
#        .withColumn("max_longitude", F.max("longitude").over(window_spec_location))
#
# # Calculate the range for latitude and longitude
# df = df.withColumn("latitude_range", F.col("max_latitude") - F.col("min_latitude")) \
#        .withColumn("longitude_range", F.col("max_longitude") - F.col("min_longitude"))
#
# # Calculate the area based on latitude and longitude range for each day in each city
# df = df.withColumn("area", F.col("latitude_range") * F.col("longitude_range"))
#
# # Group by city to find the largest area
# largest_fire_area_per_city = df.groupBy("city").agg(
#     F.max("area").alias("largest_fire_area")
# )
#
# # Show the result
# largest_fire_area_per_city.show()
# Step 1: Assign Group IDs for Consecutive Date Sequences
# Convert 'date' to a date type for easier processing
df = df.withColumn("date", F.to_date("date", "yyyy-MM-dd"))

# Define a window partitioned by 'city' and ordered by 'date'
window_spec = Window.partitionBy("city").orderBy("date")

# Calculate the difference in days between the current and previous fire event in the same city
df = df.withColumn("prev_date", F.lag("date").over(window_spec))
df = df.withColumn("date_diff", F.when(F.col("prev_date").isNull(), 1)
                                 .otherwise(F.datediff("date", "prev_date")))

# Create a 'group_id' column to group consecutive dates together
df = df.withColumn("group_id", F.sum(
    F.when(F.col("date_diff") > 1, 1).otherwise(0)).over(window_spec))

# Step 2: Aggregate Latitude and Longitude Points for Each Group
# Collect all latitude and longitude points for each group into a list
grouped_df = df.groupBy("city", "group_id").agg(
    F.collect_list(F.struct("latitude", "longitude")).alias("lat_long_points")
)

# Step 3: Calculate Area using Convex Hull
# Define a function to calculate the area of the convex hull


def calculate_convex_hull_area(points):
    if len(points) < 3:
        return 0.0  # Convex hull area is zero if there are less than 3 points
    coords = [(point.latitude, point.longitude) for point in points]
    multi_point = MultiPoint(coords)
    convex_hull = multi_point.convex_hull
    return convex_hull.area


# Register the function as a UDF
calculate_convex_hull_area_udf = F.udf(
    calculate_convex_hull_area, T.DoubleType())

# Apply the UDF to calculate the area of the convex hull for each group
grouped_df = grouped_df.withColumn(
    "convex_hull_area", calculate_convex_hull_area_udf(F.col("lat_long_points")))

# Step 4: Find the Largest Fire Area Per City
largest_fire_area_per_city = grouped_df.groupBy("city").agg(
    F.max("convex_hull_area").alias("largest_fire_area")
)

# Display the result
largest_fire_area_per_city.show()


# Collect average largest fire intensity per city into Pandas DataFrame
avg_largest_fire_intensity_pd = average_largest_fire_intensity_per_city.toPandas()

# Collect largest fire area per city into Pandas DataFrame
largest_fire_area_pd = largest_fire_area_per_city.toPandas()

# Step 2: Plot and Save Average Largest Fire Intensity Per City

plt.figure(figsize=(10, 6))
plt.bar(avg_largest_fire_intensity_pd['city'],
        avg_largest_fire_intensity_pd['average_largest_fire_intensity'], color='orange')
plt.xlabel('City')
plt.ylabel('Average Largest Fire Intensity')
plt.title('Average Largest Fire Intensity Per City Over All Years')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save the figure as PNG
plt.savefig('average_largest_fire_intensity_per_city.png')
plt.close()

# Step 3: Plot and Save Largest Fire Area Per City

plt.figure(figsize=(10, 6))
plt.bar(largest_fire_area_pd['city'],
        largest_fire_area_pd['largest_fire_area'], color='red')
plt.xlabel('City')
plt.ylabel('Largest Fire Area')
plt.title('Largest Fire Area Per City')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save the figure as PNG
plt.savefig('largest_fire_area_per_city.png')
plt.close()
