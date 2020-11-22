# BIG DATA

# Import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import pyspark dependencies
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.conf import SparkConf
from pyspark import SparkContext

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

pd.set_option('display.max_columns', None); pd.set_option('display.max_rows', None);
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import findspark
findspark.init("C:\spark")

spark = SparkSession.builder \
    .master("local") \
    .appName("pyspark_application") \
    .config("spark.executer.memory", "16gb") \
    .getOrCreate()

sc = spark.sparkContext
sc
# sc.stop()


## Basic DataFrame Operations

# Spark DataFrame
spark_df = spark.read.csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\churn2.csv", header=True, inferSchema=True)
spark_df
spark_df.printSchema()
type(spark_df) # pyspark.sql.dataframe.DataFrame

# Pandas DataFrame
df = sns.load_dataset("diamonds")
type(df) # pandas.core.frame.DataFrame

df.head()
spark_df.head()

df.dtypes
spark_df.dtypes

# df.shape
# spark_df.shape

#
spark_df.show(3, truncate=True)
spark_df.count()
spark_df.columns
spark_df.describe().show()
spark_df.describe("Age", "Churn").show(5)

# Variable and Observation Selection

spark_df.select("Age", "Churn").show(5)

# Show the ones older than 40
spark_df.filter(spark_df.Age > 40).show()
spark_df.filter(spark_df.Age > 40).count()

# group by operations
spark_df.groupby("Churn").count().show()
spark_df.groupby("Churn").agg({"Age": "mean"}).show()

spark_df.crosstab("Churn", "Age").show()


## SQL operations

# Creating table view, temp_table
spark_df.createOrReplaceTempView("tbl_df")

# Basics of SQL
spark.sql("show databases").show()
spark.sql("show tables").show()
spark.sql("select Age from tbl_df limit 5").show()
spark.sql("select Exited as churn, avg(Age) as average from tbl_df group by Churn").show()


## Big Data Visualization

# Leider geht's nicht!
sns.barplot(x="Churn", y=spark_df.Churn.index, data=spark_df)
plt.show()

# Make aggregations in big data platform, then create a dataframe for the results and finally visualize values.
spark_df.groupby("Churn").agg({"Age": "mean"}).show()
spark_df.groupby("Churn").agg({"Age": "mean"}).toPandas()
sdf = spark_df.groupby("Churn").agg({"Age": "mean"}).toPandas()
type(sdf) # pandas.core.frame.DataFrame

# Now, we can visualize what we created.
sdf.columns # Index(['Exited', 'avg(Age)'], dtype='object')
sns.barplot(x="Churn", y='avg(Age)', data=sdf)
plt.show()


## Customer Churn Prediction Model

# Data Preparation

# Load the dataset again
spark_df = spark.read.csv(r"C:\Users\yakup\PycharmProjects\dsmlbc\datasets\churn2.csv", header=True, inferSchema=True)
spark_df.printSchema()
# Lower all feature names for simplicity
spark_df = spark_df.toDF(*[c.lower() for c in spark_df.columns]) # lower all feature names
spark_df.show(5)
# Rename index column
spark_df = spark_df.withColumnRenamed("_c0", "index") # rename index column
spark_df.show(5)
spark_df.columns

# Data Understanding
spark_df.select('age', 'total_purchase', 'account_manager', 'years', 'num_sites', 'churn').describe().toPandas().transpose()
spark_df.filter(spark_df.age > 47).count()
spark_df.groupby("churn").count().show()

# See the means with respect to target variable
for col in ['age', 'total_purchase', 'account_manager', 'years', 'num_sites']:
    spark_df.groupby("churn").agg({col: "mean"}).show()

# Check for missing values
from pyspark.sql.functions import when, count, col
spark_df.select([count(when(col(c).isNull(), c)).alias(c) for c in spark_df.columns]).toPandas().T

# Drop missing values
spark_df = spark_df.dropna()

# Drop the columns that we do not need for our Model
spark_df = spark_df.drop('index', 'names')
spark_df.show(5)

# Create a new feature called age/tenure
spark_df = spark_df.withColumn('age/years', spark_df.age/spark_df.years)
spark_df.show(5)

# Define dependent variable
stringIndexer = StringIndexer(inputCol="churn", outputCol="label")
mod = stringIndexer.fit(spark_df)
indexed = mod.transform(spark_df)

spark_df = indexed.withColumn("label", indexed["label"].cast("integer"))
spark_df.show(5)

# Define independent variable
spark_df.columns
cols = ['age', 'total_purchase', 'account_manager', 'years', 'num_sites']
va = VectorAssembler(inputCols=cols, outputCol="features")
va_df = va.transform(spark_df)
va_df.show(5)

# Final dataframe
final_df = va_df.select(["features", "label"])
final_df.show(5)

# Test Train Split
splits = final_df.randomSplit([0.70, 0.30])
train_df = splits[0]
test_df = splits[1]

train_df.show(10)
test_df.show(10)


## MODELING

# GBM
gbm = GBTClassifier(maxIter=10, featuresCol="features", labelCol="label")
gbm_model = gbm.fit(train_df)

# Make predictions
y_pred = gbm_model.transform(test_df)
y_pred.show()

# Calculate accuracy
ac = y_pred.select("label", "prediction")
ac.show(5)
ac.filter(ac.label == ac.prediction).count() / ac.count() # 0.865546218487395

# Model Tuning

evaluator = BinaryClassificationEvaluator()

paramGrid = (ParamGridBuilder()
             .addGrid(gbm.maxDepth, [2, 4, 6])
             .addGrid(gbm.maxBins, [20, 30])
             .addGrid(gbm.maxIter, [10, 20])
             .build())

cv = CrossValidator(estimator=gbm, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
cv_model = cv.fit(train_df)

y_pred = cv_model.transform(test_df)
ac = y_pred.select("label", "prediction")
ac.filter(ac.label == ac.prediction).count() / ac.count() # 0.8613445378151261
evaluator.evaluate(y_pred) # 0.8209007741027446

print("Test Area Under ROC: " + str(evaluator.evaluate(y_pred, {evaluator.metricName: "areaUnderROC"})))

# Apply the model for new customers
names = pd.Series(["Ali Ahmetoğlu", "Taner Gün", "Harika Gündüz", "Polat Alemdar", "Ata Bakmayan Ali"])
age = pd.Series([38, 43, 34, 50, 40])
total_purchase = pd.Series([30000, 10000, 6000, 30000, 100000])
account_manager = pd.Series([1, 0, 0, 1, 1])
years = pd.Series([20, 10, 3, 8, 30])
num_sites = pd.Series([30, 8, 8, 6, 50])

# Create a dataframe for new_customers
new_customers = pd.DataFrame({'names': names,
                              'age': age,
                              'total_purchase': total_purchase,
                              'account_manager': account_manager,
                              'years': years,
                              'num_sites': num_sites})
# Convert it into spark_df
new_sdf = spark.createDataFrame(new_customers)
type(new_sdf)
new_sdf.show()

new_customers = va.transform(new_sdf)
results = cv_model.transform(new_customers)
results.select("names", "prediction").show()

# Stop Spark session
sc.stop()
