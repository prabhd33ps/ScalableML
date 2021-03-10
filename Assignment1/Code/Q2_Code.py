from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from pyspark.ml.clustering import KMeans

#import pyspark.sql.functions as F
#from pyspark.sql.functions import udf
#from pyspark.sql import types as t
#import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("COM6021_Question_2") \
    .config("spark.local.dir","/fastdata/lip20ps") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

ratings = spark.read.load('/data/lip20ps/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true")

ratings = ratings.orderBy('timestamp',ascending=True)

n = ratings.count()

print("Starting loading train data")
train = ratings.limit(int(n/2))
train = train.cache()
train.show(10)

print("Train data loaded")

print("Starting loading test data")
ratings = ratings.orderBy('timestamp',ascending=False)
test = ratings.limit(int(n/2))
test = test.cache()
test.show(10)

print("Train data loaded")

## ALS with lab settings
myseed = 12345

als_50 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")

model_50 = als_50.fit(train)

#Perdictions
predictions_50 = model_50.transform(test)


## Question 2.A.3 for time-split 50%
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse_50 = evaluator_rmse.evaluate(predictions_50)
print("Root-mean-square error = " + str(rmse_50))

evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
mse_50 = evaluator_mse.evaluate(predictions_50)
print("Mean-square error = " + str(mse_50))

evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
mae_50 = evaluator_mae.evaluate(predictions_50)
print("Mean-Absolute error = " + str(mae_50))

## Question 3.1 for 50% Split


## Fecting the userFactors from ALS
userFactors_50 = model_50.userFactors

kmeans_50 = KMeans(k=20, seed=myseed)

kmeansmodel_50 = kmeans_50.fit(userFactors_50)

summary_50 = kmeansmodel_50.summary

#summary_50.clusterSizes

print("Clusters sizes are "+summary_50.clusterSizes)



