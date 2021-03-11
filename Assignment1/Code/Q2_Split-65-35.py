from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
from collections import Counter
import time

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


start = time.time()

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("COM6021_Assignment1_Question_2") \
    .config("spark.local.dir","/fastdata/lip20ps") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")


########################################################################################################################
######################################The below code is for 65,35 Split ################################################
########################################################################################################################


############################################################################################################
######################################## Question 2.A.1 ####################################################
############################################################################################################


# Loading the Rating data
ratings = spark.read.load('../Data/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true")
# ratings = spark.read.load('/Users/pskaloya/Downloads/ml-latest-small/ratings.csv', format = 'csv', inferSchema = "true", header = "true")


# Sorting the data with timestamp
ratings = ratings.orderBy('timestamp',ascending=True)

#Coutning the number of rows in rating. It will be used for fetching x% of rows.
n = ratings.count()

# Loading 50% of training data for 1st split
print("Starting loading train data")
train = ratings.limit(int(0.65*n))
train = train.cache()
train.show(10)
print("Train data loaded")

# Loading 50% of remaining as test data for 1st split
print("Starting loading test data")
ratings = ratings.orderBy('timestamp',ascending=False)
test = ratings.limit(int(0.35*n))
test = test.cache()
test.show(10)
print("Train data loaded")

############################################################################################################
######################### Question 2.A.2 ALS with lab settings #############################################
############################################################################################################


myseed = 200206518

als_65 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")

# Trainnig the model
model_65 = als_65.fit(train)

#Perdictions
predictions_65 = model_65.transform(test)
predictions_65 = predictions_65.cache()

print("Evaluation for 65/35 split")
## Question 2.A.3 for time-split 65%
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse_65 = evaluator_rmse.evaluate(predictions_65)
print("Root-mean-square error = " + str(rmse_65))

evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
mse_65 = evaluator_mse.evaluate(predictions_65)
print("Mean-square error = " + str(mse_65))

evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
mae_65 = evaluator_mae.evaluate(predictions_65)
print("Mean-Absolute error = " + str(mae_65))

############################################################################################################
################################## ASL Variation number 1 ##################################################
############################################################################################################

als_65_1 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop", rank=5,maxIter=15)

# Traninig the model
model_65_1 = als_65_1.fit(train)

#Perdictions
predictions_65_1 = model_65_1.transform(test)
predictions_65_1 = predictions_65_1.cache()


rmse_65_1 = evaluator_rmse.evaluate(predictions_65_1)
print("Root-mean-square error Variable 1 = " + str(rmse_65_1))

mse_65_1 = evaluator_mse.evaluate(predictions_65_1)
print("Mean-square error Variable 1 = " + str(mse_65_1))

mae_65_1 = evaluator_mae.evaluate(predictions_65_1)
print("Mean-Absolute error Variable 1 = " + str(mae_65_1))


############################################################################################################
################################## ASL Variation number 2 ##################################################
############################################################################################################

als_65_2 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop",maxIter=10, regParam=0.05, rank=20)

# Traninig the model
model_65_2 = als_65_2.fit(train)

#Perdictions
predictions_65_2 = model_65_2.transform(test)
predictions_65_2 = predictions_65_2.cache()


rmse_65_2 = evaluator_rmse.evaluate(predictions_65_2)
print("Root-mean-square error variation 2 = " + str(rmse_65_2))

mse_65_2 = evaluator_mse.evaluate(predictions_65_2)
print("Mean-square error variation 2 = " + str(mse_65_2))

mae_65_2 = evaluator_mae.evaluate(predictions_65_2)
print("Mean-Absolute error variation 2 = " + str(mae_65_2))



stop = time.time() - start
print("Time take",stop)

print("#####################################################")
print("Clearing cache ")
print("#####################################################")
spark.catalog.clearCache()
