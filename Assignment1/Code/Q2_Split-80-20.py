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

# spark = SparkSession.builder \
#     .master("local[*]") \
#     .appName("COM6021_Assignment1_Question_2") \
#     .getOrCreate()


spark = SparkSession.builder \
    .master("local[*]") \
    .appName("COM6021_Assignment1_Question_2") \
    .config("spark.local.dir","/fastdata/lip20ps") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")




########################################################################################################################
############################The below code is for 80,20 Split for Question 2############################################
########################################################################################################################
########################################################################################################################



# Loading the Rating data
ratings = spark.read.load('../Data/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true")
# ratings = spark.read.load('/Users/pskaloya/Downloads/ml-latest-small/ratings.csv', format = 'csv', inferSchema = "true", header = "true")


# Sorting the data with timestamp
ratings = ratings.orderBy('timestamp',ascending=True)

#Coutning the number of rows in rating. It will be used for fetching x% of rows.
n = ratings.count()

# Loading 50% of training data for 1st split
print("Starting loading train data")
train = ratings.limit(int(0.8*n))
train = train.cache()
train.show(10)
print("Train data loaded")

# Loading 50% of remaining as test data for 1st split
print("Starting loading test data")
ratings = ratings.orderBy('timestamp',ascending=False)
test = ratings.limit(int(0.2*n))
test = test.cache()
test.show(10)
print("Train data loaded")

############################################################################################################
######################### Question 2.A.2 ALS with lab settings #############################################
############################################################################################################

myseed = 200206518

als_80 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")

# Trainnig the model
model_80 = als_80.fit(train)

#Perdictions
predictions_80 = model_80.transform(test)
predictions_80 = predictions_80.cache()

print("Evaluation for 80/20 split")

## Question 2.A.3 for time-split 80%
evaluator_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse_80 = evaluator_rmse.evaluate(predictions_80)
print("Root-mean-square error = " + str(rmse_80))

evaluator_mse = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
mse_80 = evaluator_mse.evaluate(predictions_80)
print("Mean-square error = " + str(mse_80))

evaluator_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")
mae_80 = evaluator_mae.evaluate(predictions_80)
print("Mean-Absolute error = " + str(mae_80))

############################################################################################################
################################## ASL Variation number 1 ##################################################
############################################################################################################

als_80_1 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop", rank=5,maxIter=15)

# Traninig the model
model_80_1 = als_80_1.fit(train)

#Perdictions
predictions_80_1 = model_80_1.transform(test)
predictions_80_1 = predictions_80_1.cache()


rmse_80_1 = evaluator_rmse.evaluate(predictions_80_1)
print("Root-mean-square error Variable 1 = " + str(rmse_80_1))

mse_80_1 = evaluator_mse.evaluate(predictions_80_1)
print("Mean-square error Variable 1 = " + str(mse_80_1))

mae_80_1 = evaluator_mae.evaluate(predictions_80_1)
print("Mean-Absolute error Variable 1 = " + str(mae_80_1))


############################################################################################################
################################## ASL Variation number 2 ##################################################
############################################################################################################

als_80_2 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop",maxIter=10, regParam=0.05, rank=20)

# Training the model
model_80_2 = als_80_2.fit(train)

#Perdictions
predictions_80_2 = model_80_2.transform(test)
predictions_80_2 = predictions_80_2.cache()


rmse_80_2 = evaluator_rmse.evaluate(predictions_80_2)
print("Root-mean-square error variation 2 = " + str(rmse_80_2))

mse_80_2 = evaluator_mse.evaluate(predictions_80_2)
print("Mean-square error variation 2 = " + str(mse_80_2))

mae_80_2 = evaluator_mae.evaluate(predictions_80_2)
print("Mean-Absolute error variation 2 = " + str(mae_80_2))








stop = time.time() - start
print("Time take",stop)

print("#####################################################")
print("Clearing cache ")
print("#####################################################")
spark.catalog.clearCache()
