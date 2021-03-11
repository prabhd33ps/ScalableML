from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.clustering import KMeans
from collections import Counter
import time

#import pyspark.sql.functions as F
#from pyspark.sql.functions import udf
#from pyspark.sql import types as t
#import matplotlib.pyplot as plt

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
######################################The below code is for 50,50 Split ################################################
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
train = ratings.limit(int(0.5*n))
train = train.cache()
train.show(10)
print("Train data loaded")

# Loading 50% of remaining as test data for 1st split
print("Starting loading test data")
ratings = ratings.orderBy('timestamp',ascending=False)
test = ratings.limit(int(0.5*n))
test = test.cache()
test.show(10)
print("Train data loaded")

############################################################################################################
######################### Question 2.A.2 ALS with lab settings #############################################
############################################################################################################


myseed = 200206518

als_50 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")

# Trainnig the model
model_50 = als_50.fit(train)

#Perdictions
predictions_50 = model_50.transform(test)

print("Evaluation for 50/50 split")
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

############################################################################################################
################################## ASL Variation number 1 ##################################################
############################################################################################################

als_50_1 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop", rank=5,maxIter=15)

# Traninig the model
model_50_1 = als_50_1.fit(train)

#Perdictions
predictions_50_1 = model_50_1.transform(test)


rmse_50_1 = evaluator_rmse.evaluate(predictions_50_1)
print("Root-mean-square error Variable 1 = " + str(rmse_50_1))

mse_50_1 = evaluator_mse.evaluate(predictions_50_1)
print("Mean-square error Variable 1 = " + str(mse_50_1))

mae_50_1 = evaluator_mae.evaluate(predictions_50_1)
print("Mean-Absolute error Variable 1 = " + str(mae_50_1))


############################################################################################################
################################## ASL Variation number 2 ##################################################
############################################################################################################

als_50_2 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop",maxIter=10, regParam=0.05, rank=20)

# Traninig the model
model_50_2 = als_50_2.fit(train)

#Perdictions
predictions_50_2 = model_50_1.transform(test)


rmse_50_2 = evaluator_rmse.evaluate(predictions_50_2)
print("Root-mean-square error variation 2 = " + str(rmse_50_2))

mse_50_2 = evaluator_mse.evaluate(predictions_50_2)
print("Mean-square error variation 2 = " + str(mse_50_2))

mae_50_2 = evaluator_mae.evaluate(predictions_50_2)
print("Mean-Absolute error variation 2 = " + str(mae_50_2))




############################################################################################################
################################  Question 3.1 for 50% Split ###############################################
############################################################################################################


## Fecting the userFactors from ALS
userFactors_50 = model_50.userFactors

kmeans_50 = KMeans(k=20, seed=myseed)

kmeansmodel_50 = kmeans_50.fit(userFactors_50)

summary_50 = kmeansmodel_50.summary

clusterSizes_50 = summary_50.clusterSizes

print("Clusters sizes are "+str(clusterSizes_50))

# Get the list of index from cluster sizes, where last element of the list will be index of largest cluster
indexes_50 = sorted(range(len(clusterSizes_50)), key=lambda i:clusterSizes_50[i])

print("Top 3 cluster with 50/50 split are {},{} and {}".format(clusterSizes_50[indexes_50[-1]],clusterSizes_50[indexes_50[-2]],clusterSizes_50[indexes_50[-3]]))

## Getting the perdiction data

predictions_50 = summary_50.predictions
predictions_50 = predictions_50.cache() ## Cache the perdiction data


## Getting the list of all the user from largest cluster
print("Getting the list of all the user from largest cluster for 50 split")

largestClusterUsers_50 = predictions_50.filter(predictions_50['prediction']==indexes_50[-1]).select('id').collect()
print("Converting it a python list - largestClusterUsers_50_list")
largestClusterUsers_50_list = [int(row.id) for row in largestClusterUsers_50]



print("Getting the movies id for all the users from largest cluster - Split 50")
moviesforLargestCuster_50 = train.filter(train['userID'].isin(largestClusterUsers_50_list)).filter(ratings['rating']>=4).select('movieId').collect()
print("Converting it a python list - moviesforLargestCuster_50_list")
moviesforLargestCuster_50_list = [int(r.movieId) for r in moviesforLargestCuster_50]

# largestClusterUsers_4plus_filtered = train.filter(train['userID'].isin(largestClusterUsers_50_list)).filter(ratings['rating']>=4).select('movieId').collect()

# print("geting moviesforLargestCuster")

# moviesforLargestCuster = largestClusterUsers_4plus_filtered.select('movieId').collect()

# Loading the movies data
movies = spark.read.load('../Data/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
# movies = spark.read.load('/Users/pskaloya/Downloads/ml-latest-small/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()

print("Getting all the genres for the movies against the largest cluster - split 50")
genres_largestCluster_50 = movies.filter(movies['movieID'].isin(moviesforLargestCuster_50_list)).select('genres').collect()
print("Converting it a python list - genres_largestCluster_list")
genres_largestCluster_list = [r.genres for r in genres_largestCluster_50]

# print("geting genres_largestCluster")

# genres_largestCluster = selectedMoviesID_largestCluster.select('genres').collect()

# print("geting genres_largestCluster_list")


print("Split the pipe '|' from the genres and adding them to list")
final_genres_50 = []

for genre in genres_largestCluster_list:
    if '|' in genre:
        for x in genre.split('|'):
            final_genres_50.append(x)

    else:
        final_genres_50.append(genre)



top5genres_50 = Counter(final_genres_50).most_common(5)

genres_list = [name for (name, value) in top5genres_50]

print("Top 5 genres in ",genres_list)
stop = time.time() - start
print("Time take",stop)

#
# di = {}
#
# for genre in genres_largestCluster_list:
#     if '|' in genre:
#         for x in genre.split('|'):
#             if x in di.keys():
#                 di[x] += 1
#             else:
#                 di[x] = 1
#     else:
#         if genre in di.keys():
#             di[genre] += 1
#         else:
#             di[genre] = 1

print("#####################################################")
print("Clearing cache before proceeding with the next split")
print("#####################################################")
spark.catalog.clearCache()

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


rmse_65_2 = evaluator_rmse.evaluate(predictions_65_2)
print("Root-mean-square error variation 2 = " + str(rmse_65_2))

mse_65_2 = evaluator_mse.evaluate(predictions_65_2)
print("Mean-square error variation 2 = " + str(mse_65_2))

mae_65_2 = evaluator_mae.evaluate(predictions_65_2)
print("Mean-Absolute error variation 2 = " + str(mae_65_2))

















print("#####################################################")
print("Clearing cache before proceeding with the next split")
print("#####################################################")
spark.catalog.clearCache()

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

# Traninig the model
model_80_2 = als_80_2.fit(train)

#Perdictions
predictions_80_2 = model_80_2.transform(test)


rmse_80_2 = evaluator_rmse.evaluate(predictions_80_2)
print("Root-mean-square error variation 2 = " + str(rmse_80_2))

mse_80_2 = evaluator_mse.evaluate(predictions_80_2)
print("Mean-square error variation 2 = " + str(mse_80_2))

mae_80_2 = evaluator_mae.evaluate(predictions_80_2)
print("Mean-Absolute error variation 2 = " + str(mae_80_2))



