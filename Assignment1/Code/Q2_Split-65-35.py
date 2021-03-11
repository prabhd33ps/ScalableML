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

# Loading 65% of training data for 1st split
print("Starting loading train data")
train = ratings.limit(int(0.65*n))
train = train.cache()
train.show(10)
print("Train data loaded")

# Loading 35% of remaining as test data for 1st split
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
print("Root-mean-square error variation 1 = " + str(rmse_65_1))

mse_65_1 = evaluator_mse.evaluate(predictions_65_1)
print("Mean-square error variation 1 = " + str(mse_65_1))

mae_65_1 = evaluator_mae.evaluate(predictions_65_1)
print("Mean-Absolute error variation 1 = " + str(mae_65_1))


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


############################################################################################################
################################  Question 3.1 for 65-35 Split ###############################################
############################################################################################################


## Fecting the userFactors from ALS
userFactors_65 = model_65.userFactors

kmeans_65 = KMeans(k=20, seed=myseed)

kmeansmodel_65 = kmeans_65.fit(userFactors_65)

summary_65 = kmeansmodel_65.summary

clusterSizes_65 = summary_65.clusterSizes

print("Clusters sizes are "+str(clusterSizes_65))

# Get the list of index from cluster sizes, where last element of the list will be index of largest cluster
indexes_65 = sorted(range(len(clusterSizes_65)), key=lambda i:clusterSizes_65[i])

print("Top 3 cluster with 65/35 split are {},{} and {}".format(clusterSizes_65[indexes_65[-1]],clusterSizes_65[indexes_65[-2]],clusterSizes_65[indexes_65[-3]]))

## Getting the perdiction data

predictions_65 = summary_65.predictions
predictions_65 = predictions_65.cache() ## Cache the perdiction data


## Getting the list of all the user from largest cluster
print("Getting the list of all the user from largest cluster for 65 split")

largestClusterUsers_65 = predictions_65.filter(predictions_65['prediction']==indexes_65[-1]).select('id').collect()
print("Converting it a python list - largestClusterUsers_65_list")
largestClusterUsers_65_list = [int(row.id) for row in largestClusterUsers_65]



print("Fetching for train data")
print("Getting the movies id for all the users from largest cluster - Split 65")
moviesforLargestCuster_65 = train.filter(train['userID'].isin(largestClusterUsers_65_list)).filter(ratings['rating']>=4).select('movieId').collect()
print("Converting it a python list - moviesforLargestCuster_65_list")
moviesforLargestCuster_65_list = [int(r.movieId) for r in moviesforLargestCuster_65]

## Removing the dulpicate values
moviesforLargestCuster_65_set = set(moviesforLargestCuster_65_list)

# Loading the movies data
movies = spark.read.load('../Data/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
# movies = spark.read.load('/Users/pskaloya/Downloads/ml-latest-small/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()

print("Getting all the genres for the movies against the largest cluster - split 65")
genres_largestCluster_65 = movies.filter(movies['movieID'].isin(moviesforLargestCuster_65_set)).select('genres').collect()
print("Converting it a python list - genres_largestCluster_list")
genres_largestCluster_list = [r.genres for r in genres_largestCluster_65]



print("Split the pipe '|' from the genres and adding them to list")
final_genres_65 = []

for genre in genres_largestCluster_list:
    if '|' in genre:
        for x in genre.split('|'):
            final_genres_65.append(x)

    else:
        final_genres_65.append(genre)


top5genres_65 = Counter(final_genres_65).most_common(5)

genres_list = [name for (name, value) in top5genres_65]

print("Top 5 genres for 65-35 split ", genres_list)








print("Fetching for test data")
print("Getting the movies id for all the users from largest cluster - Split 50")
moviesforLargestCuster_65_test = test.filter(test['userID'].isin(largestClusterUsers_65_list)).filter(ratings['rating']>=4).select('movieId').collect()
print("Converting it a python list - moviesforLargestCuster_50_list")
moviesforLargestCuster_65_test_list = [int(r.movieId) for r in moviesforLargestCuster_65_test]

## Removing the dulpicate values
moviesforLargestCuster_65_test_set = set(moviesforLargestCuster_65_test_list)

print("Getting all the genres for the movies against the largest cluster - split 50")
genres_largestCluster_test_65 = movies.filter(movies['movieID'].isin(moviesforLargestCuster_65_test_set)).select('genres').collect()
print("Converting it a python list - genres_largestCluster_list")
genres_largestCluster_test_list = [r.genres for r in genres_largestCluster_test_65]



print("Split the pipe '|' from the genres and adding them to list")
final_genres_test_65 = []

for genre in genres_largestCluster_test_list:
    if '|' in genre:
        for x in genre.split('|'):
            final_genres_test_65.append(x)

    else:
        final_genres_test_65.append(genre)


top5genres_test_65 = Counter(final_genres_test_65).most_common(5)

genres_test_list = [name for (name, value) in top5genres_test_65]

print("Top 5 genres for 50-50 split for train data ", genres_test_list)


stop = time.time() - start
print("Time take",stop)

print("#####################################################")
print("Clearing cache ")
print("#####################################################")
spark.catalog.clearCache()

spark.stop()