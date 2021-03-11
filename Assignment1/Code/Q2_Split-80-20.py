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

# Loading 80% of training data for 1st split
print("Starting loading train data")
train = ratings.limit(int(0.8*n))
train = train.cache()
train.show(10)
print("Train data loaded")

# Loading 20% of remaining as test data for 1st split
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
print("Root-mean-square error variation 1 = " + str(rmse_80_1))

mse_80_1 = evaluator_mse.evaluate(predictions_80_1)
print("Mean-square error variation 1 = " + str(mse_80_1))

mae_80_1 = evaluator_mae.evaluate(predictions_80_1)
print("Mean-Absolute error variation 1 = " + str(mae_80_1))


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


############################################################################################################
################################  Question 3.1 for 80% Split ###############################################
############################################################################################################


## Fecting the userFactors from ALS
userFactors_80 = model_80.userFactors

kmeans_80 = KMeans(k=20, seed=myseed)

kmeansmodel_80 = kmeans_80.fit(userFactors_80)

summary_80 = kmeansmodel_80.summary

clusterSizes_80 = summary_80.clusterSizes

print("Clusters sizes are "+str(clusterSizes_80))

# Get the list of index from cluster sizes, where last element of the list will be index of largest cluster
indexes_80 = sorted(range(len(clusterSizes_80)), key=lambda i:clusterSizes_80[i])

print("Top 3 cluster with 80/20 split are {},{} and {}".format(clusterSizes_80[indexes_80[-1]],clusterSizes_80[indexes_80[-2]],clusterSizes_80[indexes_80[-3]]))

## Getting the perdiction data

predictions_80 = summary_80.predictions
predictions_80 = predictions_80.cache() ## Cache the perdiction data


## Getting the list of all the user from largest cluster
print("Getting the list of all the user from largest cluster for 80 split")

largestClusterUsers_80 = predictions_80.filter(predictions_80['prediction']==indexes_80[-1]).select('id').collect()
print("Converting it a python list - largestClusterUsers_80_list")
largestClusterUsers_80_list = [int(row.id) for row in largestClusterUsers_80]


print("Fetching for train data")
print("Getting the movies id for all the users from largest cluster - Split 80")
moviesforLargestCuster_80 = train.filter(train['userID'].isin(largestClusterUsers_80_list)).filter(ratings['rating']>=4).select('movieId').collect()
print("Converting it a python list - moviesforLargestCuster_80_list")
moviesforLargestCuster_80_list = [int(r.movieId) for r in moviesforLargestCuster_80]

## Removing the dulpicate values
moviesforLargestCuster_80_set = set(moviesforLargestCuster_80_list)

# Loading the movies data
movies = spark.read.load('../Data/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
# movies = spark.read.load('/Users/pskaloya/Downloads/ml-latest-small/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()

print("Getting all the genres for the movies against the largest cluster - split 80")
genres_largestCluster_80 = movies.filter(movies['movieID'].isin(moviesforLargestCuster_80_set)).select('genres').collect()
print("Converting it a python list - genres_largestCluster_list")
genres_largestCluster_list = [r.genres for r in genres_largestCluster_80]



print("Split the pipe '|' from the genres and adding them to list")
final_genres_80 = []

for genre in genres_largestCluster_list:
    if '|' in genre:
        for x in genre.split('|'):
            final_genres_80.append(x)

    else:
        final_genres_80.append(genre)


top5genres_80 = Counter(final_genres_80).most_common(5)

genres_list = [name for (name, value) in top5genres_80]

print("Top 5 genres for 80-20 split ", genres_list)





print("Fetching for test data")
print("Getting the movies id for all the users from largest cluster - Split 50")
moviesforLargestCuster_80_test = test.filter(test['userID'].isin(largestClusterUsers_80_list)).filter(ratings['rating']>=4).select('movieId').collect()
print("Converting it a python list - moviesforLargestCuster_50_list")
moviesforLargestCuster_80_test_list = [int(r.movieId) for r in moviesforLargestCuster_80_test]

## Removing the dulpicate values
moviesforLargestCuster_80_test_set = set(moviesforLargestCuster_80_test_list)

print("Getting all the genres for the movies against the largest cluster - split 50")
genres_largestCluster_test_80 = movies.filter(movies['movieID'].isin(moviesforLargestCuster_80_test_set)).select('genres').collect()
print("Converting it a python list - genres_largestCluster_list")
genres_largestCluster_test_list = [r.genres for r in genres_largestCluster_test_80]



print("Split the pipe '|' from the genres and adding them to list")
final_genres_test_80 = []

for genre in genres_largestCluster_test_list:
    if '|' in genre:
        for x in genre.split('|'):
            final_genres_test_80.append(x)

    else:
        final_genres_test_80.append(genre)


top5genres_test_80 = Counter(final_genres_test_80).most_common(5)

genres_test_list = [name for (name, value) in top5genres_test_80]

print("Top 5 genres for 50-50 split for train data ", genres_test_list)



stop = time.time() - start
print("Time take",stop)

print("#####################################################")
print("Clearing cache ")
print("#####################################################")
spark.catalog.clearCache()

spark.stop()
