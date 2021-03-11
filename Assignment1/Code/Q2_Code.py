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

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("COM6021_Assignment1_Question_2") \
    .getOrCreate()


# spark = SparkSession.builder \
#     .master("local[*]") \
#     .appName("COM6021_Assignment1_Question_2") \
#     .config("spark.local.dir","/fastdata/lip20ps") \
#     .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Loading the Rating data
ratings = spark.read.load('/data/lip20ps/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true")
#ratings = spark.read.load('/Users/pskaloya/Downloads/ml-latest-small/ratings.csv', format = 'csv', inferSchema = "true", header = "true")


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

clusterSizes_50 = summary_50.clusterSizes

print("Clusters sizes are "+str(clusterSizes_50))


# Get the list of index from cluster sizes, where last element of the list will be index of largest cluster
indexes_50 = sorted(range(len(clusterSizes_50)), key=lambda i:clusterSizes_50[i])

predictions_50 = summary_50.predictions

largestClusterUsers = predictions_50.filter(predictions_50['prediction']==indexes_50[-1]).select('id').collect()

largestClusterUsers_list = [int(row.id) for row in largestClusterUsers]

largestClusterUsers_4plus_filtered = ratings.filter(ratings['userID'].isin(largestClusterUsers_list)).filter(ratings['rating']>=4)

moviesforLargestCuster = largestClusterUsers_4plus_filtered.select('movieId').collect()

moviesforLargestCuster_list = [int(r.movieId) for r in moviesforLargestCuster]

# Loading the movies data
movies = spark.read.load('/data/lip20ps/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true")
#movies = spark.read.load('/Users/pskaloya/Downloads/ml-latest-small/movies.csv', format = 'csv', inferSchema = "true", header = "true")


selectedMoviesID_largestCluster = movies.filter(movies['movieID'].isin(moviesforLargestCuster_list))

genres_largestCluster = selectedMoviesID_largestCluster.select('genres').collect()

genres_largestCluster_list = [r.genres for r in genres_largestCluster]

final_genres = []

for genre in genres_largestCluster_list:
    if '|' in genre:
        for x in genre.split('|'):
            final_genres.append(x)

    else:
        final_genres.append(genre)



top3genres = Counter(final_genres).most_common(3)

genres_list = [name for (name, value) in top3genres]

print("Top 3 genres in ",genres_list)

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