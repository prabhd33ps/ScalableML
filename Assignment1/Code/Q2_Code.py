from pyspark.sql import SparkSession
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