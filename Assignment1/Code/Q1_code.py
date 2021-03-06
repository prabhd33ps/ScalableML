from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql import types as t
import matplotlib.pyplot as plt


def extract_website(s):
    s1 = s.split(".")
    return s[s.index(s1[-3]):]

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6021_Question_1") \
    .config("spark.local.dir","/fastdata/lip20ps") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

logFile = spark.read.text("Data/NASA_access_log_Jul95.gz").cache()

host_data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)).drop("value").cache()


hosts_Japan_uni = host_data.filter(host_data.host.endswith(".ac.jp"))
hosts_UK_uni = host_data.filter(host_data.host.endswith(".ac.uk"))
hosts_US_uni = host_data.filter(host_data.host.endswith(".edu"))

print("\n\nThere are %i hosts from Japan Universities.\n\n" % (hosts_Japan_uni.count()))
print("\n\nThere are %i hosts from UK Universities.\n\n" % (hosts_UK_uni.count()))
print("\n\nThere are %i hosts from US Universities.\n\n" % (hosts_US_uni.count()))


fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.bar(["Japan","UK","US"],[hosts_Japan_uni.count(),hosts_UK_uni.count(),hosts_US_uni.count()])
ax.set_xlabel('Count')
ax.set_ylabel('Countries')
plt.savefig("Output/Question1_A.png")



extract_website_udf = udf(extract_website, t.StringType())

hosts_Japan_uni_extracted = hosts_Japan_uni.withColumn('host',extract_website_udf('host')).groupBy("host").count().orderBy('count',ascending=False).limit(9).cache()

hostnames_Japan_top9 = hosts_Japan_uni_extracted.select('host').collect()
hostnames_Japan_count_top9 = hosts_Japan_uni_extracted.select('count').collect()

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.pie(x=hostnames_Japan_count_top9,labels=hostnames_Japan_top9,radius=2,textprops = {'fontsize':10, 'color':'black'},autopct = '%3.2f%%')
ax.set_xlabel('Count')
ax.set_ylabel('Countries')
plt.savefig("Output/Question1_B.png")


spark.stop()