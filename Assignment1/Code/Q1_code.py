from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql import types as t
import matplotlib.pyplot as plt




spark = SparkSession.builder \
    .master("local[*]") \
    .appName("COM6021_Question_1") \
    .config("spark.local.dir","/fastdata/lip20ps") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

logFile = spark.read.text("Data/NASA_access_log_Jul95.gz").cache()

host_data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)).drop("value").cache()

data2 = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)).drop("value").cache()


jj_uni = jj.withColumn('date',F.regexp_extract('timestamp', '(^\d+).*:(\d+):\d+:',1)) \
    .withColumn('hour',F.regexp_extract('timestamp', '.*:(\d+):\d+:',1)).cache()


df3 = df3.withColumn("date", df3["date"].cast(IntegerType()))
df3 = df3.withColumn("hour", df3["hour"].cast(IntegerType()))

data2 = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)).drop("value").cache()

heatmap1_data = pd.pivot_table(df2, values='count',
                     index=['date'],
                     columns='hour')

sns.heatmap(heatmap1_data, cmap="YlGnBu")



####################################################################################################
################################## Question 1 Part A ###############################################
####################################################################################################

hosts_Japan_uni = host_data.filter(host_data.host.endswith(".ac.jp")).cache()
hosts_UK_uni = host_data.filter(host_data.host.endswith(".ac.uk")).cache()
hosts_US_uni = host_data.filter(host_data.host.endswith(".edu")).cache()

hosts_Japan_uni_count = hosts_Japan_uni.count()
hosts_UK_uni_count = hosts_UK_uni.count()
hosts_US_uni_count = hosts_US_uni.count()

print("\n\nThere are %i hosts from Japan Universities.\n\n" % (hosts_Japan_uni_count))
print("\n\nThere are %i hosts from UK Universities.\n\n" % (hosts_UK_uni_count))
print("\n\nThere are %i hosts from US Universities.\n\n" % (hosts_US_uni_count))


fig = plt.figure(figsize=(16,6))




ax0 = fig.add_subplot(121)
ax0.bar(["Japan","UK","US"],[hosts_Japan_uni_count,hosts_UK_uni_count,hosts_US_uni_count])
ax0.set_xlabel('Count')
ax0.set_ylabel('Countries')



####################################################################################################
################################## Question 1 Part B ###############################################
####################################################################################################


def extract_website(s):
    s1 = s.split(".")
    return s[s.index(s1[-3]):]


def extract_website_US(s):
    s1 = s.split(".")
    return s[s.index(s1[-3]):]

extract_website_udf = udf(extract_website, t.StringType())

extract_website_US_udf = udf(extract_website_US, t.StringType())

hosts_Japan_uni_count = hosts_Japan_uni.withColumn('host',extract_website_udf('host')).groupBy("host").count().orderBy('count',ascending=False).cache()

hosts_Japan_uni_extracted = hosts_Japan_uni_count.limit(9)

hostnames_Japan_top9 = hosts_Japan_uni_extracted.select('host').collect()
labels = [str(row['host']) for row in hostnames_Japan_top9]

hostnames_Japan_count_top9 = hosts_Japan_uni_extracted.select('count').collect()
x_Japan = [int(row['count']) for row in hostnames_Japan_count_top9]

Total_japan_hits = hosts_Japan_uni_count.agg({'count':'sum'}).collect()

ax1 = fig.add_subplot(122)
ax1.pie(x=x_Japan, labels=labels, radius=2, textprops = {'fontsize':10, 'color':'black'}, autopct = '%3.2f%%')
ax1.title("Pie Chart for Japan University Hosts")





plt.tight_layout()
plt.savefig('plots.png')
print("Finished")


spark.stop()