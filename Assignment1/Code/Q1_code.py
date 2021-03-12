from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
from pyspark.sql import types as t
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt



spark = SparkSession.builder \
    .master("local[*]") \
    .appName("COM6021_Question_1") \
    .config("spark.local.dir","/fastdata/lip20ps") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.


###########################################################################################################
################################## Loading the data file ##################################################
###########################################################################################################

logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz")

data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)).drop("value").cache()


#########################################################################################################
#################################   Question 1.A   ######################################################
#########################################################################################################

host_data = data.select('host').cache()

hosts_Japan_uni = host_data.filter(host_data.host.endswith(".ac.jp")).cache()
hosts_UK_uni = host_data.filter(host_data.host.endswith(".ac.uk")).cache()
hosts_US_uni = host_data.filter(host_data.host.endswith(".edu")).cache()

hosts_Japan_uni_count = hosts_Japan_uni.count()
hosts_UK_uni_count = hosts_UK_uni.count()
hosts_US_uni_count = hosts_US_uni.count()

print("\n\nThere are %i hosts from Japan Universities.\n\n" % (hosts_Japan_uni_count))
print("\n\nThere are %i hosts from UK Universities.\n\n" % (hosts_UK_uni_count))
print("\n\nThere are %i hosts from US Universities.\n\n" % (hosts_US_uni_count))


plt.figure(figsize=(8,6))
plt.xlabel('Countries')
plt.ylabel('Count')
plt.bar(["Japan","UK","US"],[hosts_Japan_uni_count,hosts_UK_uni_count,hosts_US_uni_count])
plt.savefig("../Output/Q1_bar_chart.png")

#Clearning the figure for next figure
plt.clf()

print("Finish Question 1 Part A")


####################################################################################################
################################## Question 1 Part B ###############################################
####################################################################################################


def extract_website(s):
    s1 = s.split(".")
    return s[s.index(s1[-3]):]


def extract_website_US(s):
    s1 = s.split(".")
    return s[s.index(s1[-2]):]

extract_website_udf = udf(extract_website, t.StringType())

extract_website_US_udf = udf(extract_website_US, t.StringType())

####################################################################
################ Calculations for Japan ############################
####################################################################

hosts_Japan_uni_count = hosts_Japan_uni.withColumn('host',extract_website_udf('host')).groupBy("host").count().orderBy('count',ascending=False).cache()

hosts_Japan_uni_extracted = hosts_Japan_uni_count.limit(9)


## Get name of the top 9 Universities
hostnames_Japan_top9 = hosts_Japan_uni_extracted.select('host').collect()
labels_Japan = [str(row['host']) for row in hostnames_Japan_top9]

## Get count of the top 9 Universities
hostnames_Japan_count_top9 = hosts_Japan_uni_extracted.select('count').collect()
Top_Japan_Values = [int(row['count']) for row in hostnames_Japan_count_top9]

## Adding value of the all the count
Total_japan_hits = hosts_Japan_uni_count.agg({'count':'sum'}).collect()

## Getting the value for rest of the Universities
rest_jap_uni = Total_japan_hits[0][0] - (sum(Top_Japan_Values))

## Adding values for Rest
Top_Japan_Values.append(rest_jap_uni)
labels_Japan.append('Rest')

print("PLotting pie chart for Japan")
plt.figure(figsize=(15,12))
plt.title("Pie Chart for Japan University Hosts")
plt.pie(x=Top_Japan_Values, labels=labels_Japan, radius=2, textprops = {'fontsize':10, 'color':'black'}, autopct = '%3.2f%%')
plt.tight_layout()

plt.savefig("../Output/Q1_pie_chart_Japan.png")

#Clearning the figure for next figure
plt.clf()

####################################################################
################ Calculations for UK ############################
####################################################################

hosts_UK_uni_count = hosts_UK_uni.withColumn('host',extract_website_udf('host')).groupBy("host").count().orderBy('count',ascending=False).cache()

hosts_UK_uni_extracted = hosts_UK_uni_count.limit(9)

## Get names of the top 9 Universities
hostnames_UK_top9 = hosts_UK_uni_extracted.select('host').collect()
labels_UK = [str(row['host']) for row in hostnames_UK_top9]

## Get count of the top 9 Universities
hostnames_UK_count_top9 = hosts_UK_uni_extracted.select('count').collect()
Top_UK_values = [int(row['count']) for row in hostnames_UK_count_top9]

## Adding value of the all the count
Total_UK_hits = hosts_UK_uni_count.agg({'count':'sum'}).collect()

## Getting the value for rest of the Universities
rest_UK_uni = Total_UK_hits[0][0] - (sum(Top_UK_values))

## Adding values for Rest
Top_UK_values.append(rest_UK_uni)
labels_UK.append('Rest')

print("Ploting Pie Chart for UK")
plt.figure(figsize=(15,12))
plt.title("Pie Chart for UK University Hosts")
plt.pie(x=Top_UK_values, labels=labels_UK, radius=2, textprops = {'fontsize':10, 'color':'black'}, autopct = '%3.2f%%')
plt.tight_layout()
plt.savefig("../Output/Q1_pie_chart_UK.png")

#Clearning the figure for next figure
plt.clf()


####################################################################
################ Calculations for USA ############################
####################################################################

hosts_US_uni_count = hosts_US_uni.withColumn('host',extract_website_US_udf('host')).groupBy("host").count().orderBy('count',ascending=False).cache()

hosts_US_uni_extracted = hosts_US_uni_count.limit(9)

## Get names of the top 9 Universities
hostnames_US_top9 = hosts_US_uni_extracted.select('host').collect()
labels_US = [str(row['host']) for row in hostnames_US_top9]

## Get count of the top 9 Universities
hostnames_US_count_top9 = hosts_US_uni_extracted.select('count').collect()
Top_US_values = [int(row['count']) for row in hostnames_US_count_top9]

## Adding value of the all the count
Total_US_hits = hosts_US_uni_count.agg({'count':'sum'}).collect()

## Getting the value for rest of the Universities
rest_US_uni = Total_US_hits[0][0] - (sum(Top_US_values))

## Adding values for Rest
Top_US_values.append(rest_US_uni)
labels_US.append('Rest')

print("Ploting Pie Chart for US")
plt.figure(figsize=(15,12))
plt.title("Pie Chart for US University Hosts")
plt.pie(x=Total_US_hits, labels=labels_US, radius=2, textprops = {'fontsize':10, 'color':'black'}, autopct = '%3.2f%%')
plt.tight_layout()
plt.savefig("../Output/Q1_pie_chart_US.png")

#Clearning the figure for next figure
plt.clf()


####################################################################################################
################################## Question 1 Part C ###############################################
####################################################################################################

print("Fetching data for Question 1 Part C")

MostFrequest_Japan = hosts_Japan_uni_count.limit(1).collect()[0].host
MostFrequest_UK = hosts_UK_uni_count.limit(1).collect()[0].host
MostFrequest_US = hosts_US_uni_count.limit(1).collect()[0].host

MostFrequest_Japan_data = data.filter(data.host.contains(MostFrequest_Japan))
MostFrequest_UK_data = data.filter(data.host.contains(MostFrequest_UK))
MostFrequest_US_data = data.filter(data.host.contains(MostFrequest_US))



Japan_date_hour = MostFrequest_Japan_data.withColumn('date',F.regexp_extract('timestamp', '(^\d+).*:\d+:\d+:',1)) \
        .withColumn('hour',F.regexp_extract('timestamp', '.*:(\d+):\d+:',1)).drop('host','timestamp').cache()
Japan_date_hour_count = Japan_date_hour.groupBy('date','hour').count()
Japan_heatmap_DF = pd.pivot_table(Japan_date_hour_count, values='count',index='hour', columns='date')


UK_date_hour = MostFrequest_UK_data.withColumn('date',F.regexp_extract('timestamp', '(^\d+).*:\d+:\d+:',1)) \
        .withColumn('hour',F.regexp_extract('timestamp', '.*:(\d+):\d+:',1)).drop('host','timestamp').cache()
UK_date_hour_count = UK_date_hour.groupBy('date','hour').count()
UK_heatmap_DF = pd.pivot_table(UK_date_hour_count, values='count',index='hour', columns='date')


US_date_hour = MostFrequest_US_data.withColumn('date',F.regexp_extract('timestamp', '(^\d+).*:\d+:\d+:',1)) \
        .withColumn('hour',F.regexp_extract('timestamp', '.*:(\d+):\d+:',1)).drop('host','timestamp').cache()
US_date_hour_count = US_date_hour.groupBy('date','hour').count()
US_heatmap_DF = pd.pivot_table(US_date_hour_count, values='count',index='hour', columns='date')


## Ploting heatmap for Japan
print("Ploting heatmap for Japan")
plt.figure(figsize=(12,10))
plt.imshow(Japan_heatmap_DF, cmap='hot')
plt.xticks(np.arange(max(Japan_heatmap_DF['date'])))
plt.yticks(np.arange(max(Japan_heatmap_DF['hour'])))
plt.xlabel("Date", fontsize=13)
plt.ylabel("Hour", fontsize=13)
plt.title("Heatmaps for Top University in Japan", fontsize=18)
plt.savefig("../Output/Q1_HeatMap_Japan.png")

#Clearning the figure for next figure
plt.clf()


## Ploting heatmap for UK
print("Ploting heatmap for UK")
plt.figure(figsize=(12,10))
plt.imshow(UK_heatmap_DF, cmap='hot')
plt.xticks(np.arange(max(UK_heatmap_DF['date'])))
plt.yticks(np.arange(max(UK_heatmap_DF['hour'])))
plt.xlabel("Date", fontsize=13)
plt.ylabel("Hour", fontsize=13)
plt.title("Heatmaps for Top University in UK", fontsize=18)
plt.savefig("../Output/Q1_HeatMap_UK.png")

#Clearning the figure for next figure
plt.clf()

## Ploting heatmap for UK
print("Ploting heatmap for UK")
plt.figure(figsize=(12,10))
plt.imshow(US_heatmap_DF, cmap='hot')
plt.xticks(np.arange(max(US_heatmap_DF['date'])))
plt.yticks(np.arange(max(US_heatmap_DF['hour'])))
plt.xlabel("Date", fontsize=13)
plt.ylabel("Hour", fontsize=13)
plt.title("Heatmaps for Top University in UK", fontsize=18)
plt.savefig("../Output/Q1_HeatMap_US.png")

#Clearning the figure for next figure
plt.clf()

print("Finished")

spark.stop()
