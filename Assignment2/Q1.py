

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json



spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Assignment2_Question1-part1") \
        .config("spark.local.dir","/fastdata/lip20ps") \
        .getOrCreate()


sc = spark.sparkContext
sc.setLogLevel("WARN")


data = spark.read.csv('./Data/HIGGS.csv.gz')


from pyspark.sql.types import DoubleType

# Renaming the label coulmn
data = data.withColumnRenamed('_c0','label')



number_of_columns = len(data.columns)

column_names = data.schema.names

# Casting the 
for i in range(number_of_columns):
    data = data.withColumn(column_names[i], data[column_names[i]].cast(DoubleType()))



data.printSchema()

sample_data = data.sample(False,0.01,seed=42)

# split training data and test data 
(trainingData, testData) = sample_data.randomSplit([0.7, 0.3], 42)



########################################################################################################################
########################################################################################################################

# Concatenating all the features in a vector
assembler = VectorAssembler(inputCols = column_names[1:number_of_columns], outputCol = 'features')

# Creating an instance for Random forest classifier
rf = RandomForestClassifier(labelCol="label", featuresCol=assembler.getOutputCol())


rf_pipeline = Pipeline(stages=[assembler,rf])



# rfc_paramGrid = ParamGridBuilder().addGrid(rfc.maxDepth,[5,6,7]).addGrid(rf.impurity, ['entropy', 'gini']).addGrid(rf.maxBins, [32,16,48]).build()

rf_paramGrid = ParamGridBuilder().addGrid(rf.maxDepth,[1, 5, 10]).addGrid(rf.impurity, ['entropy', 'gini']).addGrid(rf.maxBins, [2, 10, 20]).build()



AUC_evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")



rf_paramGrid = CrossValidator(estimator=rf_pipeline,
                              estimatorParamMaps=rf_paramGrid,
                              evaluator=accuracy_evaluator,
                              numFolds=3)




cvModel_rf = rf_paramGrid.fit(trainingData)   

BestPipeline_rf = cvModel_rf.bestModel

paramDict = {param[0].name: param[1] for param in BestPipeline_rf.stages[-1].extractParamMap().items()}


# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like

print(json.dumps(paramDict, indent = 4))




predction_rfc = cvModel_rf.transform(testData)

AUC_rfc = AUC_evaluator.evaluate(predction_rfc)
print("area under curve of rfc:",AUC_rfc)


















