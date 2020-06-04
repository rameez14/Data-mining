from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[10]").config("spark.local.dir","/fastdata/acp18rkh").appName("Q1_1").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")


rawdata = spark.read.option('header', "false").csv('Data/HIGGS.csv.gz')
rawdata.cache()

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
from pyspark.ml.feature import Binarizer


schemaNames = rawdata.schema.names
ncolumns = len(rawdata.columns)
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata = rawdata.withColumn(schemaNames[i], rawdata[schemaNames[i]].cast(DoubleType()))
rawdata = rawdata.withColumnRenamed('_c0', 'labels')

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns], outputCol = 'features') 
raw_plus_vector = assembler.transform(rawdata)

data = raw_plus_vector.select('features','labels')

(small_dataset, _) = data.randomSplit([0.05, 0.95], 42)

(trainingData, testData) = small_dataset.randomSplit([0.7, 0.3], 42)
trainingData = trainingData.cache()
testData = testData.cache()

def running_metric(pipeline, paramGrid, evaluator):

  cvalidation = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator= evaluator,
                            numFolds=5) 
  
  return cvalidation

evaluator_bin = BinaryClassificationEvaluator(labelCol="labels", rawPredictionCol="prediction", metricName="areaUnderROC") 
evaluator_multi= MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")


#Random Forest
rfr = RandomForestClassifier(labelCol="labels", featuresCol="features", predictionCol='prediction_c')
bina = Binarizer(threshold = 0.5, inputCol="prediction_c", outputCol="prediction")
paramGrid = ParamGridBuilder() \
    .addGrid(rfr.maxDepth, [5,10,15]) \
    .addGrid(rfr.numTrees, [3,10,20]) \
    .build()
pipeline = Pipeline(stages=[rfr, bina])

print("Random Forest Classifier, best parameters: Area Under ROC")
cvalidation = running_metric(pipeline,paramGrid,evaluator_bin)
crossvalidation_Model = cvalidation.fit(trainingData)
print(crossvalidation_Model.getEstimatorParamMaps()[ np.argmax(crossvalidation_Model.avgMetrics) ])
print("Area Under ROC:",np.max(crossvalidation_Model.avgMetrics))

print("Random Forest Classifier, best parameters Metric: Accuracy")
cvalidation = running_metric(pipeline,paramGrid,evaluator_multi)
crossvalidation_Model = cvalidation.fit(trainingData)
print(crossvalidation_Model.getEstimatorParamMaps()[ np.argmax(crossvalidation_Model.avgMetrics) ])
print("Accuracy:",np.max(crossvalidation_Model.avgMetrics))

#Gradient Boosting

gbr = GBTClassifier(labelCol="labels", featuresCol="features", predictionCol='prediction_c')
paramGrid = ParamGridBuilder() \
    .addGrid(gbr.maxDepth, [5, 10,14]) \
    .addGrid(gbr.maxIter, [5, 10,15]) \
    .build()
pipeline = Pipeline(stages=[gbr, bina])

print("Gradient Boosting Classifier, best Parameters: Area Under ROC")
cvalidation = running_metric(pipeline,paramGrid,evaluator_bin)
crossvalidation_Model = cvalidation.fit(trainingData)
print(crossvalidation_Model.getEstimatorParamMaps()[ np.argmax(crossvalidation_Model.avgMetrics) ])
print("Area Under ROC:",np.max(crossvalidation_Model.avgMetrics))

print("Gradient Boosting Classifier, best Parameters Metric: Accuracy")
cvalidation = running_metric(pipeline,paramGrid,evaluator_multi)
crossvalidation_Model = cvalidation.fit(trainingData)
print(crossvalidation_Model.getEstimatorParamMaps()[ np.argmax(crossvalidation_Model.avgMetrics) ])
print("Accuracy:",np.max(crossvalidation_Model.avgMetrics))



spark.stop()

