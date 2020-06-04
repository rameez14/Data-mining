from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[20]").config("spark.local.dir","/fastdata/acp18rkh").appName("Q1_2").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")

rawdata = spark.read.option('header', "false").csv('Data/HIGGS.csv.gz')
rawdata.cache()

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import time 
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


(trainingData, testData) = data.randomSplit([0.7, 0.3], 42)
trainingData = trainingData.cache()
testData = testData.cache()

evaluator_bin = BinaryClassificationEvaluator(labelCol="labels", rawPredictionCol="prediction", metricName="areaUnderROC") 
evaluator_multi= MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

def best_model(algo, bin, log):
  mdl = algo.fit(trainingData)
  pred = mdl.transform(testData)
  if bin:
    bina = Binarizer(threshold = 0.5, inputCol="prediction_c", outputCol="prediction")
    pred = bina.transform(pred) 
  acc = evaluator_multi.evaluate(pred)
  area_under_curve = evaluator_bin.evaluate(pred)
  print("Accuracy:", acc)
  print("Area Under ROC:", area_under_curve)
  print("Top Three Features")
  if log:
    feature_importance = mdl.coefficients.values
    for f in np.abs(feature_importance).argsort()[-3:][::-1]:
      print(schemaNames[f+1], end=" ")
    print("")

  else:
    feature_importance = mdl.featureImportances

    top_features = np.zeros(ncolumns-1)
    top_features[feature_importance.indices] = feature_importance.values
    for f in top_features.argsort()[-3:][::-1]:
        print(schemaNames[f+1], end=" ")
    print("")
  
  return best_model


bina = Binarizer(threshold = 0.5, inputCol="prediction_c", outputCol="prediction")

#Random Forest Classifer
start = time.time()
print("Random Forest Classifier")
rfr = RandomForestClassifier(labelCol="labels", featuresCol="features", predictionCol='prediction_c', maxDepth = 15, numTrees = 20)
mdl = best_model(rfr,True,False)
stop = time.time()
print("Time:", (stop-start)//60, "minutes", (stop-start)%60, "seconds")

#Gradient Boosting Classifier 
start = time.time()
print("Gradient Boosting Classifier")
gbr = GBTClassifier(labelCol="labels", featuresCol="features", predictionCol='prediction_c', maxDepth = 10, maxIter = 15)
mdl = best_model(gbr,True,False)
stop = time.time()
print("Time:", (stop-start)//60, "minutes", (stop-start)%60, "seconds")

spark.stop()