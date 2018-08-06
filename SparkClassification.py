from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier as RF
from pyspark.mllib.evaluation import MulticlassMetrics
import six
import random

sc = SparkContext()
sqlC = SQLContext(sc)
responses = sqlC.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(
    'iris.csv')
responses.show(1)
print((responses.count(), len(responses.columns)))

print(responses.describe().toPandas().transpose())

responses.printSchema()

stringIndexer = StringIndexer(inputCol="Species", outputCol="SPECIES_Catogery")
si_model = stringIndexer.fit(responses)
irisNormDf = si_model.transform(responses)

print(irisNormDf.select("Species","SPECIES_Catogery").distinct().collect())

for i in irisNormDf.columns:
    if not (isinstance(irisNormDf.select(i).take(1)[0][0], six.string_types)):
        print("Correlation to  for ", i, irisNormDf.stat.corr('SPECIES_Catogery', i))

#[Row(Species='versicolor', SPECIES_Catogery=0.0), Row(Species='setosa', SPECIES_Catogery=2.0), Row(Species='virginica', SPECIES_Catogery=1.0)]

iris_final = irisNormDf.drop('Species')
vectorAssembler = VectorAssembler(inputCols=['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],outputCol='features')
iris_final = vectorAssembler.transform(iris_final)
iris_final = iris_final.select(['features', 'SPECIES_Catogery'])
    #print(vauto_df)
iris_final.show(3)

random.seed(100)
splits = iris_final.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]

print(train_df.count())
print(test_df.count())

##############################----DECISION TREE CLASSIFICATION----########################################

dtreeeClassifer = DecisionTreeClassifier(maxDepth=2, labelCol="SPECIES_Catogery",featuresCol="features")
dtreeModel = dtreeeClassifer.fit(train_df)

predictions = dtreeModel.transform(test_df)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="SPECIES_Catogery",metricName="accuracy")
print(evaluator.evaluate(predictions))

predictions.groupBy("SPECIES_Catogery","prediction").count().show()


###########################----RANDOM FOREST CLASSIFIER----##########################################33

iris_rf = RF(labelCol='SPECIES_Catogery', featuresCol='features',numTrees=200)
fit = iris_rf.fit(train_df)
prediction_rf = fit.transform(test_df)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="SPECIES_Catogery",metricName="accuracy")
print(evaluator.evaluate(prediction_rf))



predictionAndLabels = prediction_rf.select(['prediction', 'SPECIES_Catogery'])

metrics = MulticlassMetrics(predictionAndLabels.rdd)
confusion_mat = metrics.confusionMatrix()
print(confusion_mat)
