# The decision tree prediction in this script differs from that in decision_tree_prediction_simple.py
# at the feature engineering phase, where here we assume some features as categorical.
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType
import numpy as np


data_hdfs = 'hdfs://localhost:9000/user/data/covtype.data'

if sc.getConf().get('spark.executor.memory') != '4g':
    # Dynamically configure spark.executor.memory of the cluster. Amount of memory to use per executor process.
    SparkContext.setSystemProperty('spark.executor.memory','6g')
    SparkContext.setSystemProperty('spark.driver.memory','6g')
    
# it is a good practice, if the desired schema is known beforehand, to pass it in the read command with the schema() method for
# double reading the dataset
df = spark.read.option('inferSchema','true').option("header", False).csv(data_hdfs)

# assign header to the dataset
# The column names are given in the companion file, covtype.info.
col_names = ["Elevation", "Aspect", "Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm","Horizontal_Distance_To_Fire_Points"]+[f"Wilderness_Area_{i}" for i in range(1,5)]+[f"Soil_Type_{i}" for i in range(1,41)]+["Cover_Type"]
df = df.rdd.toDF(col_names)
# This is the target column and serves well to cast it to double as many methods in the ML and the MLLib api will consume double type as their input.
df = df.withColumn('Cover_Type',df.Cover_Type.cast('double'))


from pyspark.ml.feature import VectorAssembler

vaw = VectorAssembler().setInputCols([f"Wilderness_Area_{i}" for i in range(1,5)]).setOutputCol("wilderness")

vas = VectorAssembler().setInputCols([f"Soil_Type_{i}" for i in range(1,41)]).setOutputCol("soil")

df1 = vas.transform(vaw.transform(df).drop(*[f"Wilderness_Area_{i}" for i in range(1,5)])).drop(*[f"Soil_Type_{i}" for i in range(1,41)])   
index_of_udf = F.udf(lambda c : float(c.toArray().tolist().index(1)), DoubleType())
df1 = df1.withColumn('wilderness',index_of_udf(df1.wilderness)).withColumn('soil',index_of_udf(df1.soil))

#For splitting the dataset in train and test dayasets we may either pick a randomSplit
# or better choose a stratified approach in dataset splitting.

# 1st approach

cov_type_values = df1.groupBy('Cover_Type').count().select('Cover_Type').collect()
for i, r in enumerate(cov_type_values):
    v = r['Cover_Type']
    if i == 0:
        training2, test2 = df1.where(f'Cover_Type = {v}').randomSplit([0.9, 0.1], seed = 42)
    else: 
        train_temp, test_temp = df1.where(f'Cover_Type = {v}').randomSplit([0.9, 0.1], seed = 42)
        training2 = training2.union(train_temp)
        test2 = test2.union(test_temp)

# 2nd approach
#training,test = df1.randomSplit([0.9, 0.1])      

training2 = training2.cache()
test2 = test2.cache()

new_columns = list(set(df1.columns).difference(set([f"Wilderness_Area_{i}" for i in range(1,5)]+ [f"Soil_Type_{i}" for i in range(1,41)])))   
cnames = [col for col in new_columns if col != 'Cover_Type']
va1 = VectorAssembler().setInputCols(cnames).setOutputCol("features")  

# we are going to use pyspark's Vector Indexer transformer for the representation of some categorical features
from pyspark.ml.feature import VectorIndexer
																																		
    
vi = VectorIndexer().setMaxCategories(40).setInputCol("features").setOutputCol("indexed").setHandleInvalid('skip')

# Classifier's definition. The training of the model will be made from the pipeline defined below.
from pyspark.ml.classification import DecisionTreeClassifier 
dtc1 = DecisionTreeClassifier().setFeaturesCol("indexed").setLabelCol("Cover_Type").setPredictionCol("prediction").setSeed(42)
 
# the evaluation will be done using MulticlassClassificationEvaluator
# metricName = 'f1' by default 
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(predictionCol = 'prediction', labelCol = 'Cover_Type')   
 
from pyspark.ml import Pipeline

pipe2 = Pipeline().setStages([va1,vi,dtc1])

from pyspark.ml.tuning import ParamGridBuilder   
# With ParamGridBuilder we will create a grid of hyperparameter values, ranging between edge values that we will specify and 
# that will be tried in hyperparameter tuning.                              
parameter_grid = ParamGridBuilder().addGrid(dtc1.maxDepth,[1,20]).addGrid(dtc1.maxBins,[40,300]).addGrid(dtc1.impurity,['gini','entropy']).addGrid(dtc1.minInfoGain, [0,0.05]).build()
				       							   	   
print('Train and validate with hyperparameter tuning')
print('---------------------')                              
# Train and validate the model with hyperparameter tuning for picking the best model using TrainValidationSplit.                                                                     
from pyspark.ml.tuning import TrainValidationSplit
# CrossValidator could be used instead of TrainAndValidationSplit
tvs2 = TrainValidationSplit(estimator = pipe2, evaluator = evaluator, estimatorParamMaps = parameter_grid, parallelism = 1, seed = 42).setTrainRatio(0.9)

tvs_model2 = tvs2.fit(training2)

tvs_model2.validationMetrics         

best_model2 = tvs_model2.bestModel

best_predictions2 = best_model2.transform(test2)
print('Best Model')
print('------------------')
best_predictions2.show()
f1_score_best2 = evaluator.evaluate(best_predictions2)           
print(f'F1 score for the best model (using categorical features) {f1_score_best2}')