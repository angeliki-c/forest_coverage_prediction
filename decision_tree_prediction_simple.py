from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.types import StructType, DoubleType

import numpy as np

data_hdfs = 'hdfs://localhost:9000/user/data/covtype.data'

verbose = False
if sc.getConf().get('spark.executor.memory') != '4g':
    # Dynamically configure spark.executor.memory of the cluster. Amount of memory to use per executor process.
    SparkContext.setSystemProperty('spark.executor.memory','4g')
    SparkContext.setSystemProperty('spark.driver.memory','6g')

# it is a good practice, if the desired schema is known beforehand, to pass it in the read command with the schema() method for
# double reading the dataset  
df = spark.read.option('inferSchema','true').option("header", False).csv(data_hdfs)
 

# assign header to the dataset
# The column names are given in the companion file, covtype.info.
col_names = ["Elevation", "Aspect", "Slope","Horizontal_Distance_To_Hydrology",                          
 "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",							 
 "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm","Horizontal_Distance_To_Fire_Points"]+[f"Wilderness_Area_{i}" for i in range(1,5)]+[f"Soil_Type_{i}" for i in range(1,41)]+["Cover_Type"]    
 
df = df.rdd.toDF(col_names)	
# This is the target column and serves well to cast it to double as many methods in the ML and 
# the MLLib api will consume double type as their input.
df = df.withColumn('Cover_Type',df.Cover_Type.cast('double')) 		
																									
# Data exploration
print('Data exploration')
df.show(5)
rows = df.count()
columns = len(df.columns)
if verbose:
    print('Data Summary')
    print('----------------------')
    print(f'Size : {rows}x{columns}') 
    for i in range(0,len(col_names),5):
        df.describe().select(['summary']+col_names[i:min(i+5,len(col_names))]).show()


#For splitting the dataset in train and test dayasets we may either pick a randomSplit
# or better choose a stratified approach in dataset splitting.

# 1st approach

cov_type_values = df.groupBy('Cover_Type').count().select('Cover_Type').collect()
for i, r in enumerate(cov_type_values):
    v = r['Cover_Type']
    if i == 0:
        training, test = df.where(f'Cover_Type = {v}').randomSplit([0.9, 0.1], seed = 42)
    else: 
        train_temp, test_temp = df.where(f'Cover_Type = {v}').randomSplit([0.9, 0.1], seed = 42)
        training = training.union(train_temp)
        test = test.union(test_temp)

# 2nd approach
#training,test = df.randomSplit([0.9, 0.1])      

training = training.cache()
test = test.cache()

# Feature engineering
# Treating categorical features as seperate numerical features, taking 0/1 value, expanding in the numerical category 
# of columns, increases the processing effort, needs additional decission rules for expressing the relation that holds
# between this (a categorical feature ) and the other features, as well as the variability of this feature.
# Representing categorical features as one hot vectors in this use case, we were able to see accuracy improving even by 2%.

# prepare the data in the format that the decision tree algorithm provided in the pyspark ML library accepts

va = VectorAssembler(inputCols = col_names[:-1], outputCol = 'features')
td = va.transform(training).select('features','Cover_Type').cache()
ttd = va.transform(test).select('features','Cover_Type').cache()

dtc = DecisionTreeClassifier(maxDepth = 2,labelCol = 'Cover_Type', leafCol = 'leafId').setSeed(42)

#Training
print('Training')   
print('----------------------')       
                                                                                               
# a decision tree classification model of type DecisionTreeClassificationModel
dt = dtc.fit(td)                   

# show the description details of the model trained
if verbose:
    print(f'Model description  : {dt.toDebugString}')      

# estimate each features's importance based on a generalization of the idea of the Gini importance
# Importances for the tree have been normalized to sum to 1.
if verbose:
    print(f'Feature Importances : {dt.featureImportances}')                                                                               

    zipped = zip(dt.featureImportances.toArray().tolist(),col_names[:-1])

    sorted_zipped = sorted(zipped,reverse = True)
    print(f'{sorted_zipped}')

predictions = dt.transform(ttd)

if verbose:
    print('Predictions')
    print('----------------------') 
    predictions.show()


# evaluation using MulticlassClassificationEvaluator
print('Evaluation')
print('----------------------')

# metricName = 'f1' by default 
evaluator = MulticlassClassificationEvaluator(predictionCol = 'prediction', labelCol = 'Cover_Type')    
f1_score = evaluator.evaluate(predictions)
print(f'F1 Score : {f1_score}')           
  
# other metric names that can be used are accuracy, logLoss, hummingLoss etc
# evaluating against the true positive rate
evaluator.setMetricName = "truePositiveRateByLabel"        
#evaluator.setMetricLabel(1)
tpr = evaluator.evaluate(predictions)     
print(f'True positive rate by label : {tpr}')                           
     
   
print('Confusion Matrix')
print('----------------------')
																													   
# Create the confusion matrix. For this we are going to use MulticlassMetrics from the pyspark MLLib, which is rdd based. 
predictionAndLabelRDD = predictions.select(['prediction','Cover_Type']).rdd
#predictionAndLabelsWithProbabilityRDD = predictions.select(['prediction','Cover_Type','probability']).rdd.toDF(['prediction','label','probability']).rdd
metrics = MulticlassMetrics(predictionAndLabelRDD)
#metrics2 = MulticlassMetrics(predictionAndLabelsWithProbabilityRDD)
metrics.confusionMatrix()
metrics.confusionMatrix().toArray()
confusion_matrix = predictionAndLabelRDD.toDF(['prediction','label']).groupBy('label').pivot('prediction',range(1,8)).count().fillna(0).orderBy('label')   

avg_precision = 0
avg_recall = 0
for i in range(1,8):
    avg_precision = avg_precision + metrics.precision(i)
    avg_recall = avg_recall + metrics.recall(i)
    
avg_precision = avg_precision / 7
avg_recall = avg_recall / 7
print(f'Confusion Matrix : {confusion_matrix.show()}')
print(f'Average recision for all classes  : {avg_precision}     Average recall for all classes : {avg_recall} ')    



#compare with baseline                                                                            
# as a baseline we assume the model that assigns predictions based on the probability of each class in the data set 
# (computed from the statistical mean of each class in the dataset)                                                                                                         
class_probs = df.groupBy('Cover_Type').count().withColumn('fraction', F.col('count')/rows).select('fraction').collect()
class_probs = [r['fraction'] for r in class_probs]
baseline_model_predictions = np.random.choice([1,2,3,4,5,6,7],size = test.count(),p = class_probs, replace = True)
baseline_model_predictions_list = [float(el) for el in baseline_model_predictions]
label_list = [ r['Cover_Type'] for r in ttd.select('Cover_Type').collect()]

base_predictions = spark.createDataFrame(zip(baseline_model_predictions_list,label_list), StructType().add('prediction',DoubleType()).add('Cover_Type',DoubleType()))
#base_metrics = MulticlassMetrics(base_predictions.rdd)
evaluator.setMetricName = 'f1'
f1_score_base = evaluator.evaluate(base_predictions)
print(f'F1 Score for the baseline  : {f1_score_base}') 



# Selecting the best model with hyperparameter tuning
print('Hyperparameter tuning')
print('----------------------')

print("The hyperparameters of the decision tree that we are going to tune are : maxDepth, maxBins, impurity measure, minimum information gain")

# maxDepth should be selected appropriately because greater values may lead to overfitting

# as maxBins increases the decision tree algorithm may lead to a more optimal decission rule, whereas it increases the processing time of the algorithm

################################a decision rule is good only when it leads to a seperation of the dataset where the different parts have different distribution of the target values, minimizing the impurity of the sub-datasets it induces
#We are considering Gini impurity and entropy for the calculation of impurity.
						  
						  
                   
				   
# minimum information gain     :    Rules that do not improve the subsets' impurity enough are rejected. This parameter also contributes in avoiding overfitting.
									
							
									
from pyspark.ml import Pipeline

pipe = Pipeline().setStages([va,dtc])   

from pyspark.ml.tuning import ParamGridBuilder   
# With ParamGridBuilder we will create a grid of hyperparameter values, ranging between edge values that we will specify and 
# that will be tried in hyperparameter tuning.                              
parameter_grid = ParamGridBuilder().addGrid(dt.maxDepth,[1,20]).addGrid(dt.maxBins,[40,300]).addGrid(dt.impurity,['gini','entropy']).addGrid(dt.minInfoGain, [0,0.05]).build()
				       							   	   
from pyspark.ml.tuning import TrainValidationSplit
# CrossValidator could be used instead of TrainAndValidationSplit
tvs = TrainValidationSplit(estimator = pipe, estimatorParamMaps = parameter_grid, evaluator = evaluator,parallelism=1, seed=42).setTrainRatio(0.9)

tvs_model = tvs.fit(training)
tvs_model.getTrainRatio()  
# Metrics on the validation data set for the 16 models formed   
tvs_model.validationMetrics        

best_model = tvs_model.bestModel

best_predictions = best_model.transform(test)

best_f1 = evaluator.evaluate(best_predictions)

print(f'Best Model')
print('---------------------\n')

best_predictions.show()
print(f'Best model F1-score : {best_f1} ')
parameters = best_model.stages[-1].extractParamMap()          #   info on the parameters of the model (hyperparameters too)

tvs_maps = tvs_model.getEstimatorParamMaps()

