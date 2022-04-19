""" The decision tree prediction in this script differs from that in decision_tree_prediction_simple.py
    at the feature engineering phase, where here we assume some features as categorical.

    Note: Replace with the path to the data in your setting (line 30)
    
"""

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


from hyperopt import hp, fmin, SparkTrials, tpe, STATUS_OK, space_eval
import stage
import hyperparameter_tuning
import shutil
import os

data_hdfs = 'hdfs://localhost:9000/user/data/covtype.data'

stage_to_production_dom = True

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

dfp = df.toPandas()
labels = dfp['Cover_Type']
train, other = train_test_split(dfp, train_size = 0.8, random_state = 42, stratify = labels)
dev, test = train_test_split(other, train_size = 0.8, random_state = 42,  stratify = other['Cover_Type'])

y_train = train['Cover_Type']
x_train = train.drop('Cover_Type', axis = 1)

y_dev = dev['Cover_Type']
x_dev = dev.drop('Cover_Type', axis = 1)

y_test = test['Cover_Type']
x_test = test.drop('Cover_Type', axis = 1)

x_train_br = sc.broadcast(x_train)
y_train_br = sc.broadcast(y_train)
x_dev_br = sc.broadcast(x_dev)
y_dev_br = sc.broadcast(y_dev)
x_test_br = sc.broadcast(x_test)
y_test_br = sc.broadcast(y_test)

data_br = (x_train_br, y_train_br, x_dev_br, y_dev_br)

classifier = DecisionTreeClassifier
search_space = {"max_depth" : hp.quniform('max_depth', 1, 20,1),"splitter" : hp.choice('splitter',['best','random']),"min_impurity_decrease": hp.quniform('min_impurity_decrease',0.0,0.1, 0.01),'criterion': hp.choice('criterion', ['gini','entropy'])}
params = search_space

best_params = hyperparameter_tuning(params, classifier, data_br, search_space, max_evals =1000)

x_train_br.unpersist()
y_train_br.unpersist()
x_dev_br.unpersist()
y_dev_br.unpersist()

print(best_params)

model_name = None
model_run_id = None
#Train the model with best hyperparameters.
with mlflow.start_run(run_name = 'best_'+type(classifier).__name__) as run:
    run_id = run.info.run_id
    model_name = run_name
    #run_name = run.data.tags['mlflow.runName']
    model_run_id = run_id
    if 'max_depth' in best_params: best_params['min_child_weight']=int(best_params['max_depth'])
    if 'min_samples_split' in params: best_params['min_samples_split']=int(best_params['min_samples_split'])
        
    mlflow.log_params(best_params)
    clf = classifier(**best_params)
    model = clf.fit(x_train, y_train)
            
    mlflow.sklearn.log_model(model, 'model')    #   persist best model to mlflow
    # Testing the model
    preds = model.predict(x_test)
    print('Best Model Predictions: ')
    print('------------------------\n')
    print(preds)
    
    f1_score = f1_score(y_test, preds, labels = range(1,8), average = 'macro')
    
    mlflow.log_metric('f1_score_in_evaluation',f1_score)
    print('Model logged under run_id "{0}" with F1 score of {1:.5f} in testing'.format(run_id, f1_score))    
    


if stage_to_production_dom:
    # The right code for this, with mlflow integration, exists in 'stage_to_production.py'
    #stage.stage_to_production(model_name)
    # Here, is a simplistic approach, simulating the transfer of artifacts to production, by copying the model files
    # to another domain under working directory called PRODUCTION_DOMAIN.
    source_dir = f"mlruns/0/{model_run_id}/artifacts/model"
    dest_dir = f'PRODUCTION_DOMAIN'+f'/{model_run_id}'
    if os.path.exists(dest_dir) == False :
        os.mkdir(dest_dir)
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        shutil.copy(source_dir+f'/{filename}',dest_dir)
    
    
    
    
    