
# Retrieve the model for forest coverage type prediction on new coming data.

# Assume that the test set consists the new coming data, in fact this may be batch data to be processed by a job or
# streaming data to be processed online by a scheduled job.

data = x_test 
model_name = 'best_ABCMeta'
import mlflow
import pandas as pd


client = mlflow.tracking.MlflowClient()

experiment_ids = [exp.experiment_id for exp in  client.list_experiments()]
model_run_id = client.search_runs(experiment_ids[0], filter_string = f"tags.`mlflow.runName` = '{model_name}'")[0].info.run_id

#classification_model = mlflow.sklearn.load_model(model_uri =  model/uri/in/production/domain)
classification_model = mlflow.sklearn.load_model(model_uri = f'Spark/projects/forest_coverage_prediction/prediction_with_mlflow_hyperopt/PRODUCTION_DOMAIN'+f'/{model_run_id}')
preds = classification_model.predict(data)

print(f"The model's results are: {preds}\n")
print(f"The model's reasoning is : {classification_model.decision_path(data)}\n")  #  non zero elements indicates that the samples goes through the nodes

# or feats_importances = pd.DataFrame(dict(zip(data.columns, classification_model.feature_importances_)),index = [0]).sort_values(by = 0, axis = 1, ascending = False).transpose()
# or just feats_importances = pd.DataFrame.from_dict(dict(zip(data.columns, classification_model.feature_importances_)), orient = 'index')
# or at leas decide once (you have not gone for shopping, remember...)
feats_importances =  pd.DataFrame(zip(data.columns, classification_model.feature_importances_),index = range(data.shape[1]), columns = ['Feature','Importance'])
print(f"Feature importances : \n {feats_importances[['Feature','Importance']]}")   #   computed as the (normalized) total reduction of the criterion brought by each and every feature