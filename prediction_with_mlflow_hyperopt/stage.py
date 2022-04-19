"""
  Ref :  https://databricks.com/notebooks/product_matching/er_03_train_deploy_model.html
"""  

import mlflow

def stage_to_production(model_name):

    #This works only with the following URI schemes : ['databricks', 'http', 'https', 'postgresql', 'mysql', 'sqlite', 'mssql']
    #mv = mlflow.register_model('runs:/{0}/model'.format(run.info.run_id),model_name)
    
    client = mlflow.tracking.MlflowClient()
    
    for mv in client.search_model_versions(f"name = '{model_name}'"):
        if mv.current_stage.lower() == 'production':
            mv.transition_model_version_stage(name = model_name, version = mv.version, stage = 'archived')
        
    client.transition_model_version_stage(name = model_name, version = mv.version, stage = 'production')    