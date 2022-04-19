#Hyperparameter tuning

#define the hyperopt train_eval function 
from hyperopt import hp, fmin, SparkTrials, tpe, STATUS_OK, space_eval
import mlflow

from sklearn.metrics import f1_score


def train_evaluate_model(params, classifier, data, score_func = f1_score):
   
    x_train_inp, y_train_inp, x_dev_inp, y_dev_inp = data
    #x_train_inp, y_train_inp, x_dev_inp, y_dev_inp = (data[0].value, data[1].value,data[2].value, date[3].value)
    #some special care on the type of few parameters for the XGBClassifier
          
    if 'max_depth' in params: params['max_depth']=int(params['max_depth'])
    if 'min_samples_split' in params: params['min_samples_split']=int(params['min_samples_split'])
         
    clf = classifier(**params)    
    model = clf.fit(x_train_inp, y_train_inp  )
    preds = model.predict(x_dev_inp)
   
    f1_score = score_func(y_dev_inp, preds, labels = range(1,8), average = 'macro')
    loss = f1_score * (-1)
    
    mlflow.log_metric('f1_score',f1_score)
    
    return     {'loss': loss,'status' : STATUS_OK}
    
    
def hyperparameter_tuning(params, classifier, data, search_space, max_evals = None, algo = tpe.suggest, parallelism = 2, verbose = True,):
    run_name = type(classifier).__name__
    with mlflow.start_run(run_name = run_name):
        argmin = fmin(fn = (lambda params, classifier = classifier, data = (data[0].value, data[1].value,data[2].value, data[3].value) : train_evaluate_model(params, classifier, data )), space = search_space,algo = algo, max_evals = max_evals, trials = SparkTrials(parallelism = parallelism), verbose = verbose)
    return space_eval(search_space, argmin)        
    

