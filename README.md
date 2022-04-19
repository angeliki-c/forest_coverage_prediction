# forest_coverage_prediction
Tree coverage prediction using decision trees

 
 
Techniques followed  
 
   	Decision trees (A good coverage exists here [1])
        * Decision trees, as a prediction method is robust to outliers in the data.
	      * In most of the cases, not much preprocessing of the data consumed by the model is required, such as 
          normalization.
        * It generalizes to a much powerful algorithm, random forest algorithm for classification and regression.
	      * Decision tree models' results can be easily interpreted by humans.
  
  
Data set

	The dataset used in this use [2] consists of records of forest-types covering parcels of land in the
	area of Colorado, US, depended on features (approximately 54) describing natural caharacteristics of 
	the land (such as soil type, slope, atmospheric conditions, water etc).
	It includes 581,012 examples.
	This data set is not considered as a big dataset and it may be manageable locally under specific settings, 
	nevertheless it gives us the chance to explore issues that emerge in data analysis at scale.
	For the most part it has already been preprocessed.
	
	
Baseline
  
  	The baseline model used it is just a simple model that assigns predictions based on the probability distribution 
	that each type of forest appears in the input data.


Challenges

   	There is a definitely big number of features of different type. It is aimed to achieve predictions
   	of high accuracy, whereas avoiding overfitting.
   	(The Decision tree algorithm, as most ml algorithms may be resource intensive, therefore we configure 
   	the Spark Cluster driver's and executors' memory appropriately and we put the dataset to the hdfs hadoop
   	filesystem.)
   

Training process
    	
	A 0.9 train and 0.1 test split has been first applied. 
	Secondly, hyperparameter tuning has been performed on a dev set, composed of 0.1 of the train examples 
	and using f1-score as performance metric.
	Training of the model follows, using the best parameters emerged from the previous phase.
	

Evaluation

	Evaluation is performed for the trained model using f1-score as a metric calculated on the test set and 
	compared against the baseline model described above.
	
	
Performance Metrics

	We experiment with various metrics that the scikit learn and Spark ML libraries provide.
	F1-score is adopted for the comparison between the models. 
 

Code

   	decision_tree_prediction_simple.py
   	decision_tree_prediction_using_categorical_feats.py
   	code under prediction_with_mlflow_hyperopt folder
   
   	All can be run interactively with pyspark shell or by submitting e.g. exec(open("project/location/forest_coverage_prediction				
	/decision_tree_prediction_simple.py").read()) for an all at once execution. The code has been tested on a Spark 
	standalone cluster. For the Spark setting, spark-3.1.2-bin-hadoop2.7 bundle has been used.
   	The external python packages that are used in this implementation exist in the requirements.txt file. Install with: 
	   pip install -r project/location/forest_coverage_prediction/requirements.txt
   	This use case is inspired from the series of experiments presented in [3], though it deviates from it, in the
   	programming language, the setting used and in the analysis followed.
   
 

References

	1. https://scikit-learn.org/stable/modules/tree.html
	2. https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
	3. Advanced Analytics with Spark, Sandy Ryza, Uri Laserson, Sean Owen, & Josh Wills


