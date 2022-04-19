
# assign the desired permission to the folder of the project for accessing code + data
sudo chmod 700 -R the/location/of/your/project/forest_coverage_prediction
# start the ssh client and server
sudo service ssh --full-restart
# start hadoop
start-dfs.sh
# copy the data to hadoop file system
hdfs dfs -put forest_coverage_prediction/data hdfs://localhost:9000/user/

pyspark