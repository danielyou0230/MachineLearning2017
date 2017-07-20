import numpy as np
import pandas 
import random_forest_module as rfm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

numClass = 5
numfeatures = 28
numTrees = 100
minLeafNode = 1000
subsample = 0.5

file_training_data = "data/X_train.csv"
file_training_targ = "data/T_train.csv"
file_testing_data  = "data/X_test.csv"
file_testing_targ  = "data/T_test.csv"

training_data, training_targ = rfm.load_data(file_training_data, \
											file_training_targ)
testing_data, testing_targ   = rfm.load_data(file_testing_data, \
											file_testing_targ)

# Use PCA as Feature Extractor
#dimension = 10
#train_features, test_features = rfm.dimension_reduction(training_data, \
#														testing_data,  \
#														dimension)

data, target = rfm.subsample_data(training_data, training_targ, subsample)

# Use Neural Network as Feature Extractor
neural_network = MLPClassifier(hidden_layer_sizes = (numfeatures, ), \
							   activation = 'logistic', \
							   solver = 'adam', \
							   batch_size = 'auto', \
							   learning_rate = 'adaptive', \
							   max_iter = 200, \
							   shuffle = True, \
							   verbose = False)
#neural_network.fit(training_data, training_targ)
neural_network.fit(data, target)
extractor = neural_network.coefs_[0]
#train_features = training_data.dot(extractor)
train_features = data.dot(extractor)

# Equivalent Random Forest 
# (Bagging classifier with base classifier = Decision Tree)
#randomForest = BaggingClassifier( \
#			   base_estimator = DecisionTreeClassifier(min_samples_leaf = minLeafNode), \
#			   n_estimators = numTrees, \
#			   max_samples = subsample, \
#			   bootstrap_features = True )

randomForest = RandomForestClassifier(n_estimators = numTrees, \
									  min_samples_leaf = minLeafNode)

randomForest = randomForest.fit(train_features, target)
for itr in range(len(randomForest.estimators_)):
	filename = "trees/tree_{:d}.dot".format(itr + 1)
	export_graphviz(randomForest.estimators_[itr], out_file=filename) 

test_features = testing_data.dot(extractor)
prediction = randomForest.predict(test_features)

rfm.evaluate_result(prediction, testing_targ, numClass)