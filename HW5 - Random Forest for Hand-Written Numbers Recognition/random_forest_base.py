import numpy as np
import pandas 
import random_forest_module as rfm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#import Draw

numClass = 5
numfeatures = 100
numTrees = 100
minLeafNode = 1000

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

# Use Neural Network as Feature Extractor
neural_network = MLPClassifier(hidden_layer_sizes = (numfeatures, ), \
							   activation = 'logistic', \
							   solver = 'adam', \
							   batch_size = 'auto', \
							   learning_rate = 'adaptive', \
							   max_iter = 200, \
							   shuffle = True, \
							   verbose = True)
neural_network.fit(training_data, training_targ)
extractor = neural_network.coefs_[0]
train_features = training_data.dot(extractor)

# Random Forest 
randomForest = RandomForestClassifier(n_estimators = numTrees, \
									  min_samples_leaf = minLeafNode)
randomForest = randomForest.fit(train_features, training_targ)
#Draw.draw_ensemble(randomForest)
test_features = testing_data.dot(extractor)
prediction = randomForest.predict(test_features)

print "Number of features extracted: {:3d}".format(numfeatures)
rfm.evaluate_result(prediction, testing_targ, numClass)