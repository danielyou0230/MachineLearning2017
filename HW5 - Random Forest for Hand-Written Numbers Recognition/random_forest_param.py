import numpy as np
import pandas 
import random_forest_module as rfm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

numClass = 5
numfeatures = 28
numTrees = 100
#minLeafNode = 1000
subsample = 0.8

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
##for itr_feat in range(1, 15):
#for numTrees in range(100, 2501, 100):
for itr_pow in range(1, 11):
#for itr_frac in range(5, 101, 5):
	##numfeatures = itr_feat * 28
	minLeafNode = pow(2, itr_pow)
	#subsample = itr_frac / 100.0
	for itr in range(10):
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
		test_features = testing_data.dot(extractor)

		# Random Forest 
		randomForest = RandomForestClassifier(n_estimators = numTrees, \
											  min_samples_leaf = minLeafNode)
		#randomForest = randomForest.fit(train_features, training_targ)
		randomForest = randomForest.fit(train_features, target)

		prediction = randomForest.predict(test_features)
		#print "fraction: {:2d}%".format(itr_frac)
		print "minLeafNode: {:4d}".format(minLeafNode)
		#print "tree: {:4d}".format(numTrees)
		depth = [itr.tree_.max_depth for itr in randomForest.estimators_]
		print "mean: {:3d}, max: {:3d}, min: {:3d}".format(int(np.array(depth).mean()), \
														   int(np.array(depth).max()),  \
														   int(np.array(depth).min()) )
		rfm.evaluate_result(prediction, testing_targ, numClass)

