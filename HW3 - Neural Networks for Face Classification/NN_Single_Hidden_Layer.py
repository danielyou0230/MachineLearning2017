import NN_Modules as nn
import numpy as np
# Machine Learning @ NCTU EE
# 0310128 Daniel You
# Homework 3 - Face Classification (Neural Network - Single Hidden Layer)

# Constants / Parameter
cv_part = 4
cv_enable = False
cv_count = cv_part if cv_enable else 1

suffle = True
reduce_dimension = 2

bLDA = False

hidden = 4
bReLU = False

bMiniBatch = False
batch_size = 5
batch_iter = batch_size if bMiniBatch else 1

reserved = [0.20, 0.20, 0.20]
l_rate = [0.10, 0.10]
n_epoch = 200
scaling = 1.0

Train_Path = "Data_Train/"
Demo_Path  = "Demo/"

# Load data
numClass = nn.getDataProperties(Train_Path)
NN_Layer = [reduce_dimension, hidden, numClass]
#numDemo  = nn.getDataProperties(Demo_Path)
training_data, trainAmt, \
testing_data,  verifAmt = nn.data_partition(Train_Path, numClass, suffle, reserved)
cv_Amnt = nn.cv_partition(cv_count, trainAmt)

train_class = nn.generate_classInfo(cv_Amnt)
verif_class = nn.generate_classInfo(verifAmt)
train_oneofk = nn.expand_oneOfk(numClass, train_class)

# Calculate average error rate in cross-validation
cv_err = list()
for itr_cv in range(cv_count):
	network = nn.initialising_network(NN_Layer)
	#print network

	cv_data = nn.cv_getData(itr_cv, cv_Amnt, training_data, trainAmt)
	# Dimension Reduction
	transformed_data, transformed_test \
	= nn.dimension_reduction(cv_data, testing_data, reduce_dimension, \
							 bLDA, train_class)
	#nn.exportData3d("pca.csv", transformed_data[:, 0] \
	#						 , transformed_data[:, 1] \
	#						 , train_class)

	# normalised training and testing data
	x_max, x_min = nn.normalise_preliminary(transformed_data)
	train_normal = nn.normalise_dataset(transformed_data, x_max, x_min, scaling)
	#nn.exportData3d("tran_norm.csv", train_normal[:, 0] \
	#							   , train_normal[:, 1] \
	#							   , train_class)

	nn.training_nn(network, train_normal, train_oneofk, l_rate, n_epoch, \
				   bReLU, bMiniBatch, batch_iter)

	test_normal  = nn.normalise_dataset(transformed_test, x_max, x_min, scaling)
	#nn.exportData3d("test_norm.csv", test_normal[:, 0] \
	#							   , test_normal[:, 1] \
	#							   , verif_class)

	classify_result = nn.nn_prediction(network, test_normal, bReLU)
	err_itr = nn.determineCorrectness(classify_result, verif_class, verifAmt, \
									  cv_enable, itr_cv, NN_Layer, \
									  bReLU, bLDA, bMiniBatch, batch_iter, suffle)
	cv_err.append(err_itr)

gen = nn.genData(scaling)
gen_result = nn.nn_prediction(network, gen, bReLU)
nn.exportData3d("gen_1H.csv", gen[:, 0] \
							, gen[:, 1] \
							, gen_result)

if cv_enable:
	print " \n=== Cross-Validation Summary === "
	print " Iterations        : {:2d}".format(cv_count)
	print " Epoch             : {:3d}".format(n_epoch)
	print " Average error rate: {:7.2f}% ".format(np.array(cv_err).mean())
	print " Maximum error rate: {:7.2f}% ".format(np.array(cv_err).max())
	print " Minimum error rate: {:7.2f}% ".format(np.array(cv_err).min())