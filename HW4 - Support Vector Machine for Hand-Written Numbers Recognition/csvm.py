import numpy as np
import pandas 
import math
from sklearn.svm import SVC

file_training_data = "data/X_train.csv"
file_training_targ = "data/T_train.csv"
file_testing_data  = "data/X_test.csv"
file_testing_targ  = "data/T_test.csv"

def load_data(train_file, test_file):
	data_train = pandas.read_csv(train_file, header=None)
	data_test  = pandas.read_csv(test_file,  header=None)
	
	data_train = data_train.as_matrix()
	data_test  = data_test.as_matrix()
	return data_train, data_test.ravel()

training_data, training_targ = load_data(file_training_data, \
										 file_training_targ)
testing_data, testing_targ   = load_data(file_testing_data, \
										 file_testing_targ)
# 
svm_mode = ['linear', 'poly', 'rbf']
poly_order = [2, 3, 4]
nu_resolution = 100.0

for itr_mode in svm_mode:
	print "Kernel: {:s}".format(itr_mode)
	orderLst = poly_order if itr_mode == 'poly' else [1]
	#
	for itr_order in orderLst:
		for C_penalty in range(0, 20):
			###
			#v = itr_nu / nu_resolution
			C_penalty = math.pow(2,C_penalty)
			model = SVC(C=C_penalty, kernel=itr_mode, degree=itr_order)
			model.fit(training_data, training_targ)
			prediction = model.predict(testing_data)
			###
			
			err = [0] * 5
			for itr in range(len(prediction)):
				if prediction[itr] != testing_targ[itr]:
					err[testing_targ[itr] - 1] += 1
			
			if itr_mode == svm_mode[1]:
				print "Order = {:d}, C = {:1.2f}, error rate: {:.2f}% ({:4d}/{:4d})" \
					  .format(itr_order, C_penalty, \
							  100.0 * sum(err) / len(prediction), \
							  sum(err), len(prediction))
			else:
				print "C = {:1.2f}, error rate: {:.2f}% ({:4d}/{:4d})" \
					  .format(C_penalty, \
							  100.0 * sum(err) / len(prediction), \
							  sum(err), len(prediction))

			print " - Class1: {:3d}, Class2: {:3d}, Class3: {:3d}, Class4: {:3d}, Class5: {:3d}" \
				  .format(err[0], err[1], err[2], err[3], err[4])
