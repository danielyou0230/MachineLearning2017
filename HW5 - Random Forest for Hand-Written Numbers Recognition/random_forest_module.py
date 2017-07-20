import numpy as np
import pandas 
from sklearn.decomposition import PCA as sklPCA
import random

def load_data(train_file, test_file):
	data_train = pandas.read_csv(train_file, header=None)
	data_test  = pandas.read_csv(test_file,  header=None)
	
	data_train = data_train.as_matrix()
	data_test  = data_test.as_matrix()
	return data_train, data_test.ravel()

def dimension_reduction(data, test, n):
	dimReduction = sklPCA(n_components=n, whiten=True)
	transformed_data = dimReduction.fit_transform(data)
	transformed_test = dimReduction.transform(test)

	return transformed_data, transformed_test

def evaluate_result(prediction, target, numClass):
	numData = len(prediction)
	err = [0] * numClass
	result = [0] * numData
	for itr in range(numData):
		if prediction[itr] != target[itr]:
			err[target[itr] - 1] += 1
			result[itr] = 1
	
	print "Error rate: {:.2f}% ({:4d}/{:4d})" \
		  .format(100.0 * sum(err) / numData, \
				  sum(err), numData)
	
	info_str = " - "
	for itr in range(numClass):
		if itr != numClass - 1:
			info_str += "Class{:2d}: {:3d}, ".format(itr + 1, err[itr])
		else:
			info_str += "Class{:2d}: {:3d}".format(itr + 1, err[itr])
	print info_str

def subsample_data(data, target, frac):
	numData = data.shape[0]
	amount = int(numData * frac)

	sample = list()
	sample_t = list()
	for itr in range(amount):
		index = random.randint(1, numData) - 1
		sample.append(data[index, :])
		sample_t.append(target[index])

	sample = np.vstack(sample)
	sample_t = np.vstack(sample_t)
	return sample, sample_t.reshape(len(sample_t), )

