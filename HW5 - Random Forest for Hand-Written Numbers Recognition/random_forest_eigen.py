import numpy as np
import pandas 

file_training_data = "data/X_train.csv"
file_training_targ = "data/T_train.csv"

def load_data(train_file, test_file):
	data_train = pandas.read_csv(train_file, header=None)
	data_test  = pandas.read_csv(test_file,  header=None)
	
	data_train = data_train.as_matrix()
	data_test  = data_test.as_matrix()
	return data_train, data_test.ravel()

def pca(data, n):
	#print "Performing PCA on the input data..."
	# n: dimension that w_pca reduces to
	# Variables that make the code more readable
	numData = data.shape[0] # number of data
	dimData = data.shape[1] # dimension of the data
	print " - Reducing dimension from {:d} to {:d}.".format(dimData, n)
	# PCA step by step
	# 1. Compute Covariance Matrix
	cov_matrix = np.cov(data, rowvar=False)

	# 2. Compute eigenvalues and corresponding eigenvectors
	eig_val, eig_vec = np.linalg.eig(cov_matrix)

	# 3. Sort the eigenvalues and corresponding eigenvectors
	#  - Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_val[itr]), eig_vec[:,itr]) for itr in range(dimData)]
	#  - Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort(key=lambda x: x[0], reverse=True)

	# 4. Pick two eigenvectors with the highest eigenvalues
	proj_mtx = np.hstack((eig_pairs[itr][1].reshape(dimData, 1) \
						  for itr in range(n)))
	eig_vals = np.hstack((eig_pairs[itr][0] for itr in range(dimData)))

	return proj_mtx, eig_vals

# 
training_data, training_targ = load_data(file_training_data, \
										 file_training_targ)

proj_mtx, eig_vals = pca(training_data, 5)

eigen_ttl = sum(eig_vals)
eig_std = [itr / eigen_ttl * 100.0 for itr in eig_vals]
df = pandas.DataFrame({'1': eig_std})
df.to_csv("hw5_eigenInfo.csv", header=False, index=False)

