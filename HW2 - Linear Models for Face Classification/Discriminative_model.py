import numpy as np
import math
from PIL import Image
import os
import pandas 
# Machine Learning @ NCTU EE
# 0310128 Daniel You
# Homework 2 - Face Classification (Discriminative Model)

# Constants / Parameter
cv_part = 3
cv_enable = False
cv_iteration = cv_part if cv_enable else 1

random_suffle = True
pca_dimension = 2
reserved_part = [0.10, 0.10, 0.20]
converge = [10, 10, 10]
order = 2
adj = pow(10.0, 7)
w_init = pow(10.0, -10)

Train_Path = "Data_Train/"
Demo_Path  = "Demo/"
#
numClass = sum(os.path.isdir(os.path.join(Train_Path, itr_dir)) \
							for itr_dir in os.listdir(Train_Path))

numDemo = sum(os.path.isfile(os.path.join(Demo_Path, itr_file)) \
							for itr_file in os.listdir(Demo_Path))

import random

#
def data_partition(Train_Path, numClass, reserved_part):
	verifAmt = []
	trainAmt = []
	print "Training info:"
	for itr_class in range(numClass):
		file_dir = "{:s}Class{:d}/".format(Train_Path, itr_class + 1)
		num_file = sum(os.path.isfile(os.path.join(file_dir, itr_file)) \
					   for itr_file in os.listdir(file_dir))

		print " - Class {:1d}:{:5d} files in total." \
			  .format(itr_class + 1, num_file)
		print "\t    {:3.1f} % of data are reserved for verifying." \
			  .format(reserved_part[itr_class] * 100)

		verifAmt.append(int(num_file * reserved_part[itr_class]))
		trainAmt.append(int(num_file * (1 - reserved_part[itr_class])))

	return verifAmt, trainAmt

###########################################################################

# number of class to be train
verifAmt, trainAmt = data_partition(Train_Path, numClass, reserved_part)
cv_trainAmt = [itr_amt / cv_iteration for itr_amt in trainAmt]

#

def exportData(filename, z):
	df = pandas.DataFrame({'z' : z})
	df.to_csv(filename, header=False, index=False)

def exportData3d(filename, C1, C2, C3):
	df = pandas.DataFrame({'Class_1': C1, 'Class_2': C2, 'Class_3': C3})
	df.to_csv(filename, header=False, index=False)

def generate_classInfo(dataAmt):
	# Generate Class information for data
	numClass = len(dataAmt)

	class_info = []
	for itr_class in range(numClass):
		for itr in range(dataAmt[itr_class]):
			class_info.append(itr_class)

	return class_info

def loadImg2npArray(random_suffle, trainAmt, verifAmt):
	#print "\nLoading images..."
	# trainAmt: a list store the number of file per class for training
	numClass = len(trainAmt)

	# Check if the data in each class are balanced.
	b_balanced = True
	for itr in range(numClass):
		b_balanced = b_balanced and (trainAmt[itr] == np.array(trainAmt).mean())

	print " - {:s} training data".format("Balanced" if b_balanced else "Unbalanced")

	###############################################
	training_list = []
	testing_list = []
	for itr_class in range(numClass):
		# Generate training data path name
		file_dir = "Data_Train/Class{:d}/".format(itr_class + 1)
		tmp_rand = []

		for itr_file in range(trainAmt[itr_class] + verifAmt[itr_class]):
			# Generate training data name
			file_name = "faceTrain{:d}_{:d}.bmp".format(itr_class + 1, itr_file + 1)

			# Convert the image to np.array for dimension reduction
			tmp_img = np.array(Image.open(file_dir + file_name))
			# reshape the image to column vectors
			tmp_img = tmp_img.reshape(1, tmp_img.shape[0] * tmp_img.shape[1])

			# Without Suffle - Divide data to training and testing data
			# Append to the data matrix
			if (not random_suffle) and (itr_file < trainAmt[itr_class]):
				training_list.append(tmp_img)
			elif (not random_suffle) and (itr_file >= trainAmt[itr_class]):
				testing_list.append(tmp_img)
			# Random Suffle - Divide data to training and testing data
			elif random_suffle:
				tmp_rand.append(tmp_img)

		# Random Suffle - Divide data to training and testing data
		if random_suffle:
			# Shuffle the data
			tmp_data = np.random.permutation(np.vstack(tmp_rand))
			training_list.append(tmp_data[:trainAmt[itr_class]])
			testing_list.append(tmp_data[-1 * verifAmt[itr_class]:])

	return np.vstack(training_list), np.vstack(testing_list)

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

def lda(data, n, trainAmt, training_class):
	dimension = data.shape[1]
	numClass = len(trainAmt)

	# 1. Computing the d-dimensional mean vectors
	offset = 0
	mean_vectors = []
	for itr_class in range(numClass):
		tmp_mean = []
		for itr_dim in range(dimension):
			mean = data[offset:offset + trainAmt[itr_class], itr_dim].mean()
			tmp_mean.append(mean)
		mean_vectors.append(np.array(tmp_mean))
		offset += trainAmt[itr_class]
	
	# 2. Computing the Scatter Matrices
	#   2.1 - Within-class scatter matrix SW
	offset = 0
	S_W = np.zeros((dimension, dimension))
	for itr_class in range(numClass):
		# scatter matrix for every class
		class_sc_matrix = np.zeros((dimension, dimension))

		numData = trainAmt[itr_class]
		itr_mean = mean_vectors[itr_class].reshape(dimension, 1)
		#for itr_data in data[training_class == itr_class]:
		for itr_element in range(trainAmt[itr_class]):
			# make column vectors
			itr_data = data[offset + itr_element, :].reshape(dimension, 1) 
			
			class_sc_matrix += (itr_data - itr_mean) \
						  .dot((itr_data - itr_mean).transpose())
		
		S_W += class_sc_matrix

	#   2.2 - Between-class scatter matrix SB
	overall_mean = np.mean(data, axis=0)
	
	S_B = np.zeros((dimension, dimension))
	for itr_class, mean_vec in enumerate(mean_vectors):  
		numData = trainAmt[itr_class]

		mean_vec = mean_vec.reshape(dimension, 1) 
		overall_mean = overall_mean.reshape(dimension, 1) 
		
		S_B += numData * (mean_vec - overall_mean). \
					 dot((mean_vec - overall_mean).transpose())

	# 3. Solving the generalized eigenvalue problem by SW_inv * SB
	eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

	# 4. Selecting linear discriminants for the new feature subspace
	# 4.1. Sorting the eigenvectors by decreasing eigenvalues
	# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
	
	# 4.2. Choosing k eigenvectors with the largest eigenvalues
	proj_mtx = np.hstack((eig_pairs[itr][1].reshape(dimension, 1) \
						  for itr in range(n)))
	eig_vals = np.hstack((eig_pairs[itr][0] for itr in range(dimension)))

	return proj_mtx, eig_vals

def polynomial(x, y, order, adj):
	# Polynomial basis function
	phi = []

	for n in range(order + 1):
		for x_n in range(n + 1):
			phi.append([pow(x, x_n) * pow(y, n-x_n) / adj])

	return np.hstack(phi)

def phi_transform(data, trainAmt, order, adj):
	# Transform data to phi domain
	numClass = len(trainAmt)
	phi = []

	offset = 0
	for itr_class in range(numClass):
		for itr_element in range(trainAmt[itr_class]):
			phi.append(polynomial(data[offset + itr_element, 0], \
								  data[offset + itr_element, 1],
								  order, adj))
		offset += trainAmt[itr_class]

	return np.vstack(phi)

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def posterior_probability(w, x):
	# y = p(Ck | phi(x)) = sigmoid(w.t * phi(x))
	numClass = 3
	numData, dimension = x.shape
	w = w.reshape(1, dimension)
	y = []

	for itr_data in range(numData):
		tmp_x = x[itr_data].reshape(1, dimension)
		a = float(np.dot(w, tmp_x.transpose()))
		y = np.append(y, sigmoid(a))

	return np.array(y).reshape(numData, 1)

def newton_raphson(numClass, dsgMtx, training_class, w_init, converge):
	# Using Newton-Raphson to find optimized w
	# Textbook P.207 Eq 4.95
	print "Optimized with Newton-Raphson's Method"
	bNewtonRaphson = [True] * numClass
	numTrain, dimension = dsgMtx.shape

	dsgMtx_T = dsgMtx.transpose()
	classify_oneOfk = expand_oneOfk(numClass, np.array(training_class))
	
	# Initialize w
	w_trans = [np.array([random.uniform(w_init, w_init)] * dimension) \
			  for itr in range(numClass)]

	# Iterative found solution
	for itr_class in range(numClass):
		NR_itr = 0
		
		# Target value
		t = classify_oneOfk[:, itr_class]
		t = t.reshape(t.shape[0], 1)
		while bNewtonRaphson[itr_class]:
			w = w_trans[itr_class].reshape(dimension, 1)
			# Predict value
			# y = p(Ck | phi(x)) = sigmoid(w.t * phi(x))
			y = posterior_probability(w, dsgMtx)
			# Weighing matrix
			R = np.zeros((y.shape[0], y.shape[0]))
			for itr in range(R.shape[0]):
				R[itr, itr] = y[itr] * (1 - y[itr])
			# 
			w_curr = w - np.dot(np.dot(np.linalg.pinv( \
									   np.dot(dsgMtx_T.dot(R), dsgMtx)), \
								dsgMtx_T), \
						 (y - t))
			
			#print "Newton-Raphson Class {:1d}, itr: {:3d}:\nw = {:s}"\
			#	  .format(itr_class + 1, NR_itr + 1, str(w_curr[itr_class].transpose()))
			w_trans[itr_class] = w_curr

			if NR_itr == converge[itr_class]:
				bNewtonRaphson[itr_class] = False
			NR_itr += 1

		print "Newton-Raphson Class {:1d}, total iteration: {:3d}" \
			  .format(itr_class + 1, NR_itr - 1)

	return np.array(w_trans)

def affine_transform(w_trans, x):
	# pass phi(x) as x 
	# a_k = w.T * x
	numClass = w_trans.shape[0]

	aff_trans = []
	for itr_class in range(numClass):
		w = w_trans[itr_class].reshape(w_trans[0].shape[0], 1)
		x = x.reshape(x.shape[0], 1)

		aff_trans.append(np.dot(w.transpose(), x))

	return aff_trans

def expand_oneOfk(numClass, classify_result):
	# Expand classify result to One-of-K format
	numTest = classify_result.shape[0]

	classify_oneOfk = np.zeros((numTest, numClass), dtype=np.int)
	for itr_test in range(numTest):
		classify_oneOfk[itr_test, classify_result[itr_test]] = 1

	return classify_oneOfk

def softmax(w_trans, x):
	numClass = w_trans.shape[0]
	numTest = x.shape[0]

	classify_result = np.array([], dtype=np.int)
	# Evaluate every test data
	for itr_test in range(numTest):
		p_Ck_x = []

		aff_trans = affine_transform(w_trans, x[itr_test])
		# P(Ck|phi), phi: phi(x)
		denominator = 0
		for itr_aff in aff_trans:
			denominator += math.exp(itr_aff)
		for itr_aff in aff_trans:
			p_Ck_x.append(math.exp(itr_aff) / denominator)
		# Find the most possible class based on probability
		classify_result = np.append(classify_result, \
									np.array(p_Ck_x).argmax())

	return classify_result

def determineCorrectness(classify_result, testing_class, verifAmt, cv_enable, itr_cv):
	#correctness = 0
	numClass = len(verifAmt)
	totalAmt = len(classify_result)
	
	error_amt = 0
	err_inclass = [0] * numClass
	for itr in range(len(classify_result)):
		#if classify_result[itr] == testing_class[itr]:
		#	correctness += 1
		if classify_result[itr] != testing_class[itr]:
			error_amt += 1 
			err_inclass[testing_class[itr]] += 1

	if not cv_enable:
		print "\n ===== Summary ===== "
	elif cv_enable:
		print " Iteration: #{:1d}".format(itr_cv + 1)
	#print "Overall correctness: {:6.2f}% ({:4d} of {:4d})" \
	#		.format(100.0 * correctness / totalAmt,
	#				correctness, totalAmt )
	print "Error rate: {:7.2f}% ({:4d} of {:4d})" \
			.format(100.0 * error_amt / totalAmt,
					error_amt, totalAmt )

	if not cv_enable:
		print "\nIn-Class error rate: "
		for itr_class in range(numClass):
			print " - Class {:1d}: {:7.2f}% ({:4d} of {:4d})" \
					.format(itr_class + 1, \
							100.0 * err_inclass[itr_class] / verifAmt[itr_class], \
							err_inclass[itr_class], \
							verifAmt[itr_class] )

	return (100.0 * error_amt / totalAmt)

# Code for Demo
def predictDemoData(Demo_Path, numDemo, proj_mtx, w_trans, order, adj):
	print "\n *** This part is used for demo ***"
	numClass = w_trans.shape[0]
	# Load Demo data
	print "Loading Demo data..."
	img_list = []
	for itr_file in range(numDemo):
		# Generate training data name
		file = "{:s}{:d}.bmp".format(Demo_Path, itr_file + 1)
		# Convert the image to np.array for dimension reduction
		tmp_img = np.array(Image.open(file))
		# 
		# Reshape image to column vector && Append to matrix
		img_list.append(tmp_img.reshape(1, tmp_img.shape[0] * tmp_img.shape[1]))

	# Project Demo data to projection matrix from PCA
	print "Projecting input Demo data to PCA subspace..."
	demo_transform = np.dot(np.vstack(img_list), proj_mtx)
	# Discriminative Model Prediction
	print "Predicting Demo data using Discriminative Model..."
	phi = phi_transform(test_transform, verifAmt, order, adj)
	classify_result = softmax(w_trans, phi)
	classify_oneOfk = expand_oneOfk(numClass, classify_result)

	# Export prediction to csv file
	print "Exporting prediction to CSV file."
	exportData3d("Demo_Target.csv", \
				 classify_oneOfk[:, 0], \
				 classify_oneOfk[:, 1], \
				 classify_oneOfk[:, 2])

#########################################################################

print " - Shuffle data    : {0}".format(random_suffle)
print " - Cross-Validation: {0}".format(cv_enable)

training_data, testing_data = loadImg2npArray(random_suffle, trainAmt, verifAmt)

testing_class  = generate_classInfo(verifAmt)
training_class = generate_classInfo(cv_trainAmt)

proj_mtx, eigenvalues = pca(training_data, pca_dimension)
#proj_mtx, eigenvalues = lda(training_data, pca_dimension, trainAmt, training_class)

# Acquiring eigenvalues data for analysis
#total = sum(eigenvalues[1:])
#eigval_std = [itr_eigval / total * 100 for itr_eigval in eigenvalues[1:]]
#exportData("eigenvalues_Dis.csv", eigval_std)

# Project training and testing data to projection matrix from PCA
#print "Projecting input training and testing data to PCA subspace..."
data_transform = np.dot(training_data, proj_mtx)
test_transform = np.dot(testing_data, proj_mtx)

# Calculate average error rate in cross-validation
cv_err = []

for itr_cv in range(cv_iteration):
	# Partition training data
	cv_data_list = []
	offset = 0
	for itr_class in range(numClass):
		cv_data_list.append(data_transform[ \
							offset + cv_trainAmt[itr_class] * itr_cv: \
							offset + cv_trainAmt[itr_class] * (itr_cv + 1), :])
		offset += trainAmt[itr_class]
	# Convert to np.array
	cv_data_transform = np.vstack(cv_data_list)
	# Transform the data to phi domain
	phi = phi_transform(cv_data_transform, cv_trainAmt, order, adj)
	
	# Training Multi-Class Discriminative Model
	# Optimized w_trans using Newton-Raphson's Method
	w_trans = newton_raphson(numClass, phi, training_class, w_init, converge)

	# Discriminative Model Prediction
	#print "Calculating class probability and making predictions..."
	phi_test = phi_transform(test_transform, verifAmt, order, adj)
	classify_result = softmax(w_trans, phi_test)
	classify_oneOfk = expand_oneOfk(numClass, classify_result)

	# Summary of verification
	err_itr = determineCorrectness(classify_result, testing_class, \
								   verifAmt, cv_enable, itr_cv)
	cv_err.append(err_itr)

if cv_enable:
	print "\n*** Average error rate: {:7.2f}% ***".format(np.array(cv_err).mean())

# Function for Demo
#predictDemoData(Demo_Path, numDemo, proj_mtx, w_trans, order, adj)
