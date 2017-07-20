import numpy as np
import math
from PIL import Image
import os
import pandas 
# Machine Learning @ NCTU EE
# 0310128 Daniel You
# Homework 2 - Face Classification (Generative Model)

# Constants / Parameter
cv_part = 3
cv_enable = True
cv_iteration = cv_part if cv_enable else 1

random_suffle = True
pca_dimension = 2
reserved_part = [0.10, 0.10, 0.10]

Train_Path = "Data_Train/"
Demo_Path  = "Demo/"

#
numClass = sum(os.path.isdir(os.path.join(Train_Path, itr_dir)) \
							for itr_dir in os.listdir(Train_Path))

numDemo = sum(os.path.isfile(os.path.join(Demo_Path, itr_file)) \
							for itr_file in os.listdir(Demo_Path))

#
def data_partition(numClass, reserved_part):
	verifAmt = []
	trainAmt = []
	print "Training info:"
	for itr_class in range(numClass):
		file_dir = "Data_Train/Class{:d}/".format(itr_class + 1)
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
verifAmt, trainAmt = data_partition(numClass, reserved_part)
cv_trainAmt = [itr_amt / cv_iteration for itr_amt in trainAmt]
prior = [1.0 * itr / sum(cv_trainAmt) for itr in cv_trainAmt]
#print "Prior: {:1.3f}, {:1.3f}, {:1.3f}".format(prior[0], prior[1], prior[2])
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

def gaussian_probability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x - mean,2) / (2 * math.pow(stdev,2))))

	return (1 / (2 * math.pi * stdev)) * exponent

def getDataAttributes(data, trainAmt):
	numClass = len(trainAmt)
	dimension = data.shape[1]

	offset = 0
	attributes = []
	for itr_class in range(numClass):
		tmp_attr = []
		for itr_dim in range(dimension):
			mean = data[offset:offset + trainAmt[itr_class], itr_dim].mean()
			std  = data[offset:offset + trainAmt[itr_class], itr_dim].std()
			tmp_attr.append([mean, std])

		attributes.append(tmp_attr)
		offset += trainAmt[itr_class]

	return attributes

def classify_image(attributes, inputVector, prior):
	# inputVector is a row vector with dimension equal to pca dimension
	# i.e. a image vector reduced by pca
	# calClassProbability
	numClass = len(attributes)
	dimension = len(inputVector)

	probabilities = []
	for itr_class in range(numClass):
		tmp_probability = 1

		for itr_dim in range(dimension):
			mean, stdev = attributes[itr_class][itr_dim]
			x = inputVector[itr_dim]
			#print "Class: {:1d} | Dim: {:1d} | mean = {:4.3f}, std = {:4.3f}" \
			#	  .format(itr_class + 1, itr_dim + 1, mean, stdev)
			tmp_probability *= gaussian_probability(x, mean, stdev)
		
		probabilities.append(tmp_probability * abs(math.log(prior[itr_class])))
		#probabilities.append(tmp_probability)
	# makePrediction
	return np.array(probabilities).argmax()

def Generative_Model(attributes, test_transform, prior):
	numTest = test_transform.shape[0]

	classify_result = [classify_image(attributes, test_transform[itr, :], prior) \
								  for itr in range(numTest)]

	return np.array(classify_result)

def Demo_Generative_Model(attributes, test_transform, prior):
	classify_result = [classify_image(attributes, test_transform[itr, :], prior) \
								  for itr in range(test_transform.shape[0])]

	return np.array(classify_result)

def expand_oneOfk(numClass, classify_result):
	# Expand classify result to One of K format
	numTest = classify_result.shape[0]

	classify_oneOfk = np.zeros((numTest, numClass), dtype=np.int)
	for itr_test in range(numTest):
		classify_oneOfk[itr_test, classify_result[itr_test]] = 1

	return classify_oneOfk

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
def predictDemoData(numDemo, proj_mtx, attributes, prior):
	print "\n *** This part is used for demo ***"
	numClass = len(attributes)
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
	# Generative Model Prediction
	print "Predicting Demo data using Generative Model..."
	classify_result = Demo_Generative_Model(attributes, test_transform, prior)
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

# Acquiring eigenvalues data for analysis
total = sum(eigenvalues)
eigval_std = [itr_eigval / total * 100 for itr_eigval in eigenvalues]
#exportData("eigenvalues_Gen.csv", eigval_std)

# Project training and testing data to projection matrix from PCA
#print "Projecting input training and testing data to PCA subspace..."
data_transform = np.dot(training_data, proj_mtx)
test_transform = np.dot(testing_data, proj_mtx)

#exportData3d("PCA_Gen.csv", data_transform[:, 0], \
#							data_transform[:, 1], \
#							training_class)

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

	# Generative Model Prediction
	# Preparing attributes for training data
	attributes = getDataAttributes(cv_data_transform, cv_trainAmt)
	classify_result = Generative_Model(attributes, test_transform, prior)

	# Summary of verification
	err_itr = determineCorrectness(classify_result, testing_class, \
								   verifAmt, cv_enable, itr_cv)
	cv_err.append(err_itr)

if cv_enable:
	print "\n*** Average error rate: {:7.2f}% ***".format(np.array(cv_err).mean())

# Function for Demo
#predictDemoData(numDemo, proj_mtx, attributes, prior)
