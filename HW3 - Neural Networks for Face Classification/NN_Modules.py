import numpy as np
import math
from PIL import Image
import os
import pandas 
from sklearn.decomposition import PCA as sklPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as sklLDA
import random
# Machine Learning @ NCTU EE
# 0310128 Daniel You
# Homework 3 - Face Classification (Neural Network)
# Modules

def exportData3d(filename, C1, C2, C3):
	df = pandas.DataFrame({'Class_1': C1, 'Class_2': C2, 'Class_3': C3})
	df.to_csv(filename, header=False, index=False)

def getDataProperties(filepath):
	return sum(os.path.isdir(os.path.join(filepath, itr_dir)) \
							for itr_dir in os.listdir(filepath))

def dimension_reduction(data, test, n, bLDA, target):
	if not bLDA:
		dimReduction = sklPCA(n_components=n, whiten=True)
		transformed_data = dimReduction.fit_transform(data)
		transformed_test = dimReduction.transform(test)
	else:
		dimReduction = sklLDA(n_components=n)
		transformed_data = dimReduction.fit_transform(data, target)
		transformed_test = dimReduction.transform(test)

	return transformed_data, transformed_test 

def generate_classInfo(dataAmt):
	# Generate Class information for data
	numClass = len(dataAmt)

	class_info = []
	for itr_class in range(numClass):
		for itr in range(dataAmt[itr_class]):
			class_info.append(itr_class)

	return np.array(class_info)

def expand_oneOfk(numClass, target):
	# Expand classify result to One-of-K format
	numTest = target.shape[0]

	target_oneofk = np.zeros((numTest, numClass), dtype=np.int)
	for itr_test in range(numTest):
		target_oneofk[itr_test, target[itr_test]] = 1

	return target_oneofk

def data_partition(filepath, numClass, suffle, reserved):
	trainIdx = list()
	verifIdx = list()
	trainAmt = list()
	verifAmt = list()

	print "Training info:"
	for itr_class in range(numClass):
		file_dir = "{:s}Class{:d}/".format(filepath, itr_class + 1)
		# Count number of files in the directory
		num_file = sum(os.path.isfile(os.path.join(file_dir, itr_file)) \
					   for itr_file in os.listdir(file_dir))
		# Store the amount of data for training / verifying
		trainAmt.append(int(num_file * (1 - reserved[itr_class])))
		verifAmt.append(int(num_file * reserved[itr_class]))
		# Generate file index list for loading
		fileLst = [itr for itr in range(1, num_file + 1)]
		# Show data partition info
		print " - Class {:1d}:{:5d} files in total." \
			  .format(itr_class + 1, num_file)
		print "\t    {:3.1f} % of data are reserved for verifying." \
			  .format(reserved[itr_class] * 100)
		# 
		if suffle:
			fileLst = list(np.random.permutation(np.hstack(fileLst)))

		trainIdx.append(fileLst[:trainAmt[itr_class]])
		verifIdx.append(fileLst[-1 * verifAmt[itr_class]:])

	#print "\nLoading images..."
	# Check if the data in each class are balanced.
	b_balanced = True
	for itr in range(numClass):
		b_balanced = b_balanced and (trainAmt[itr] == np.array(trainAmt).mean())
	print " - {:s} training data".format("Balanced" if b_balanced else "Unbalanced")

	training_data = list()
	testing_data = list()
	for itr_class in range(numClass):
		# Generate training data path name
		file_dir = "Data_Train/Class{:d}/".format(itr_class + 1)
		# Partition as training and testing data
		#print "Loading images as training data..."
		for train_idx in trainIdx[itr_class]:
			# Generate training data name
			#print "faceTrain{:d}_{:d}.bmp".format(itr_class + 1, train_idx)
			file_name = "faceTrain{:d}_{:d}.bmp".format(itr_class + 1, train_idx)
			# Convert the image to np.array for dimension reduction
			tmp_img = np.array(Image.open(file_dir + file_name))
			# reshape the image to column vectors
			tmp_img = tmp_img.reshape(1, tmp_img.shape[0] * tmp_img.shape[1])
			# Append to the training data list
			training_data.append(tmp_img)
		# Generate testing data path name
		#print "Loading images as testing data..."
		for test_idx in verifIdx[itr_class]:
			# Generate training data name
			#print "faceTrain{:d}_{:d}.bmp".format(itr_class + 1, test_idx)
			file_name = "faceTrain{:d}_{:d}.bmp".format(itr_class + 1, test_idx)
			# Convert the image to np.array for dimension reduction
			tmp_img = np.array(Image.open(file_dir + file_name))
			# reshape the image to column vectors
			tmp_img = tmp_img.reshape(1, tmp_img.shape[0] * tmp_img.shape[1])
			# Append to the training data list
			testing_data.append(tmp_img)

	return np.vstack(training_data), trainAmt, \
		   np.vstack(testing_data),  verifAmt

def cv_partition(cv_parts, amtLst):
	#cv_Amnt = [itr_amt / cv_parts for itr_amt in amtLst]
	return [itr_amt / cv_parts for itr_amt in amtLst]

def cv_getData(cv_itr, cv_Amt, data, trainAmt):
	numClass = len(cv_Amt)
	offset = 0
	cv_data = list()
	for itr in range(numClass):
		cv_data.append(data[offset + cv_Amt[itr] * cv_itr:
							offset + cv_Amt[itr] * (cv_itr + 1)])
		offset += trainAmt[itr]

	return np.vstack(cv_data)

def normalise_preliminary(data):
	dimension = data.shape[1]
	x_max = np.array([data[:, itr_dim].max() for itr_dim in range(dimension)])
	x_min = np.array([data[:, itr_dim].min() for itr_dim in range(dimension)])
	return x_max, x_min

def normalise_dataset(data, x_max, x_min, scaling):
	normalised_data = (data - x_min) / (x_max - x_min) * scaling
	return normalised_data

def initialising_network(neuronLst):
	# neuronLst: list of number of neurons per layer
	# index 0: number of inputs
	# itr_i: number of inputs to the layer
	# itr_n: number of neurons in the layer
	numLayer = len(neuronLst)
	
	network = list()
	for itr_layer in range(1, numLayer):
		tmp = [{'weights': np.array([random.uniform(0, pow(10, -6)) \
						   for itr_i in range(neuronLst[itr_layer - 1] + 1)]) } \
						   for itr_n in range(neuronLst[itr_layer])]
		network.append(tmp)
	return network

def connect_neurons(weights, x):
	# a = sum(w * x) 
	activation = weights[:-1].dot(x.transpose()) + weights[-1]
	return activation

def sigmoid_transferFcn(activation):
	# Logistic Sigmoid function
	return 1.0 / (1.0 + math.exp(-activation))

def sigmoid_derivative(x):
	# sigmoid function derivative = output * (1.0 - output)
	return x * (1.0 - x)

def rectify_transferFcn(activation):
	# Rectify function
	alpha = 0.01
	return activation if activation > 0.0 else \
		   alpha * activation 

def rectify_derivative(x):
	# Rectify function = 0 if x <= 0 else 1
	alpha = 0.01
	#print x
	return 1. if x > 0.0 else alpha

def forward_propagate(network, data, bReLU):
	# propagate from input to output
	inputs = data
	for layer in network:
		output = []
		for neuron in layer:
			neuron['input'] = inputs
			# For every neuron, connect (sum(w * x)) all the nodes
			activation = connect_neurons(neuron['weights'], inputs)
			# Activated the transfer function with sigmoid function
			if not bReLU:
				neuron['output'] = sigmoid_transferFcn(activation)
			else:
				neuron['output'] = rectify_transferFcn(activation)
			# Append to the list for next layer
			output.append(neuron['output'])

		# Propagate to next layer
		inputs = np.array(output)
	return inputs

def calError_backprop(network, target, bReLU):
	numLayer = len(network)
	numClass = len(target)
	for itr_layer in reversed(range(numLayer)):
		curr_layer = network[itr_layer]
		numNeuron = len(curr_layer)

		# Hidden Layers
		if itr_layer != numLayer - 1:
			for itr_neuron in range(numNeuron):
				error = 0.0
				# Consider error from next layer
				for nxt_neuron in network[itr_layer + 1]:
					error += nxt_neuron['weights'][itr_neuron] * nxt_neuron['delta']

				neuron = curr_layer[itr_neuron]
				if not bReLU:
					neuron['delta'] += error \
									 * sigmoid_derivative(neuron['output'])
				else:
					neuron['delta'] += error \
									 * rectify_derivative(neuron['output'])
		# Last layer (output curr_layer): Compare result with target
		else:
			# t - y for every neuron
			for itr_class in range(numClass):
				neuron = curr_layer[itr_class]
				neuron['delta'] += target[itr_class] - neuron['output']

def reset_delta_gradE(network):
	numLayer = len(network)
	for itr_layer in range(numLayer):
		for neuron in network[itr_layer]:
			neuron['delta'] = 0
			neuron['gradE'] = [np.array([]) \
							   for itr in range(len(neuron['weights'])) ]

def update_weights(network, data, l_rate, bMiniBatch, updateGradE):
	# w(new) = w(old) + learning_rate * delta * input
	# mini-batch: w(new) = w(old) + learning_rate * avg(grad(E(w)))
	numLayer = len(network)
	for itr_layer in range(numLayer):
		if itr_layer != 0:
			# Inputs of current layer = outputs from previous layer
			inputs = [neuron['output'] for neuron in network[itr_layer - 1]]
		else: 
			# Initlialize input with input data (bias not included)
			inputs = data

		numInput = len(inputs)
		for neuron in network[itr_layer]:
			for itr in range(numInput): # bias is added later
				if not bMiniBatch:
					neuron['weights'][itr] += l_rate[itr_layer] * neuron['delta'] * inputs[itr]
				elif updateGradE:
					neuron['gradE'][itr] = np.append(neuron['gradE'][itr],
													 neuron['delta'] * inputs[itr])
				else:
					neuron['weights'][itr] += l_rate[itr_layer] * neuron['gradE'][itr].mean()

			# Update bias weight / gradient E
			if not bMiniBatch:
				neuron['weights'][-1] += l_rate[itr_layer] * neuron['delta'] * 1.0
			elif updateGradE:
				neuron['gradE'][-1] = np.append(neuron['gradE'][-1],
												neuron['delta'] * 1.0)
			else:
				neuron['weights'][-1] += l_rate[itr_layer] * neuron['gradE'][-1].mean()

def softmax(probability):
	numClass = len(probability)
	classify_result = np.array([])
	# Evaluate every test data
	p_Ck_x = []
	denominator = 0

	for itr in probability:
		denominator += math.exp(itr)
	for itr in probability:
		p_Ck_x.append(math.exp(itr) / denominator)
	return p_Ck_x

def mini_batch_training(data, targ, batch_iter):
	#
	numData = data.shape[0]
	batch_count = int(numData / batch_iter)

	mbatch_data = list()
	mbatch_targ = list()
	for itr_batch in range(batch_count):
		mbatch_data.append(data[ itr_batch      * batch_iter: \
								(itr_batch + 1) * batch_iter, :] )
		mbatch_targ.append(targ[ itr_batch      * batch_iter: \
								(itr_batch + 1) * batch_iter, :] )

	return mbatch_data, mbatch_targ

def training_nn(network, data, target, l_rate, n_epoch, bReLU, bMiniBatch, batch_iter):
	# target shape: numData x numClass (in one-of-k format)
	numData, numClass = target.shape
	numLayer = len(network)
	batch_count = int(numData / batch_iter)
	iterations = batch_iter if bMiniBatch else numData
	# Randomise the sequence of data 
	bundle = np.random.permutation(np.hstack([data, target]))
	suffle_data = bundle[:, :data.shape[1]]
	suffle_targ = bundle[:, -1 * numClass:]
	#
	for epoch in range(n_epoch):
		if not bMiniBatch:
			error = [0.0] * n_epoch
			for itr_Data in range(numData):
				reset_delta_gradE(network)
				#probability = forward_propagate(network, data[itr_Data], bReLU)
				probability = forward_propagate(network, suffle_data[itr_Data], bReLU)
				prediction = softmax(probability)
				# Current class target
				#curr_target = target[itr_Data, :]
				curr_target = suffle_targ[itr_Data, :]
				error[epoch] += sum([pow(curr_target[itr] - prediction[itr], 2) \
										 for itr in range(numClass)])
				calError_backprop(network, curr_target, bReLU)
				update_weights(network, suffle_data[itr_Data], l_rate, bMiniBatch, False)
			error[epoch] /= numData
		else:
			offset = 0
			error = [0.0] * n_epoch
			mbatch_data, mbatch_targ \
						= mini_batch_training(suffle_data, suffle_targ, batch_iter)
			for itr_batch in range(batch_count):
				reset_delta_gradE(network)
				# Load current batch data and target
				b_data = mbatch_data[itr_batch]
				b_targ = mbatch_targ[itr_batch]
				for itr_Data in range(batch_iter):
					probability = forward_propagate(network, b_data[itr_Data], bReLU)
					prediction = softmax(probability)
					# Current class target
					curr_target = b_targ[itr_Data]
					error[epoch] += sum([pow(curr_target[itr] - prediction[itr], 2) \
											 for itr in range(numClass)])
					calError_backprop(network, curr_target, bReLU)
					# miniBatch update grad(E)
					update_weights(network, b_data[itr_Data], l_rate, bMiniBatch, True)
				
				# miniBatch update weights
				update_weights(network, b_data[0, :], l_rate, bMiniBatch, False)

			error[epoch] /= (batch_iter * batch_count)

		print " epoch: {:3d}, lrate: {:s}, error: {:.3f}" \
			  .format(epoch + 1, l_rate, error[epoch])


		"""# mini batch data preparation
		if bMiniBatch:
			offset = 0
			mbatch_data, mbatch_targ \
						= mini_batch_training(suffle_data, suffle_targ, batch_iter)

		for itr_batch in range(batch_count):
			# Reset delta after each minibatch iteration
			if bMiniBatch:
				reset_delta_gradE(network)
			# Load current batch data and target
			b_data = suffle_data if not bMiniBatch else mbatch_data[itr_batch]
			b_targ = suffle_targ if not bMiniBatch else mbatch_targ[itr_batch]

			for itr_Data in range(iterations):
				# Reset delta after each data iteration
				if not bMiniBatch:
					reset_delta_gradE(network)
				#probability = forward_propagate(network, data[itr_Data], bReLU)
				probability = forward_propagate(network, b_data[itr_Data], bReLU)
				prediction = softmax(probability)
				# Current class target
				curr_target = b_targ[itr_Data, :]
				error[epoch] = sum([pow(curr_target[itr] - prediction[itr], 2) \
										 for itr in range(numClass)])
				calError_backprop(network, curr_target, bReLU)
				# Stochastic Gradient Descent
				if not bMiniBatch:
					update_weights(network, b_data[itr_Data], l_rate, bMiniBatch, False)
				# miniBatch update grad(E)
				else:
					update_weights(network, b_data[itr_Data], l_rate, bMiniBatch, True)

			# miniBatch update weights
			if bMiniBatch:
				update_weights(network, b_data[0, :], l_rate, bMiniBatch, False)"""

def nn_prediction(network, data, bReLU):
	numData = data.shape[0]

	prediction = list()

	for itr_Data in range(numData):
		tmp = forward_propagate(network, data[itr_Data], bReLU)
		prediction.append(tmp.argmax())

	return np.array(prediction)
#############################################################################
def exportData(filename, z):
	df = pandas.DataFrame({'z' : z})
	df.to_csv(filename, header=False, index=False)

def determineCorrectness(classify_result, testing_class, verifAmt, \
						 cv_enable, itr_cv, NN_Layer, bReLU, bLDA,\
						 bMiniBatch, batch_iter, suffle):
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
		print "Neural Network:"
		print " - Input Layer    : {:2d} neurons".format(NN_Layer[0])
		for itr in range(1, len(NN_Layer) - 1):
			print " - Hidden Layer #{:1d}: {:2d} neurons".format(itr, NN_Layer[itr])
		print " - Output Layer   : {:2d} neurons".format(NN_Layer[-1])
		#print " Activation function        : {:s}".format("Sigmoid Function" \
		#										  if not bReLU else  \
		#										  "Rectify Function")
		if bMiniBatch:
			print " mini-Batch Gradient Descent: {0}".format(bMiniBatch)
			print " mini-Batch Size            : {:3d}".format(batch_iter)
		if bLDA:
			print " Used LDA instead of PCA."
	elif cv_enable:
		print " Iteration: #{:1d}".format(itr_cv + 1)
	#print "Overall correctness: {:6.2f}% ({:4d} of {:4d})" \
	#		.format(100.0 * correctness / totalAmt,
	#				correctness, totalAmt )
	print " Error rate: {:7.2f}% ({:4d} of {:4d})" \
			.format(100.0 * error_amt / totalAmt,
					error_amt, totalAmt )

	if not cv_enable:
		print "\n In-Class error rate: "
		for itr_class in range(numClass):
			print " - Class {:1d}: {:7.2f}% ({:4d} of {:4d})" \
					.format(itr_class + 1, \
							100.0 * err_inclass[itr_class] / verifAmt[itr_class], \
							err_inclass[itr_class], \
							verifAmt[itr_class] )

	return (100.0 * error_amt / totalAmt)

def genData(scaling):
	data = list()
	for x in range(1001):
		for y in range(1001):
			data.append(np.array([x / 1000.0 * scaling \
								, y / 1000.0 * scaling]))

	return np.vstack(data)