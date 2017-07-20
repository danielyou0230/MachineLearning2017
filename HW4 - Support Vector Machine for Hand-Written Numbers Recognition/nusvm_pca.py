import numpy as np
import pandas 
from sklearn.svm import NuSVC
from sklearn.decomposition import PCA as sklPCA

svm_mode = 'rbf' # ['linear', 'poly', ]
poly_order = [2, 3, 4]
nu_resolution = 100.0
dimension = 2
v = 0.3
scaling = 1.0
# 5% (2.16% ), 3.76%
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

def dimension_reduction(data, test, n):
	dimReduction = sklPCA(n_components=n, whiten=True)
	transformed_data = dimReduction.fit_transform(data)
	transformed_test = dimReduction.transform(test)

	return transformed_data, transformed_test

def normalise_preliminary(data):
	dimension = data.shape[1]
	x_max = np.array([data[:, itr_dim].max() for itr_dim in range(dimension)])
	x_min = np.array([data[:, itr_dim].min() for itr_dim in range(dimension)])
	return x_max, x_min

def normalise_dataset(data, x_max, x_min, scaling):
	normalised_data = (data - x_min) / (x_max - x_min) * scaling
	return normalised_data

# 
training_data, training_targ = load_data(file_training_data, \
										 file_training_targ)
testing_data, testing_targ   = load_data(file_testing_data, \
										 file_testing_targ)

transformed_data, transformed_test = \
				 dimension_reduction(training_data, testing_data, dimension)
x_max, x_min = normalise_preliminary(transformed_data)
train_normal = normalise_dataset(transformed_data, x_max, x_min, scaling)
test_normal  = normalise_dataset(transformed_test, x_max, x_min, scaling)

model = NuSVC(nu=v, kernel=svm_mode)
model.fit(training_data, training_targ)
#prediction = model.predict(testing_data)
prediction = model.predict(training_data)

###
sv = [0] * training_data.shape[0]
for itr in range(len(model.support_)):
	sv[model.support_[itr]] = 1

err = [0] * 5
result = [0] * len(prediction)
for itr in range(len(prediction)):
	if prediction[itr] != training_targ[itr]:
	#if prediction[itr] != testing_targ[itr]:
		err[training_targ[itr] - 1] += 1
		#err[testing_targ[itr] - 1] += 1
		result[itr] = 1

print "Nu = {:1.2f}, error rate: {:.2f}% ({:4d}/{:4d})" \
	  .format(v, \
			  100.0 * sum(err) / len(prediction), \
			  sum(err), len(prediction))

print " - Class1: {:3d}, Class2: {:3d}, Class3: {:3d}, Class4: {:3d}, Class5: {:3d}" \
	  .format(err[0], err[1], err[2], err[3], err[4])


#df = pandas.DataFrame({'1_X': test_normal[:, 0], \
#					   '1_Y': test_normal[:, 1], \
#					   '2_T': testing_targ, \
#					   '3_R': result})
df = pandas.DataFrame({'1_X': train_normal[:, 0], \
					   '1_Y': train_normal[:, 1], \
					   '2_T': training_targ, \
					   '3_R': result, \
					   '4_V': sv })
df.to_csv("distribution.csv", header=False, index=False)