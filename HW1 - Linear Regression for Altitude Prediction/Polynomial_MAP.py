import numpy as np
import pandas 
from math import sqrt
# Machine Learning @ NCTU EE
# 0310128 Daniel You
# Homework 1

# Extracting information from the file
colname = ['x', 'y']
training_data = pandas.read_csv('X_train.csv', names=colname)
rawX = training_data.x.tolist()
rawY = training_data.y.tolist()

colname = ['x_t', 'y_t']
testing_data = pandas.read_csv('X_test.csv', names=colname)
testX = testing_data.x_t.tolist()
testY = testing_data.y_t.tolist()

colname = ['t']
training_trag = pandas.read_csv('T_train.csv', names=colname)
rawT = training_trag.t.tolist()

# Constant Parameters
max_X = 1081
max_Y = 1081
region_Size = 30
num_Data = len(rawX)
skip_interval = 6
order = 2

# Partitioned region size
valid_list = []
subRegLst = []
region = 0

# Cross Validation
# - Bool
cv_enable = False
# - partitition amount
cv_part = 3
cv_amt = cv_part if cv_enable else 1
overlap = 0

def plot(X, Y, Z):
	fig = plt.figure()
	ax = Axes3D(fig)
	X = np.reshape(X, (sq_test, sq_test))
	Y = np.reshape(Y, (sq_test, sq_test))
	Z = np.reshape(Z, (sq_test, sq_test))

	surf = ax.plot_surface(X, Y, Z, \
		rstride=7, cstride=7, cmap='rainbow', alpha=0.8, linewidth=0.5, vmin=0,vmax=650)
	plt.show()

def export_coordinates(filename, x, y):
	df = pandas.DataFrame({'x' : x, 'y' : y})
	df.to_csv(filename, header=False, index=False)

def export_predictions(filename, z):
	df = pandas.DataFrame({'z' : z})
	df.to_csv(filename, header=False, index=False)

# Feature vector
def polynomial(x, y, order):
	# Polynomial curver fitting
	phi = np.array([0])
	empty = True

	for n in range(order + 1):
		for x_n in range(n + 1):
			if empty:
				phi = np.array([x**x_n * y**(n-x_n)])
				empty = False
			else:
				phi = np.append(phi, x**x_n * y**(n-x_n))
	return phi

def polynomial_MAP(valid_list, subRegLst, regularizer, itr_count, order, cv_enable, cv_amt):
	skip_interval = 6

	#print "Calculating w_MAP..."
	
	#print "mean of pts in region = " + str(np.array(valid_list).mean())
	#print "std  of pts in region = " + str(np.array(valid_list).std())
	#def solve_ML(region, subRegLst, valid_list, w_MAP):
	# Calculate w_MAP for each region
	w_MAP = []
	verify_lst = []
	region_lst = []
	# iterate each region
	for itr_region in range(region):
		#iterate each points in the region
		arrayEmpty = True
		tmp_lst = []
	
		for itr_pts in range(valid_list[itr_region]):
			
			pts_x = subRegLst[itr_region][itr_pts][0]
			pts_y = subRegLst[itr_region][itr_pts][1]
			
			if cv_enable:
				if itr_pts >= (itr_count * int(valid_list[itr_region] / cv_amt)) \
					and itr_pts <= ((itr_count + 1) * int(valid_list[itr_region] / cv_amt)):
					tmp_lst.append([pts_x, pts_y, subRegLst[itr_region][itr_pts][2]])
					continue
				#else do nothing
			else:
				# Push verify data to the list 
				if itr_pts % skip_interval == 0:
					tmp_lst.append([pts_x, pts_y, subRegLst[itr_region][itr_pts][2]])
					continue
				#else do nothing
		
			
			phi = polynomial(pts_x, pts_y, order)
			#print phi
			if arrayEmpty:
				targt = np.array([subRegLst[itr_region][itr_pts][2]])
				dsgMtx = phi
				arrayEmpty = False
				#print "First time (dsgMtx)"
				#print dsgMtx
			else:
				#print "Design Matrix " + str(itr_region) + ", " + str(itr_pts)
				#print dsgMtx
				targt = np.vstack([targt, subRegLst[itr_region][itr_pts][2]])
				dsgMtx = np.vstack([dsgMtx, phi])
	
			itr_pts = itr_pts + 1
		
		verify_lst.append(tmp_lst)
	
		# Regression Ch3 P.8
		# Solve for w_MAP
		# w_MAP = pinv(phi.' * phi) * phi.' * t
		pinv = np.linalg.pinv(regularizer * np.identity(dsgMtx.shape[1], dtype=np.int) \
							 + np.dot(dsgMtx.transpose(), dsgMtx))
		w_MAP.append(np.dot(np.dot(pinv, dsgMtx.transpose()), np.array(targt)))
	
		itr_region = itr_region + 1
	
	#print w_MAP[0]
	#print "Completed."
	#print w_MAP
	
	#print "Verifying training result..."
	arrayEmpty = True
	
	for itr_region in range(len(verify_lst)):
		# w of the corresponding region
		w_region = w_MAP[itr_region]
		#print itr_region
		for itr_pts in range(len(verify_lst[itr_region])):
			# Acquire the test points
			pts_x = verify_lst[itr_region][itr_pts][0]
			pts_y = verify_lst[itr_region][itr_pts][1]
			test = polynomial(pts_x, pts_y, order)
	
			# t_hat = w_T * x
			targt_hat = np.dot(w_region.transpose(), test)
	
			# Acquire the target
			targt = verify_lst[itr_region][itr_pts][2]
			tmp_err = (targt_hat - targt) ** 2
			#print tmp_err
	
			if arrayEmpty:
				error = np.array([tmp_err])
				arrayEmpty = False
			else:
				error = np.vstack([error, tmp_err])
	
	#map_bool = [0] * region
	#map_x = []
	#map_y = []
	#map_t = []
	t_predictions = []
	#print max_X / region_Size
	#print "Making Predictions..."
	for itr in range(len(testX)):
		x_idx = int(testX[itr] / region_Size)
		y_idx = int(testY[itr] / region_Size) 
	
		if x_idx == int(max_X / region_Size):
			x_region = (x_idx - 1) * int(max_X / region_Size)
		else:
			x_region = x_idx * int(max_X / region_Size)
		if y_idx == int(max_X / region_Size):
			y_region = (y_idx - 1)
		else:
			y_region = y_idx
	
		idx_region = x_region + y_region
	
		phi = polynomial(testX[itr], testY[itr], order)
		tmp_pred = np.dot(w_MAP[idx_region].transpose(), phi)
	
		# Push into predictions
		if itr == 0:
			t_predictions.append(float(tmp_pred))
			#map_x.append(testX[itr])
			#map_y.append(testY[itr])
			#map_t.append(float(tmp_pred))
			#map_bool[idx_region] = 1
		else:
			t_predictions.append(float(tmp_pred))
			#if map_bool[idx_region] == 0:
			#	map_x.append(testX[itr])
			#	map_y.append(testY[itr])
			#	map_t.append(float(tmp_pred))
			#	map_bool[idx_region] = 1
	
	#print " - Checking and making up missing points in each region..."
	#for itr in range(len(map_bool)):
	#	if map_bool[itr] == 0:
	#		map_x.append(subRegLst[itr][0][0])
	#		map_y.append(subRegLst[itr][0][1])
	#		map_t.append(subRegLst[itr][0][2])
	#print " - Done."
	#print "Done.\n"
	
	#print "Exporting predictions..."
	##export_coordinates(x_plot, y_plot)
	#export_coordinates('xy_map_ML.csv', map_x, map_y)
	#export_predictions('zz_map_ML.csv', map_t)
	export_predictions('MAP.csv', t_predictions)
	#print "Done.\n"
	
	#plot(testX, testY, t_predictions)

	#print "Completed.\n"
	#print "=============Parameter Infomation=============="
	#print "  Iteration : " + str(itr_count)
	if cv_enable:
		print "================== Iteration =================="
		print "  Iteration           : " + str(itr_count + 1) + " / " + str(cv_amt)
		print "  Cross-Validation    : " + str(cv_amt)
	print "===============Evaluation Result==============="
	#print "  Sub-Region size     = " + str(region_Size)
	print "  Polynomial Order    = " + str(order)
	print "  Lambda              = " + str(regularizer)
	print "  Mean Square Error   = " + str(error.mean())
	print "  Standard Deviation = " + str(sqrt(error.std()))
	#if not cv_enable:
	#	print "  Standard Deviation = " + str(sqrt(error.std()))
	#	print "  Maximum Error      = " + str(sqrt(error.max()))
	#	print "  Minimum Error      = " + str(sqrt(error.min()))
	#print "==============================================="


#print "Partitioning training data..."
for x_rng in xrange(region_Size, max_X + 20, region_Size):
	for y_rng in xrange(region_Size, max_Y + 20, region_Size):
		# initialize the np.array
		subRegion = np.zeros(shape=(1,3), dtype=np.int)
		arrayEmpty = True

		entry = 0
		for itr in range(num_Data):
			if (rawX[itr] <= x_rng + overlap) and \
			(rawX[itr] >= x_rng - region_Size - overlap) and \
			(rawY[itr] <= y_rng + overlap) and \
			(rawY[itr] >= y_rng - region_Size - overlap):

				tmp_array = np.array([rawX[itr], rawY[itr], rawT[itr]])

				if arrayEmpty:
					subRegion = tmp_array
					arrayEmpty = False
				else:
					subRegion = np.vstack([subRegion, tmp_array])
				
				#print "x = " + str(x_tmp)
				#print "y = " + str(y_tmp)
				#print" t = " + str(rawT[itr])
				entry = entry + 1

		# Push the valid number of points in each region to the list
		valid_list.append(entry - 1)
		subRegLst.append(subRegion)
		# move to next region
		region = region + 1

#print "Completed."
itr_count = 1
regularizer = 10

for itr_count in range(cv_amt):
	polynomial_MAP(valid_list, subRegLst, regularizer, itr_count, order, cv_enable, cv_amt)
#
#for regularizer in range(0, 901, 100):
#	for itr_count in range(cv_amt):
#		polynomial_MAP(valid_list, subRegLst, regularizer, itr_count, order, cv_enable, cv_amt)
	#itr_count = itr_count + 1


