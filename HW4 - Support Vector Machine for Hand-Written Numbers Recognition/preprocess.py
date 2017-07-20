import re
import numpy as np
import pandas
import math
# define number of nu's in the mode
iteration = [100, 20]
svm_mode = ['linear', 'poly', 'rbf'] # with gamma: ['poly', 'rbf']
poly_order = [2, 3, 4]
svm_type = ["nuSVM" ,"cSVM"]
file_list = ["nu_result.txt", "nu_result_gamma.txt", "c_result.txt"]

for itr_file in file_list:
	with open(itr_file, 'r') as f:
		itr_svm = svm_mode[-2:] if itr_file == "nu_result_gamma.txt" else svm_mode
		iteration_svm = iteration[0] if itr_file[0] == "n" else iteration[1]
		curr_svm = svm_type[0] if itr_file[0] == "n" else svm_type[1] 
		# prepare re for finding integers
		regex = re.compile(r'\d+')
		for itr in range(len(itr_svm)):
			# Get mode
			buff = f.readline()
			mode = buff.split()[1]
			current_itr = iteration_svm * len(poly_order) if mode == 'poly' else iteration_svm
	
			if mode in svm_mode:
				if itr_file == "nu_result_gamma.txt":
					filename = curr_svm + "_gamma" + '_' + mode + '.csv'
				else:
					filename = curr_svm + '_' + mode + '.csv'

				mode_err = list()
				mode_inclass = list()
				nu_Lst = list()
				er_Lst = list()
				or_Lst = list()
	
				for itr_line in range(current_itr):
					# Line 1 example:
					# Nu = 0.01, error rate: 4.88% ( 122/2500)
					buff = f.readline()
					# Get nu and error rate
					nu, err_rate = re.findall("\d+\.\d+", buff)
					if curr_svm == 'cSVM':
						nu = math.log(float(nu), 2.0)
					nu_Lst.append(nu)
					er_Lst.append(err_rate)
					# index -2 of all the integers is the total error number
					#print buff
					tmp = [int(x) for x in regex.findall(buff)]
					if mode == 'poly':
						or_Lst.append(tmp[0])
					total_err = tmp[-2]
					mode_err.append(total_err)
	
					# Line 2 example:
					#  - Class1:  14, Class2:   9, Class3:  46, Class4:  44, Class5:   9
					buff = f.readline()
					# [1, 14, 2, 9, 3, 46, 4, 44, 5, 9] from the example
					line = [int(x) for x in regex.findall(buff)]
					# Odd indices are the inClass error numbers
					inclass_err = [line[2 * itr + 1] for itr in range(len(line) / 2)]
					mode_inclass.append(inclass_err)
	
				mode_inclass = np.vstack(mode_inclass)
				if mode != 'poly':
					df = pandas.DataFrame({'Anu': nu_Lst, 'Arate': er_Lst, \
										  'B_ttl_e': mode_err, \
										  'Class_1': mode_inclass[:, 0], \
										  'Class_2': mode_inclass[:, 1], \
										  'Class_3': mode_inclass[:, 2], \
										  'Class_4': mode_inclass[:, 3], \
										  'Class_5': mode_inclass[:, 4] } )
				else:
					df = pandas.DataFrame({'A1nu': nu_Lst, 'A2rate': er_Lst, \
										  'A3_order': or_Lst, \
										  'A4_ttl_e': mode_err, \
										  'Class_1': mode_inclass[:, 0], \
										  'Class_2': mode_inclass[:, 1], \
										  'Class_3': mode_inclass[:, 2], \
										  'Class_4': mode_inclass[:, 3], \
										  'Class_5': mode_inclass[:, 4] } )
	
				df.to_csv(filename, header=False, index=False)
	
			else:
				print "Mode \"{:s}\" not found! Please re-check the file." \
					  .format(mode)
				break
