import re
import os
import sys
import Queue
from svmutil import *
from subprocess import *

import svmutils.plotroc as plotroc
import svmutils.SVMThread as SVMThread
from svmutils.SVMThread import SVM_TYPES, KERNEL_TYPES
from svmutils.LibSVMExtractor import LibSVMExtractor

NU = [x/100. for x in xrange(1,50,1)] 
DEGREES = [3] #xrange(2,10)
COSTS = xrange(1,100,5)
GAMMAS = xrange(1,5)

svmtrain_exe = os.path.join(os.path.dirname(__file__),r"svm-train.exe")
gnuplot_exe = r"c:\tmp\gnuplot\bin\gnuplot.exe"
grid_py = os.path.join(os.path.dirname(__file__),r".\svmutils\grid.py")
	
def extractSVM(inputFile):
	extractor = LibSVMExtractor(inputFile)
	# Extract the options for the different subjects
	subjects = extractor.extractSubjectNumbers()

	files = []
	for randomParticipant in subjects:
		print "Choosing %d as the first class" % randomParticipant
		output_training = inputFile + ".%d.training" %  randomParticipant
		output_user_test = inputFile + ".%d.user_test" % randomParticipant
		output_test = inputFile + ".%d.test" % randomParticipant
		output_impostor_test = inputFile + ".%d.impostor_test" % randomParticipant
		files.append({"SUBJECT_NUM" : randomParticipant, 
					  "TRAINING_FILE" : output_training, 
					  "TEST_FILE" : output_test, 
					  "USER_TEST_FILE" : output_user_test, 
					  "IMPOSTOR_TEST_FILE" : output_impostor_test})
		(trainingData, userTestData1, userTestData2, impostorTestData) = extractor.extract(randomParticipant)

		with open(output_training, "wb") as outTrain:
			outTrain.write(trainingData)
		
		with open(output_user_test, "wb") as outTest:
			outTest.write(userTestData1)
			outTest.write(userTestData2)
			
		with open(output_impostor_test, "wb") as outTest:
			outTest.write(impostorTestData)
		
		with open(output_test, "wb") as outTest:
			outTest.write(impostorTestData)
			outTest.write(userTestData1)
			outTest.write(userTestData2)
			
	return files

def crossValidate(sampleNum, training_file):
	PAST_C_SVM_CV_COMP =	 { 2 : (8,8),
					   3 : (8,8),
					   4 : (128,2),
					   5 : (8,2),
					   7 : (8,2),
					   8 : (128,2),
					   10 : (32,2),
					   11 : (32,2),
					   12 : (128,0.5),
					   13 : (8,8),
					   15 : (32,2),
					   16 : (128,0.125),
					   17 : (32,2),
					   18 : (32,2),
					   19 : (128,0.5),
					   20 : (32,0.5),
					   21 : (2,8),
					   22 : (128,0.125),
					   24 : (128,0.5),
					   25 : (8192,0.125) }
					   
	PAST_NU_SVM_COMP = { 56: (2, 0.010000),
						54: (2, 0.010000),
						42: (1, 0.020000),
						43: (1, 0.010000),
						53: (1, 0.010000),
						52: (1, 0.010000),
						24: (1, 0.010000),
						25: (2, 0.010000),
						26: (4, 0.010000),
						27: (3, 0.010000),
						20: (4, 0.010000),
						21: (4, 0.010000),
						22: (2, 0.010000),
						49: (1, 0.010000),
						46: (1, 0.010000),
						47: (2, 0.010000),
						44: (4, 0.010000),
						28: (2, 0.010000),
						29: (1, 0.010000),
						40: (1, 0.010000),
						41: (2, 0.020000),
						3: (1, 0.030000),
						2: (1, 0.020000),
						5: (4, 0.010000),
						4: (2, 0.010000),
						7: (4, 0.010000),
						8: (3, 0.010000),
						51: (1, 0.020000),
						39: (1, 0.010000),
						38: (2, 0.010000),
						11: (2, 0.020000),
						10: (3, 0.010000),
						13: (1, 0.020000),
						12: (4, 0.010000),
						15: (1, 0.010000),
						48: (4, 0.020000),
						17: (1, 0.010000),
						16: (1, 0.010000),
						19: (1, 0.010000),
						18: (3, 0.020000),
						31: (3, 0.010000),
						30: (2, 0.010000),
						37: (3, 0.010000),
						36: (1, 0.010000),
						35: (1, 0.020000),
						34: (1, 0.020000),
						33: (1, 0.010000),
						55: (4, 0.010000),
						32: (1, 0.010000),
						57: (2, 0.010000),
						50: (2, 0.010000) }
	
	# Check if the sample was already computed in the past, to reduce running times		   
	if sampleNum in PAST_NU_SVM_COMP:
		return PAST_NU_SVM_COMP[sampleNum]
	else:
		cmd = '{0} -svmtrain "{1}" -gnuplot "{2}" "{3}"'.format(grid_py, svmtrain_exe, gnuplot_exe, training_file)
		print('Cross validation...')
		f = Popen(cmd, shell = True, stdout = PIPE).stdout

		line = ''
		while True:
			last_line = line
			line = f.readline()
			if not line: break
		c,g,rate = map(float,last_line.split())
		print('Best c={0}, g={1}'.format(c,g))
		return c,g
	
def TrainSVM(sampleNum, trainFile):
	g,n = crossValidate(sampleNum, trainFile)
	param = "-n %f -g %f -q -s 1 -t 2" % (n,g)
	#param = "-g %d -c %d -q" % (g,c)
	train_y, train_x = svm_read_problem(trainFile)
	model = svm_train(train_y, train_x, param)

	return {sampleNum : model}

def computeSVMScore(subjectNum, subjectUserData, subjectImposterData, full_svm_model):
	realLabels = []
	decisions = []
	
	print "> %s" % subjectNum
	for modelNum in full_svm_model:
		currentClassifierModel = full_svm_model[modelNum]
		labels = currentClassifierModel.get_labels()
		
		# Compute user-score
		user_test_y, user_test_x = svm_read_problem(subjectUserData)
		py, evals, user_decisions = svm_predict(user_test_y, user_test_x, currentClassifierModel)
	
		if subjectNum == modelNum:
			pys = len(filter(lambda x: x > 0.0, py))
			print "User: Own Model %d: " % (modelNum),
			realLabels += user_test_y
			decisions += [labels[0]*val[0] for val in user_decisions]
			print "%f %%" % ((pys / float(len(user_test_y))) * 100)
		
		else:
			pys = len(filter(lambda x: x < 0.0, py))
			print "User: Other Model %d: " % (modelNum),
			# What we get here is the probability of a certain user to be selected as another authentic user (e.g practiacally an imposter)
			realLabels += [-x for x in user_test_y]
			decisions += [labels[0]*val[0] for val in user_decisions]
			print "%f %%" % ((pys / float(len(user_test_y))) * 100)
		
		# Compute imposter-score
		if subjectNum == modelNum:
			imposter_test_y, imposter_test_x = svm_read_problem(subjectImposterData)
			py, evals, imposter_decisionsi = svm_predict(imposter_test_y, imposter_test_x, currentClassifierModel)
			pys = len(filter(lambda x: x < 0.0, py))
			print "Imposter: Own Model %d: " % (modelNum),
			print "%f %%" % ((pys / float(len(imposter_test_y))) * 100)
			
			realLabels += imposter_test_y
			decisions += [labels[0]*val[0] for val in imposter_decisionsi]
		
	return realLabels, decisions
	
def computeFRR(realLabels, decisions):
	frr = 0
	for i in xrange(len(realLabels)):
		# Choose the genuine users
		if (realLabels[i] > 0):
			# Which were decisionslared imposters
			if (decisions[i] < 0):
				frr += 1
	authentics_number = float(len(filter(lambda x: x > 0.0,realLabels)))
	return (frr / authentics_number)
	
def computeFAR(realLabels, decisions):
	far = 0
	for i in xrange(len(realLabels)):
		# Choose the real imposters
		if (realLabels[i] < 0):
			# Which were decisionslared authentic users
			if (decisions[i] > 0):
				far += 1
	imposters_number = float(len(filter(lambda x: x < 0.0,realLabels)))
	if (imposters_number == 0):
		return 0.0
	else:
		return (far / imposters_number)
		
def initMultiMissions(missionsPool, subject, subjectTrainingDataFile, subjectUserTestDataFile, subjectImpostorDataFile):
	for svm_type in SVM_TYPES:
		# Cost applies only on C-SVC, epsilon-SVR, and nu-SVR
		if svm_type == "C-SVC":
			costs = COSTS
		else:
			costs = [1]
				
		if (svm_type != "nu-SVC"):
			continue
			
		#Nu applies only to nu-SVC, one-class SVM, and nu-SVR
		if ((svm_type == "nu-SVC") or (svm_type == "one-class SVM")):
			NUs = NU
		else:
			NUs = [0.5]
						
		if (svm_type == "one-class SVM"):
			continue
			
		for kernel_type in KERNEL_TYPES:
			# Linear kernel does not work on that data
			if kernel_type == "linear":
				continue
			# Degree applies only to polynomial kernel type
			if kernel_type == "polynomial":
				degrees = DEGREES
			else:
				degrees = [1]
			
			for gamma in GAMMAS:
				for degree in degrees:	
					for cost in costs:	
						for nu in NUs:
							missionsPool.put({"svm_type" : SVM_TYPES.index(svm_type),
											"kernel_type" : KERNEL_TYPES.index(kernel_type),
											"cost" : cost, 
											"nu" : nu,
											"degree" : degree, 
											"gamma" : gamma, 
											"subject" :subject, 
											"training_data_file" : subjectTrainingDataFile, 
											"user_test_data_file" : subjectUserTestDataFile, 
											"impostor_test_data_file" : subjectImpostorDataFile})
						
				
	return missionsPool
	
if __name__=="__main__":
	if (len(sys.argv) < 2):	
		print '%s input_file' % sys.argv[0]
		sys.exit(1)
	inputFile = sys.argv[1]
	
	plotroc.check_gnuplot_exe()
	
	# Extract all the training, testing and user testing parameters 
	subjectDataFiles = extractSVM(inputFile)
	"""
	# Find the best SVM 
	missionsPool = Queue.Queue(0)
	for subjectData in subjectDataFiles:
		missionsPool = initMultiMissions(missionsPool, subjectData["SUBJECT_NUM"], subjectData["TRAINING_FILE"], subjectData["USER_TEST_FILE"], subjectData["IMPOSTOR_TEST_FILE"])
	SVMThread.startMissions(5, missionsPool)
	
	sys.exit()
	"""
	full_svm_model = {}
	print 'Training...'
	# Create all the binary classifiers (one model per user)
	for subjectData in subjectDataFiles:
		subjectBinaryClassifierModel = TrainSVM(subjectData["SUBJECT_NUM"], subjectData["TRAINING_FILE"])
		full_svm_model.update(subjectBinaryClassifierModel)

	averageFRR = 0.0
	averageFAR = 0.0
	
	# For each of the users, test its remaining data on the trained classifiers to achieve FAR and FRR rate
	for subjectData in subjectDataFiles:
		subjectNum = subjectData["SUBJECT_NUM"]
		subjectUserData = subjectData["USER_TEST_FILE"]
		subjectImposterData = subjectData["IMPOSTOR_TEST_FILE"]
		
		# Compute the score using the imposter and user data for the current subject
		realLabels, decisions = computeSVMScore(subjectNum, subjectUserData, subjectImposterData, full_svm_model)
		# Create the ROC plot for the subject scores
		
		output_title = "Subject %d"  % subjectNum
		output_file = "subject-%d-roc.png" % subjectNum
		plotroc.plot_roc(decisions, realLabels, output_file, output_title)
		
		# Compute FRR - False Reject Rate
		frr = computeFRR(realLabels, decisions)
		averageFRR += frr
		# Compute FAR - False Accept Rate
		far = computeFAR(realLabels, decisions)
		averageFAR += far
		
		print "FAR - %f" % far
		print "FRR - %f" % frr
	
	print "Average FRR = %f" % (averageFRR / len(subjectDataFiles))
	print "Average FAR = %f" % (averageFAR / len(subjectDataFiles))
	
	# Delete all the extracted data
	for subjectData in subjectDataFiles:
		os.unlink(subjectData["TRAINING_FILE"])
		os.unlink(subjectData["TEST_FILE"])
		os.unlink(subjectData["USER_TEST_FILE"])
		os.unlink(subjectData["IMPOSTOR_TEST_FILE"])
	