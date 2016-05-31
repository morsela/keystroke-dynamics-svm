import sys
import os
import time
from subprocess import *
import threading
import thread
import Queue

svmtrain_exe = r"svm-train.exe"
svmpredict_exe = r"svm-predict.exe"

SVM_TYPES = ["C-SVC", "nu-SVC"]# , "one-class SVM", "epsilon-SVR", "nu-SVR"]
KERNEL_TYPES = ["linear", "polynomial", "radial basis function", "sigmoid"]

class SVMThread(threading.Thread):
	def __init__(self, baseDirPath, missionsPool, resultsQueue):
		self.baseDirPath = baseDirPath
		self.missionsPool = missionsPool
		self.resultsQueue = resultsQueue
		threading.Thread.__init__(self)
		
	def run(self):
		self.predict_test_file = os.path.join(self.baseDirPath, "Temp", "predict.%d" % thread.get_ident())
		self.model_file = os.path.join(self.baseDirPath, "Temp", "model.%d" % thread.get_ident())
		
		while not self.missionsPool.empty():
			mission = self.missionsPool.get(block=False)
			if (mission != None):
				svm_type = mission["svm_type"]
				kernel_type = mission["kernel_type"]
				cost = mission["cost"]
				nu = mission["nu"]
				degree = mission["degree"]
				gamma = mission["gamma"]
				subject = mission["subject"]
				training_data_file = mission["training_data_file"]
				user_test_data_file = mission["user_test_data_file"]
				impostor_test_data_file = mission["impostor_test_data_file"]

				cmd = '%s -c %d -s %d -n %f -t %d -d %d -g %d "%s" "%s"' % (svmtrain_exe,cost, svm_type,nu,kernel_type, degree, gamma, training_data_file,self.model_file)
				
				(stdoutdata, stderrdata) = Popen(cmd, shell = True, stdout = PIPE,stderr=PIPE).communicate()

				if (stderrdata.find("Error") != -1):
					continue
	
				# Predict using the model on the user data
				cmd = '%s "%s" "%s" "%s"' % (svmpredict_exe, user_test_data_file, self.model_file, self.predict_test_file)
				(stdoutUserdata, stderrUserdata) = Popen(cmd, shell = True, stdout=PIPE, stderr=PIPE).communicate()	
				# Predict using the model on the impostor data
				cmd = '%s "%s" "%s" "%s"' % (svmpredict_exe, impostor_test_data_file, self.model_file, self.predict_test_file)
				(stdoutImpostordata, stderrImpostordata) = Popen(cmd, shell = True, stdout=PIPE, stderr=PIPE).communicate()	
				
				if (os.path.exists(self.predict_test_file)):
					os.remove(self.predict_test_file)
				if (os.path.exists(self.model_file)):
					os.remove(self.model_file)
				
				if len(stdoutdata) != 0:
					result = '[Subject #%d] [User:%s] [Impostor:%s] (SVM Type: %s, Kernel Type: %s, Cost: %d, Degree: %d, NU: %f, Gamma: %d)' % \
						(subject, stdoutUserdata[:-2], stdoutImpostordata[:-2], SVM_TYPES[svm_type], KERNEL_TYPES[kernel_type], cost, degree, nu, gamma)
					self.resultsQueue.put(result)
					
def startMissions(threadsNum, missionsPool):
	baseDirPath = os.path.dirname(__file__)
	currentTime = time.strftime("%d%m%y_%H-%M-%S")
	resultFile = open(os.path.join(baseDirPath, "result-%s.txt" % currentTime),"wb")
	resultsQueue = Queue.Queue(0)

	if not os.path.exists(os.path.join(baseDirPath, "Temp")):
		os.mkdir(os.path.join(baseDirPath, "Temp"))
		
	threads = []
	for i in xrange(threadsNum):
		threads.append(SVMThread(baseDirPath, missionsPool, resultsQueue))
		threads[i].start()

	while (not missionsPool.empty()):
		if (not resultsQueue.empty()):
			result =  resultsQueue.get(block=False)
			print result
			resultFile.write("%s\r\n" % result)
			resultFile.flush()
		time.sleep(1)

	for i in xrange(threadsNum):	
		threads[i].join()
			
	while (not resultsQueue.empty()):
		result = resultsQueue.get(block=False)
		print result
		resultFile.write("%s\r\n" % result)
		resultFile.flush()

	resultFile.close()
	