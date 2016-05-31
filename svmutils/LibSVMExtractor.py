import sys
import random

class LibSVMExtractor(object):
	DEFUALT_USER_TRAINING_SESSIONS = 4
	
	def __init__(self, inputFile, userTrainingSessions=0):
		with open(inputFile) as f:
			self.rawData = f.read()
		# Split by rows
		# Remove the first row (column titles)
		self.lineData = self.rawData.split("\n")[1:]
		if (userTrainingSessions == 0):
			self.userTrainingSessions = self.DEFUALT_USER_TRAINING_SESSIONS
		else:
			self.userTrainingSessions = userTrainingSessions

	def extractSubjectNumbers(self):
		uniqueSubjects = []
		
		for line in self.lineData:
			(subjectLabel, sessionIndex, data) = self.parseDataLine(line)
			if uniqueSubjects.count(subjectLabel) == 0:
				uniqueSubjects.append(subjectLabel)
			
		return uniqueSubjects
			
	@staticmethod
	def parseDataLine(line):
		splitrawData = line.split()
		subjectLabel = int(splitrawData[0][1:])
		sessionIndex = int(splitrawData[1])
		
		data = ""
		index = 1
		# Remove meta data
		for datum in splitrawData[3:]:
			data += "%d:%s " % (index, datum)
			index += 1
		data += "\r\n"
		
		return (subjectLabel, sessionIndex, data)
			
	@staticmethod
	def participantLabeling(participantLabel, subjectLabel):
		if (subjectLabel == participantLabel):
			label = 1
		else:
			label = -1
			
		return label
	
	def extract(self, evalSubject, trainingSubjectsNum = 0):
		trainingData = ""
		userTestData1 = ""
		userTestData2 = ""
		impostorTestData = ""

		subjects = self.extractSubjectNumbers()
		subjects.remove(evalSubject)
		
		if (trainingSubjectsNum == 0):
			trainingSubjectsNum = len(subjects)
			
		trainingSubjects = random.sample(subjects, min(len(subjects), trainingSubjectsNum))
		
		for line in self.lineData:
			(subjectLabel, sessionIndex, data) = self.parseDataLine(line)
			
			label = self.participantLabeling(evalSubject, subjectLabel)
			
			lineData = "%d %s" % (label, data)

			# The training matrix is the first 100 password repetitions for the subject, corresponding to the first 2 sessions of passwords.
			if ((sessionIndex in [1,2,3,4]) 
				and ((evalSubject == subjectLabel) or (subjectLabel in trainingSubjects))):
				trainingData += lineData
			# The user scoring matrix is the last 300 password repetitions
			elif ((sessionIndex in [5,6]) and (evalSubject == subjectLabel)):
				userTestData1 += lineData
			# The user scoring matrix is the last 300 password repetitions
			elif ((sessionIndex in [7,8]) and (evalSubject == subjectLabel)):
				userTestData2 += lineData
			# The impostor scoring matrix is the first 5 repetitions from all the other subjects.
			elif ((sessionIndex > 5) and (evalSubject != subjectLabel)):
				impostorTestData += lineData

		return (trainingData, userTestData1, userTestData2, impostorTestData)
	

if (__name__=="__main__"):
	if (len(sys.argv) < 2):	
		print '%s input_file [first class number]' % sys.argv[0]
		sys.exit(1)
	
	inputFile = sys.argv[1]
	extractor = LibSVMExtractor(inputFile)
	# Extract the options for the different subjects
	subjects = extractor.extractSubjectNumbers()
	if (len(sys.argv) == 3):
		randomParticipant = int(sys.argv[2])
		if (not randomParticipant in subjects):
			print 'Unexisting participant!'
			sys.exit(1)
	else:
		randomParticipant = subjects[random.randint(1,len(subjects))]
	print "Choosing %d as the first class" % randomParticipant
	output_training = inputFile + ".%d.training" %  randomParticipant
	output_user_test = inputFile + ".%d.user_test" % randomParticipant
	output_impostor_test = inputFile + ".%d.impostor_test" % randomParticipant
	output_test = inputFile + "%d.test" % randomParticipant
	
	(trainingData, userTestData, impostorTestData) = extractor.extract(randomParticipant)

	with open(output_training, "wb") as outTrain:
		outTrain.write(trainingData)
	
	with open(output_user_test, "wb") as outTest:
		outTest.write(userTestData)
		
	with open(output_impostor_test, "wb") as outTest:
		outTest.write(impostorTestData)
    
	with open(output_test, "wb") as outTest:
		outTest.write(impostorTestData)
		outTest.write(userTestData)
