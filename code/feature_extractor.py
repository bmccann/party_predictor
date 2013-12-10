from random import shuffle
import os, os.path
import numpy
from sklearn.feature_extraction.text import CountVectorizer

class FeatureExtractor():
	def __init__(self, split):
		"""
		Usage example: 
			from feature_extractor import FeatureExtractor 
			fe = FeatureExtractor(split=.7)
			classNames, trainFeatureMatrix, trainLabels, devFeatureMatrix, devLabels = fe.featurizeFiles('../data')

		@param (split): a float between 0 and 1, indicating the fraction of the data you want returned 
		for training (shuffled randomly by default). 

		@return (classNames): the classes in our input data
		@return (trainData): the portion of your data returned for training
		@return (devData): the portion of your data returned for developing your algorithm	
		@return (trainLabels): the labels for your training data in order of example
		@return (devLabels): the labels for your development data in order of example
		"""
		self.split = split

	def featurizeFiles(self, inputDir, type='train'):
		"""
		@param (inputDir): the directory containing subdirectories for each class 
		@param (type='train'): the type of split you want 
						-- 'train' be default returns trainData and devData
		@param (random=True): whether you want a random split
		"""

		# print "Input Directory: ", inputDir
		classDirs = [ os.path.join(inputDir, name) for name in os.listdir(inputDir) \
			if os.path.isdir(os.path.join(inputDir, name)) ]
		classNames = [d.split('/')[-1] for d in classDirs]
		numClasses = len(classDirs)
		# print "Classes: ", numClasses
		# print "Class Names: ", classNames

		# print "Retrieving file names..."
		dataFiles = []
		for d in classDirs:
			labeledFileNames = [ os.path.join(d, name) for name in os.listdir(d) \
				if os.path.isfile(os.path.join(d, name)) and 'txt' in os.path.join(d, name)]
			dataFiles.append(labeledFileNames)

		# print "Splitting Files..."
		trainFilesAndLabels = []
		devFilesAndLabels = []
		for c in range(numClasses):
			numTrainingFiles = int(self.split * len(dataFiles[c]))
			# shuffles this class of examples so we get a random sample from the class
			shuffle(dataFiles[c]) 
		 	trainFilesAndLabels.extend([(example, c) for example in dataFiles[c][:numTrainingFiles]])
		 	devFilesAndLabels.extend([(example, c) for example in dataFiles[c][numTrainingFiles:]])

		# shuffles the train and dev sets of file
		# so that we don't have all of one class, then all the next, etc.
		shuffle(trainFilesAndLabels)
		shuffle(devFilesAndLabels)

		trainFiles, trainLabels = [], []
		for t in trainFilesAndLabels:
			trainFiles.append(t[0])
			trainLabels.append(t[1])

		devFiles, devLabels = [], []
		for t in devFilesAndLabels:
			devFiles.append(t[0])
			devLabels.append(t[1])

		# print "NumTraining: ", len(trainFiles)
		# print "NumDev: ", len(devFiles)

		trainData = []
		devData = []
		for trainFile in trainFiles:
			with open(trainFile) as tf:
				trainData.append(tf.read())

		for devFile in devFiles:
			with open(devFile) as tf:
				devData.append(tf.read())

		numTraining = len(trainData)
		data = trainData + devData

		print "\nExtracting Features..."
		cv = CountVectorizer(stop_words='english', strip_accents='ascii', 
												 charset_error='replace',dtype=numpy.float32,binary=False);
		fittedCV = cv.fit_transform(data);
		featureMatrix = fittedCV.todense()

		return classNames, featureMatrix[:numTraining], trainLabels, featureMatrix[numTraining:], devLabels, trainFiles, devFiles


