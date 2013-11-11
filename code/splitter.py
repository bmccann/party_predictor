from random import shuffle
import os, os.path

class DataSplitter():
	def __init__(self, split):
		"""
		ex: 
			from splitter import DataSplitter 
			ds = DataSplitter(split=.7)
			trainData, devData = ds.splitDir(inputDir='../data')

		@split: a number between 0 and 1, indicating the fraction of the data you want returned 
		for training (shuffled randomly by default). 

		@return classes: the classes in our input data
		@return trainData: the portion of your data returned for training
		@return devData: the portion of your data returned for developing your algorithm	
		"""
		self.split = split

	def splitDir(self, inputDir, type='train', randomize=True):
		"""
		@inputDir: the directory containing subdirectories for each class 
		@type: the type of split you want 
						-- 'train' be default returns trainData and devData
		@random: whether you want a random split
		"""

		print "Input Directory: ", inputDir
		# retrieving the classes, and their respective directories
		classDirs = [ os.path.join(inputDir, name) for name in os.listdir(inputDir) \
			if os.path.isdir(os.path.join(inputDir, name)) ]
		classNames = [d.split('/')[-1] for d in classDirs]
		numClasses = len(classDirs)
		print "Classes: ", numClasses
		print "Class Names: ", classNames

		print "Retrieving file names..."
		dataFiles = []
		for d in classDirs:
			fileNames = [ os.path.join(d, name) for name in os.listdir(d) \
				if os.path.isfile(os.path.join(d, name)) and 'txt' in os.path.join(d, name)]
			dataFiles.append(fileNames)

		print "Splitting Files..."
		trainFilesAndLabels = []
		devFilesAndLabels = []
		for c in range(numClasses):
			numTrainingFiles = int(self.split * len(dataFiles[c]))
		 	trainFilesAndLabels.extend([(example, c) for example in dataFiles[c][:numTrainingFiles-1]])
		 	devFilesAndLabels.extend([(example, c) for example in dataFiles[c][numTrainingFiles-1:]])

		if randomize: shuffle(trainFilesAndLabels)
		if randomize: shuffle(devFilesAndLabels)

		trainFiles = [t[0] for t in trainFilesAndLabels]
		devFiles = [t[0] for t in devFilesAndLabels]
		trainLabels = [t[1] for t in trainFilesAndLabels]
		devLabels = [t[1] for t in devFilesAndLabels]

		print "NumTraining: ", len(trainFiles)
		print "NumDev: ", len(devFiles)

		trainData = []
		devData = []
		for trainFile in trainFiles:
			with open(trainFile) as tf:
				trainData.append(tf.read())
		for devFile in devFiles:
			with open(devFile) as tf:
				devData.append(tf.read())

		return classNames, trainData, devData, trainLabels, devLabels


