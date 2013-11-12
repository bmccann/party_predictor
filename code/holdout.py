from sklearn.feature_extraction.text import CountVectorizer
from numpy import vstack
from splitter import DataSplitter

class Holdout:
	def __init__(self, dataDir, model=None, maxHoldout=1):
		"""
		@param (dataDir): the directory containing subdirectories for each class of your data
		@param (model): the model to be used during classification
		@param (maxHoldout): the maximum number of examples to hold out during training;
												 classifcation will be run for range(1, maxHoldout)
		"""
		self.model = model
		self.dataDir = dataDir
		self.maxHoldout = maxHoldout

	def prepData(self):
		"""
		prepares data for feature extraction
		"""
		print "\nSplitting Data..."
		ds = DataSplitter(1)
		self.classNames, self.data, self.labels, not_needed1, not_needed2 = ds.splitDir('../data')

	def extractFeatures(self):
		"""
		extract features from the data prepared in self.prepData()
		"""
		assert self.data, "Must prepare data before extracting features"
		print "\nExtracting Features..."
		cv = CountVectorizer(stop_words='english', strip_accents='ascii', charset_error='replace');
		fittedCV = cv.fit_transform(self.data);
		self.featureMatrix = fittedCV.todense()	

	def setModel(self, model):
		self.model = model

	def run(self, normalize=False, binarize=False):
		"""
		runs hold out given the current set of parameters. must have set a model and extracted features
		"""
		assert self.model, "Must set model before running"
		assert self.featureMatrix.any(), "Must extract features before running"

		for numHoldout in range(1, self.maxHoldout + 1):
			print "\n\tRunning Holdout: ", str(numHoldout)

			numExamples = len(self.featureMatrix) - numHoldout
			assert numExamples > 0, "Holding out too many; 0 or fewer examples for training"
			# print "Examples to Train: ", str(numExamples)

			totalScore = 0
			for holdout in range(numExamples):
				finalHoldout = holdout + numHoldout
				# print '...removing example(s): ', range(holdout, finalHoldout)
				holdouts = self.featureMatrix[holdout:finalHoldout]
				holdoutLabels = self.labels[holdout:finalHoldout]
				trainExamples = vstack([self.featureMatrix[:holdout], self.featureMatrix[finalHoldout:]])
				trainLabels = self.labels[:holdout] + self.labels[finalHoldout:]
				
				if normalize:
					Z = 0
					for row in holdouts:
						Z = sum(row)
						holdouts = [entry/Z for entry in row]
					Z = 0
					for row in trainExamples:
						Z = sum(row)
						holdouts = [entry/Z for entry in row]					

				if binarize:
					holdouts = [[entry > 0 for entry in row] for row in holdouts]

				# print '...scoring: ' 
				self.model.fit(trainExamples, trainLabels)	
				totalScore = totalScore + self.model.score(holdouts, holdoutLabels)

			print '\tAverage Score: ', totalScore/numExamples

