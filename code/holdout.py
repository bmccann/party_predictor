from sklearn.feature_extraction.text import CountVectorizer
import numpy
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
		cv = CountVectorizer(stop_words='english', strip_accents='ascii', charset_error='replace',dtype=numpy.float32,binary=False);
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

		for numHoldout in range(self.maxHoldout, self.maxHoldout + 1):
			print "\n\tRunning Holdout: ", str(numHoldout)

			numExamples = len(self.featureMatrix) - numHoldout
			assert numExamples > 0, "Holding out too many; 0 or fewer examples for training"
			# print "Examples to Train: ", str(numExamples)
			repMisses=0
			demMisses=0
			totalScore = 0
			for i, holdout in enumerate(range(numExamples)):
				if i % numHoldout and i != 0: continue
				print i
				finalHoldout = holdout + numHoldout
				# print '...removing example(s): ', range(holdout, finalHoldout)
				holdouts = self.featureMatrix[holdout:finalHoldout]
				holdoutLabels = self.labels[holdout:finalHoldout]
				trainExamples = vstack([self.featureMatrix[:holdout], self.featureMatrix[finalHoldout:]])
				trainLabels = self.labels[:holdout] + self.labels[finalHoldout:]	
				if normalize:
					Z = 0
					##holdouts=holdouts.T
					for i in range(holdouts.shape[1]):
						Z = numpy.sum(holdouts[:,i])
						avg=float(Z)/holdouts.shape[0]
						if numpy.std(holdouts[:,i]):
							holdouts[:,i]-=avg
							holdouts[:,i]/=numpy.std(holdouts[:,i])
						if not numpy.std(holdouts[:,i]) and Z:
							print numpy.std(holdouts[:,i])
						
							for j in  holdouts[:,i]:
								print j
					##holdouts=holdouts.T	##print numpy.sum(holdouts[i,:])
					Z = 0
					##trainExamples=trainExamples.T
					for i in range(trainExamples.shape[1]):
						Z = numpy.sum(trainExamples[:,i])
						avg=float(Z)/trainExamples.shape[0]
						if not numpy.std(trainExamples[:,i]) and Z:
							print trainExamples[:,i]
							print 'hi'
							print trainExamples.size
						if numpy.std(trainExamples[:,i]):
							trainExamples[:,i]-=avg
							trainExamples[:,i]/=numpy.std(trainExamples[:,i])						
						
							##print numpy.sum(trainExamples[i,:])
					##trainExamples=trainExamples.T
				# print '...scoring: ' 
				self.model.fit(trainExamples, trainLabels)
				predicted_labels=self.model.predict(holdouts)	
				for i,pred in enumerate(predicted_labels):
					if not pred==holdoutLabels[i]:
						if holdoutLabels[i]:
							repMisses+=1
						else:
							demMisses+=1
				totalScore = totalScore + numHoldout*self.model.score(holdouts, holdoutLabels)

			print '\tAverage Score: ', 1-(demMisses+repMisses)/float(numExamples), ' ', totalScore/float(numExamples)
			print '\tMissed ', demMisses, ' Democrats and ',repMisses, " Republicans"
