from numpy import vstack
from feature_extractor import FeatureExtractor
from collections import Counter

class Holdout:
	def __init__(self, dataDir, model=None, maxHoldout=1):
		"""
		@param (dataDir): the directory containing subdirectories 
											for each class of your data
		@param (model): the model to be used during classification
		@param (maxHoldout): the maximum number of examples 
												 to hold out during training;
												 classifcation will be run for range(1, maxHoldout)
		"""
		self.model = model
		self.dataDir = dataDir
		self.maxHoldout = maxHoldout

		fe = FeatureExtractor(1)
		featurized = fe.featurizeFiles('../data')
		self.classNames, self.featureMatrix, self.labels = featurized[:3]
		self.fileNames = featurized[5]

		self.classCounts = Counter()
		for l in self.labels:
			self.classCounts[l] += 1

	def setModel(self, model, params):
		self.model = model
		self.model_params = params

	def run(self):
		
		"""
		runs hold out given the current set of parameters. 
		must have set a model and extracted features
		"""
		assert self.model, "Must set model before running"
		assert self.featureMatrix.any(), "Must extract features before running"

		for numHoldout in range(self.maxHoldout, self.maxHoldout + 1):
			print "\n\tRunning Holdout: ", str(numHoldout)

			numRounds = float(len(self.featureMatrix) / numHoldout)
			# assert numRounds > 0, "Holding out too many; 0 examples for training"

			# print "Examples to Train: ", str(numExamples)
			errors, sumOfScores, rankedExamples = Counter(), 0, []

			for i in range(int(numRounds)):
				print "Round", str(i + 1)
				holdout = i * numHoldout
				finalHoldout = holdout + numHoldout
				# print '...removing example(s): ', range(holdout, finalHoldout)
				holdouts = self.featureMatrix[holdout:finalHoldout]
				holdoutLabels = self.labels[holdout:finalHoldout]
				trainExamples = vstack([self.featureMatrix[:holdout], 
																self.featureMatrix[finalHoldout:]])
				trainLabels = self.labels[:holdout] + self.labels[finalHoldout:]							

				# print '...scoring: ' 
				self.model.fit(trainExamples, trainLabels)
				currScore = self.model.score(holdouts, holdoutLabels)
				sumOfScores += currScore
				

				# print '...calculating details'
				predicted_labels = self.model.predict(holdouts)

				for j,pred in enumerate(predicted_labels):
					if not pred == holdoutLabels[j]:
						errors[holdoutLabels[j]] += 1


				if numHoldout == 1: 
					rankedExamples.append((self.model.predict_proba(holdouts)[0], 
																	self.fileNames[i], 
																	predicted_labels == self.labels[i]))		
			results = ''
			for i, c in enumerate(self.classNames):
				missRate = str(float(errors[i]) / self.classCounts[i])
				results += '\t' + c + ' error: ' + missRate + '\n'

			results += '\tAverage Score: ' + str(sumOfScores / numRounds)
			fileName = 'results/scores/LRholdout_' + str(numHoldout)
			for param in self.model_params:
				fileName += '_' + str(param)
			fileName += '.txt'
			with open(fileName, 'w') as f:
				f.write(results)
			print results


			print '..ranking examples'
			if len(rankedExamples):
				#most democrat will appear first in file
				examples = sorted(rankedExamples, key=lambda e: e[0][1]) 
				fileName = 'results/rankedExamples/LRholdout_' + str(numHoldout)
				for param in self.model_params:
					fileName += '_' + str(param)
				fileName += '.txt'
				with open(fileName,'w') as f:
					for e in examples:
						results = e[1]
						results += '\n\t Probability of class '
						results += self.classNames[0] + ': '
						results += str(e[0][0])
						results += '\n\t Correct: ' + str(e[2]) + '\n'
						f.write(results)
