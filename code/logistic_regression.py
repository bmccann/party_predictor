import sklearn.linear_model as sk
from holdout import Holdout
from feature_extractor import FeatureExtractor
from collections import Counter

class LogisticRegression:
	"""
	An abstract LogisticRegression model; must retrieve either a scikit learn implementation
	or a custom implementation (for comparison) before using
	"""

	def scikit(self, penalty='l2', C=1):
		"""
		@param (penalty='l2'): Used to specify the norm used in the penalization.
		@param (C=1): Inverse of regularization strength; must be a positive float. 
									Like in support vector machines, smaller values specify 
									stronger regularization.
		@return (model): scikit learn's implementation of LogisticRegression
		"""
		return sk.LogisticRegression(penalty=penalty,C=C)

	def custom(self):
		"""
		@return (model): a custom implementation of LogisticRegression
		"""
		return None

def holdout(penalties, constants, numHoldout):
	holdout = Holdout('../data', maxHoldout=numHoldout)

	for penalty in penalties:
		for C in constants:
			print "\nPenalty, regularization: ", str(penalty), str(C)
			abstractModel = LogisticRegression()
			model = abstractModel.scikit(penalty, C)
			params = (penalty, C)
			holdout.setModel(model, params)
			holdout.run()

def runOnSplit(penalties, constants, split):
	"Running on a " + str(split*100) + '/' + str((1-split)*100) + ' split' 
	fe = FeatureExtractor(split)
	featurized = fe.featurizeFiles('../data')
	classNames = featurized[0]
	trainMatrix, trainLabels = featurized[1:3]
	devMatrix, devLabels = featurized[3:5]
	trainFiles, devFiles = featurized[5:]


	classCounts = Counter()
	for l in devLabels:
		classCounts[l] += 1

	for penalty in penalties:
		for C in constants:
			print "\nPenalty, regularization: ", str(penalty), str(C)

			abstractModel = LogisticRegression()
			model = abstractModel.scikit(penalty, C)
			model_params = (penalty, C)
			model.fit(trainMatrix, trainLabels)

			errors, rankedExamples = Counter(), []

			score = model.score(devMatrix, devLabels)
			predicted_labels = model.predict(devMatrix)

			probs = model.predict_proba(devMatrix)

			for j,pred in enumerate(predicted_labels):
				if not pred == devLabels[j]:
					errors[devLabels[j]] += 1

			for i, p in enumerate(probs):
				rankedExamples.append((p, devFiles[i], predicted_labels[i] == devLabels[i]))		

			results = ''
			for i, c in enumerate(classNames):
				missRate = str(float(errors[i]) / classCounts[i])
				results += '\t' + c + ' error: ' + missRate + '\n'

			results += '\tScore: ' + str(score)
			fileName = 'results/scores/LRsplit'
			for param in model_params:
				fileName += '_' + str(param)
			fileName += '.txt'
			with open(fileName, 'w') as f:
				f.write(results)
			print results

			print '..ranking examples'
			if len(rankedExamples):
				examples = sorted(rankedExamples, key=lambda e: e[0][0])
				fileName = 'results/rankedExamples/LRsplit_' + str(split*100)
				for param in model_params:
					fileName += '_' + str(param)
				fileName += '.txt'
				with open(fileName,'w') as f:
					for e in examples:
						results = e[1]
						results += '\n\t Probability of class '
						results += classNames[0] + ': '
						results += str(e[0][0])
						results += '\n\t Correct: ' + str(e[2])
						f.write(results)


if  __name__ == '__main__':
	"""
	Usage: python logistic_regression.py <penalties> <holdout> <constants>
		where penalties and constants are comma delimited strings
	"""
	penalties, constants, numHoldout, split = ['l1'], [ .1, .3, .5, .7], 34, .7

	holdout(penalties, constants, numHoldout)
	# runOnSplit(penalties, constants, split)


