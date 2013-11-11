import sklearn.linear_model as sk
from holdout import Holdout
import sys

class LogisticRegression:
	"""
	An abstract LogisticRegression model; must retrieve either a scikit learn implementation
	or a custom implementation before using
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

def main(penalties, constants):
	holdout = Holdout('../data')
	holdout.prepData()
	holdout.extractFeatures()

	for penalty in penalties:
		for C in constants:
			print "\nPenalty, regularization: ", str(penalty), str(C)
			abstractModel = LogisticRegression()
			model = abstractModel.scikit(penalty, C)
			holdout.setModel(model)
			holdout.run()


if  __name__ =='__main__':
	"""
	Usage: python logistic_regression.py <penalties constants>
		where penalties and constants are comma delimited strings
	"""
	penalties, constants = ['l1', 'l2'], [.001, .1, .5, 1]

	if len(sys.argv) > 1:
		penalties = sys.argv[1].split(',')
	if len(sys.argv) > 2:
		constants = sys.argv[2].split(',')

	main(penalties, constants)

