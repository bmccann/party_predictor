from sklearn.feature_extraction.text import CountVectorizer
import os
import numpy as np

def getFileNames():

	Names = np.array([])

	os.chdir('../data/republican/')
	for fileName in os.listdir('.'):
		if not fileName == 'README.md':
			Names = np.append(Names, fileName)
	os.chdir('../../code')

	os.chdir('../data/democrat/')
	for fileName in os.listdir('.'):
		if not fileName == 'README.md':
			Names = np.append(Names, fileName)
	os.chdir('../../code')

	##os.chdir('../data/independent/')
	##for fileName in os.listdir('.'):
	##	if not fileName == 'README.md':
	##		Names = np.append(Names, fileName)
	##os.chdir('../../code')
	
	return Names
