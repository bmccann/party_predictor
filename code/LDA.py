import vectorizeFiles as VF
from sklearn.lda import LDA
import numpy as np
import getFileNames as gf
import sys

[repubAndDemMatrix,vectorizerRepubDem,labels]=VF.extractWordCounts(True,True,False)
labels = 2 * np.ones(len(labels)) - np.array(labels) 
#play around with the number of parameters for LDA! (just enter a number in the brackets "()"; default is none)
#clf = LDA() # <- add number of params here!
#clf.fit(repubAndDemMatrix, labels)

#print('Training error is')
#print(clf.score(repubAndDemMatrix, labels))

totalCorrect = 0
sz = len(labels)

'''
def getLDAMat(trainMat, trainLabels, k):
    lda = LDA(n_components = k)
    lda.fit(trainMat, trainLabels)
    #print(pca.explained_variance_ratio_)
    return lda
'''

for i in range(len(labels)):
	clf = LDA()
	
	#trainMat = repubAndDemMatrix
	#trainLabels = labels
	
	trainMat = np.concatenate((repubAndDemMatrix[0:i],repubAndDemMatrix[i+1:sz]), axis = 0)
	trainLabels = np.concatenate((labels[0:i], labels[i+1:sz]), axis = 0)
	#trainLabels = labels[0:i] + labels[i+1:sz]
	
	
	clf.fit(trainMat, trainLabels)
	#clf.fit(repubAndDemMatrix, labels)
	#clf = getLDAMat(trainMat, trainLabels, 5);
	if clf.predict([repubAndDemMatrix[i].tolist()]) == labels[i]:
		totalCorrect = totalCorrect + 1
	print 'predicted =', clf.predict([repubAndDemMatrix[i].tolist()]), '; acutal =', labels[i]
	print(i)
	print(float(totalCorrect) / float(i+1))


print('LOOCV test error is')
print(float(totalCorrect) / float(len(repubAndDemMatrix)))



