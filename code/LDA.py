import vectorizeFiles as VF
from sklearn.lda import LDA
import numpy as np
import getFileNames as gf
import sys
import scipy
from sklearn import grid_search

[repubAndDemMatrix,vectorizerRepubDem,labels]=VF.extractWordCounts(True,True,False)
#labels = 2 * np.ones(len(labels)) - np.array(labels) 
#play around with the number of parameters for LDA! (just enter a number in the brackets "()"; default is none)
#clf = LDA() # <- add number of params here!
#clf.fit(repubAndDemMatrix, labels)

#print('Training error is')
#print(clf.score(repubAndDemMatrix, labels))

totalCorrect = 0
sz = len(labels)

Names = gf.getFileNames()

nonzero = np.nonzero(np.sum(repubAndDemMatrix > 0, axis = 0) == 1)[0]
repubAndDemMatrix = scipy.delete(repubAndDemMatrix, nonzero, 1)


#repubAndDemMatrix = repubAndDemMatrix + abs(np.random.randn(len(repubAndDemMatrix), len(repubAndDemMatrix[0])) / 100.0)


'''
def getLDAMat(trainMat, trainLabels, k):
    lda = LDA(n_components = k)
    lda.fit(trainMat, trainLabels)
    #print(pca.explained_variance_ratio_)
    return lda
'''

'''
parameters = {'n_components':[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
lda = LDA()
print "starting grid search!"
clf = grid_search.GridSearchCV(lda, parameters)
clf.fit(repubAndDemMatrix, labels)
print clf.best_estimator_ #<-lots of detail
print clf.best_params_ #<-more useful
print clf.best_score_ #<-this is the cv error
print clf.score(repubAndDemMatrix, labels) #<-training error
'''

trueDem = 0
trueRep = 0
predDem = 0
predRep = 0

for i in range(len(labels)):
	clf = LDA()
	#trainMat = repubAndDemMatrix
	#trainLabels = labels
	trainMat = np.concatenate((repubAndDemMatrix[0:i],repubAndDemMatrix[i+1:sz]), axis = 0)
	trainLabels = np.concatenate((labels[0:i], labels[i+1:sz]), axis = 0)
	#trainMat = repubAndDemMatrix[0:163]
	#trainLabels = labels[0:163]
	#print type(trainMat)
	#print type(trainLabels)
	#trainLabels = labels[0:i] + labels[i+1:sz]
	clf.fit(trainMat, trainLabels)
	#clf.fit(repubAndDemMatrix, labels)
	#clf = getLDAMat(trainMat, trainLabels, 5);
	if clf.predict([repubAndDemMatrix[i].tolist()]) == labels[i]:
		totalCorrect = totalCorrect + 1
	# print clf.coef_
	print(i)
	predicted = clf.predict([repubAndDemMatrix[i].tolist()])
	print 'predicted =', predicted, '; actual =', labels[i]
	
	if labels[i] == 0:
		trueDem += 1
	else:
		trueRep += 1
	if predicted == 0:
		predDem += 1
	else:
		predRep +=1
	
	if not (predicted == labels[i]):
		print Names[i]
	print(float(totalCorrect) / float(i+1))


print('LOOCV test error is') #0.748466257669
print(float(totalCorrect) / float(len(repubAndDemMatrix)))

print
print('trueDem / predDem =')
print(float(trueDem) / float(predDem))
print
print('trueRep / predRep =')
print(float(trueRep) / float(predRep))
print
print trueDem
print predDem
print trueRep
print predRep