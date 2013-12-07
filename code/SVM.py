import numpy as np
import vectorizeFiles as VF
#from sklearn.svm import LinearSVC <- different then a linear kernel
from sklearn import svm, grid_search

[repubAndDemMatrix,vectorizerRepubDem, labels]=VF.extractWordCounts(True,True,False)
parameters = {'kernel':('linear','rbf','poly','sigmoid'), 'C':[1,2,3,4,5,6,7,8,9,10], 'gamma':[1.0]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(repubAndDemMatrix, labels)
print clf.best_estimator_ #<-lots of detail
print clf.best_params_ #<-more useful
print clf.best_score_ #<-this is the cv error
print clf.score(repubAndDemMatrix, labels) #<-training error


# Below is manually created leave-one-out cross validation testing, however the code above is better and faster
#
# for C in xrange(1, 11): #xrange(1,51)
#     clf1 = svm.SVC(C=C, kernel='linear')
#     clf2 = svm.SVC(C=C, kernel='poly')
#     clf3 = svm.SVC(C=C, kernel='rbf')
#     clf4 = svm.SVC(C=C, kernel='sigmoid')
#     totalCorrect1 = 0
#     totalCorrect2 = 0
#     totalCorrect3 = 0
#     totalCorrect4 = 0
#     for i in range(len(labels)):
#         trainMat = np.concatenate((repubAndDemMatrix[0:i], repubAndDemMatrix[i+1:len(labels)]), axis = 0)
#         trainLabels = np.concatenate((labels[0:i], labels[i+1:len(labels)]), axis = 0)
#         clf1.fit(trainMat, trainLabels)
#         clf2.fit(trainMat, trainLabels)
#         clf3.fit(trainMat, trainLabels)
#         clf4.fit(trainMat, trainLabels)
#         if clf1.predict(repubAndDemMatrix[i]) == labels[i]:
#             totalCorrect1 = totalCorrect1 + 1
#         if clf2.predict(repubAndDemMatrix[i]) == labels[i]:
#             totalCorrect2 = totalCorrect2 + 1
#         if clf3.predict(repubAndDemMatrix[i]) == labels[i]:
#             totalCorrect3 = totalCorrect3 + 1
#         if clf4.predict(repubAndDemMatrix[i]) == labels[i]:
#             totalCorrect4 = totalCorrect4 + 1
#     print 'LOOCV accuracy (C =', C, ', linear)' 'is', float(totalCorrect1) / float(len(labels))
#     print 'LOOCV accuracy (C =', C, ', poly)' 'is', float(totalCorrect2) / float(len(labels))
#     print 'LOOCV accuracy (C =', C, ', rbf)' 'is', float(totalCorrect3) / float(len(labels))
#     print 'LOOCV accuracy (C =', C, ', sigmoid)' 'is', float(totalCorrect4) / float(len(labels))

