#!/usr/bin/python

import vectorizeFiles as VF
from sklearn.cluster import KMeans
import getFileNames as gf
import sys

[repubAndDemMatrix,vectorizerRepubDem]=VF.extractWordCounts(True,True,False)
totalScore = 0
Names = gf.getFileNames()

def partyKMeans(k):
    clf = KMeans(k)
    clf.fit(repubAndDemMatrix)
    pred = clf.predict(repubAndDemMatrix)
    for i in range(k):
        print 'Cluster #', i+1
        print
        print Names[pred == i]
        print

k = int(sys.argv[1])
print 'Performing KMeans with', k, 'clusters.'
print
partyKMeans(k)


# for j in range(len(repubAndDemMatrix)):
#     ##print('training, removing example number', j)
#     trainSetIndices=[]
#     testSetIndices=[]
#     testMatrix=[]
#     trainMatrix=[]
#     trainLabels=[]
#     testLabels=[]
#     for i in range(len(repubAndDemMatrix)):
#         if not i==j:
#             trainSetIndices.append(i)
#             trainMatrix.append(repubAndDemMatrix[i])
#             if i<len(repubMatrix):
#                 trainLabels.append(1)
#             else:
#                 trainLabels.append(0)    
#         else:
#             testSetIndices.append(i)
#             testMatrix.append(repubAndDemMatrix[i])
#             if i<len(repubMatrix):
#                 testLabels.append(1)
#             else:
#                 testLabels.append(0)    
#     clf= MultinomialNB()
#     clf.fit(trainMatrix,trainLabels)
#     totalScore=totalScore+clf.score(testMatrix,testLabels)
# trainLabelsWhole=[]
# for j in range(len(repubAndDemMatrix)):
#     if j<len(repubMatrix):
#         trainLabelsWhole.append(1)
#     else:
#         trainLabelsWhole.append(0)        
# clfT=MultinomialNB()
# clfT.fit(repubAndDemMatrix,trainLabelsWhole)
# coefficientsRep=clf.feature_log_prob_[1]
# coefficientsDem=clf.feature_log_prob_[0]


# print totalScore/len(repubAndDemMatrix)

