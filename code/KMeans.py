#!/usr/bin/python

import vectorizeFiles as VF
from sklearn.cluster import KMeans
import getFileNames as gf
import sys

[repubAndDemMatrix,vectorizer,labels]=VF.extractWordCounts(True,True,False)
totalScore = 0
Names = gf.getFileNames()

def partyKMeans(k):
    clf = KMeans(k)
    clf.fit(repubAndDemMatrix)
    pred = clf.predict(repubAndDemMatrix)
    demSum = 0
    repSum = 0
    for i in range(k):
        print 'Cluster #', i+1
        print
        print len(Names[pred == i])
        print
        print Names[pred == i]
        print
			#         for j in [pred == i]:
			# if labels[j] == 0:
			# 	demSum = demSum + 1
			# if labels[j] == 1:
			# 	repSum = repSum + 1
			#         print 'demSum =', demSum
			#         print 'repSum =', repSum
    return pred

k = int(sys.argv[1])
print 'Performing KMeans with', k, 'clusters.'
print
pred = partyKMeans(k)
print len(Names)
if k == 2:
    print
    score = sum(abs(pred - labels))
    score = max(score, len(labels) - score)
    score = float(score) / float(len(labels))
    print score