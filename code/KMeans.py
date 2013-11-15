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
    for i in range(k):
        print 'Cluster #', i+1
        print
        print len(Names[pred == i])
        print
        print Names[pred == i]
        print

k = int(sys.argv[1])
print 'Performing KMeans with', k, 'clusters.'
print
partyKMeans(k)
print len(Names)
