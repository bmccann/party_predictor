# import vectorizeFiles as VF
from sklearn.neighbors import KNeighborsClassifier#, DistanceMetric
# import numpy as np
# import getFileNames as gf
# import sys
# import scipy
from sklearn import grid_search
from feature_extractor import FeatureExtractor


fe = FeatureExtractor(1)
featurized = fe.featurizeFiles('../data')
classNames, repubAndDemMatrix, labels = featurized[:3]
# [repubAndDemMatrix,vectorizerRepubDem,labels]=VF.extractWordCounts(True,True,False)
parameters = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
#,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}
#'weights':('uniform','distance'), 'p':[1, 2, 3, 4, 5]
#'metric':('euclidean', 'manhattan','chebyshev','minkowski','jaccard','maching','dice','kulsinki','rogerstanimoto','russellrao','sokalmichener','sokalsneath'), 
kn = KNeighborsClassifier()
clf = grid_search.GridSearchCV(kn, parameters)
clf.fit(repubAndDemMatrix, labels)
print clf.best_estimator_ #<-lots of detail
print clf.best_params_ #<-more useful
print clf.best_score_ #<-this is the cv error
print clf.score(repubAndDemMatrix, labels) #<-training error


#optimal parameter of 4 neighbors, best test error is 0.668573607933, best training error is 0.828488372093

# if we use shuffles the training data so that it is not all democrats and then all republicans,
# we get an optimal param of 1 neighbor, .689 test error, 1.0 training error
