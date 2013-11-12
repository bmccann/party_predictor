#!/usr/bin/python

import vectorizeFiles as VF
from sklearn.decomposition import PCA
import getFileNames as gf
import sys

'''
PCA projects the data onto a set of principal component axes that best explain the 
OVERALL variance in the data; the first principal component explains the most variance 
in the data, the second explains the second most variance in the data, etc.
'''

[repubAndDemMatrix,vectorizerRepubDem, labels]=VF.extractWordCounts(True,True,False)
Names = gf.getFileNames()

# The first 2 principal components explain significantly more variance than the others
pca = PCA(n_components = 2)
pca.fit(repubAndDemMatrix)
print(pca.explained_variance_ratio_)

reducedFeatureMatrix = pca.transform(repubAndDemMatrix)