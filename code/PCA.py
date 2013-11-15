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

<<<<<<< HEAD
#[repubAndDemMatrix,vectorizerRepubDem, labels]=VF.extractWordCounts(True,True,False)

# The first k principal components
def getPCAMat(repubAndDemMatrix, k):
    pca = PCA(n_components = k)
    pca.fit(repubAndDemMatrix)
    #print(pca.explained_variance_ratio_)
    return pca.transform(repubAndDemMatrix)

=======


# The first k principal components
# Usage: [repubAndDemMatrix,vectorizerRepubDem, labels]=VF.extractWordCounts(True,True,False)
#        transformed = PCA.getPCAMat(repubAndDemMatrix)

def getPCAMat(repubAndDemMatrix, k):
    pca = PCA(n_components = k)
    pca.fit(repubAndDemMatrix)
    print(pca.explained_variance_ratio_)
    return pca.transform(repubAndDemMatrix)
>>>>>>> 9a03f806d60fef6d66299e092ccc42c942f24a02
