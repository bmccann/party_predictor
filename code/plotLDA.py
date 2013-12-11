import matplotlib
matplotlib.use('Agg')
import vectorizeFiles as VF
from sklearn.lda import LDA
import numpy as np
import getFileNames as gf
import sys
import scipy
import vectorizeFiles as VF
import getFileNames as gf
import matplotlib.pyplot as plot
import numpy as np





# from feature_extractor import FeatureExtractor


# fe = FeatureExtractor(1)
# featurized = fe.featurizeFiles('../data')
# classNames, repubAndDemMatrix, labels = featurized[:3]
[repubAndDemMatrix,vectorizerRepubDem,labels]=VF.extractWordCounts(True,True,False)
files=gf.getFileNames()
lda = LDA()
lda.fit(repubAndDemMatrix, labels)
transformed = lda.transform(repubAndDemMatrix)
repub=np.array([list(x) for i,x in enumerate(transformed) if labels[i]==1])
dem=np.array([list(x) for i,x in enumerate(transformed) if labels[i]==0])
plot.figure()
plot.scatter(repub[:,0],np.random.rand(len(repub[:,0])),c='r',marker='x')
plot.scatter(dem[:,0],np.random.rand(len(dem[:,0])),c='b',marker='x')
##plot.annotate(s=files[0],xy=transformed[0])
plot.savefig('results/images/VFLDA.png')
# plot.savefig('results/images/PCA.png')

'''
transformedWords=PCA.getPCAMat(repubAndDemMatrix.T, k)
vocab=vectorizerRepubDem.vocabulary_
indicesOfInterest=[]
f=open('wordsInterest.txt','r')
wordsOfInterest=[line.split()[0] for line in f]
print wordsOfInterest
vocab=vectorizerRepubDem.get_feature_names()
for i,word in enumerate(vocab):
	if word in wordsOfInterest:
		indicesOfInterest.append(i)

plot.figure()
for i,word in enumerate(transformedWords):
	if i in indicesOfInterest:
		plot.annotate(s=vectorizerRepubDem.get_feature_names()[i],xy=transformedWords[i])
		plot.scatter(transformedWords[i][0],transformedWords[i][1],c='r',marker='x')
plot.savefig('wordPCA.png')
'''
