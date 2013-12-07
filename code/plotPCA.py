import PCA as PCA
import vectorizeFiles as VF
import getFileNames as gf
import matplotlib.pyplot as plot
import numpy as np
[repubAndDemMatrix,vectorizerRepubDem, labels]=VF.extractWordCounts(True,True,False)
k = 2
files=gf.getFileNames()
##transformed = PCA.getPCAMat(repubAndDemMatrix, k)
##repub=np.array([list(x) for i,x in enumerate(transformed) if labels[i]==1])
##dem=np.array([list(x) for i,x in enumerate(transformed) if labels[i]==0])
##plot.scatter(repub[:,0],repub[:,1],c='r',marker='x')
##plot.scatter(dem[:,0],dem[:,1],c='b',marker='x')
##plot.annotate(s=files[0],xy=transformed[0])
##plot.savefig('docPCA.png')


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

