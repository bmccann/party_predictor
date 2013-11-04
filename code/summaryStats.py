import vectorizeFiles as VF
from sklearn.naive_bayes import MultinomialNB
import random 

##republican data
fR=open('topRepubWords.txt','w')
[repubMatrix,repubVectorizer]=VF.extractWordCounts(True,False,False);
repubColSums=repubMatrix.sum(axis=0)

sortedIndices=[i[0] for i in sorted(enumerate(repubColSums),key=lambda x:-1*repubColSums[x[0]])]
topWords=[word for word in repubVectorizer.get_feature_names()]
topWords=[topWords[ind] for ind in sortedIndices[0:100]]
for word in topWords:
	fR.write(word+'\n')
fR.close();

##democrate data
fD=open('topDemWords.txt','w')

[demMatrix,demVectorizer]=VF.extractWordCounts(False,True,False);
demColSums=demMatrix.sum(axis=0)

sortedIndices=[i[0] for i in sorted(enumerate(demColSums),key=lambda x:-1*demColSums[x[0]])]

topWords=[word for word in demVectorizer.get_feature_names()]
topWords=[topWords[ind] for ind in sortedIndices[0:100]]

for word in topWords:
	fD.write(word+'\n');
fD.close()

[repubAndDemMatrix,vectorizerRepubDem]=VF.extractWordCounts(True,True,False);
totalScore=0
for j in range(1,len(repubAndDemMatrix)):
	trainSetIndices=[]
	testSetIndices=[]
	testMatrix=[]
	trainMatrix=[]
	trainLabels=[]
	testLabels=[]
	for i in range(len(repubAndDemMatrix)):
		if not i==j:
			trainSetIndices.append(i)
			trainMatrix.append(repubAndDemMatrix[i])
			trainLabels.append(i<len(repubMatrix))	
		else:
			testSetIndices.append(i)
			testMatrix.append(repubAndDemMatrix[i])
			testLabels.append(i<len(repubMatrix))	

##labels=[(x<len(repubMatrix)) for x in range(0,len(repubAndDemMatrix))]
	clf= MultinomialNB()
	clf.fit(trainMatrix,trainLabels)
	totalScore=totalScore+clf.score(testMatrix,testLabels)
print totalScore/len(repubAndDemMatrix)

##independent data-no data yet
##indMatrix=VF.extractWordCounts(False,False,True);


