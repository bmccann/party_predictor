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

sortedindices=[i[0] for i in sorted(enumerate(demColSums),key=lambda x:-1*demColSums[x[0]])]

topwords=[word for word in demVectorizer.get_feature_names()]
topwords=[topwords[ind] for ind in sortedindices[0:100]]

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
			if i<len(repubMatrix):
				trainLabels.append(1)
			else:
				trainLabels.append(0)	
		else:
			testSetIndices.append(i)
			testMatrix.append(repubAndDemMatrix[i])
			if i<len(repubMatrix):
				testLabels.append(1)
			else:
				testLabels.append(0)	
##labels=[(x<len(repubMatrix)) for x in range(0,len(repubAndDemMatrix))]
	print trainLabels;
	clf= MultinomialNB()
	clf.fit(trainMatrix,trainLabels)
	print clf.classes_
	coefficientsRep=clf.feature_log_prob_[1]
	coefficientsDem=clf.feature_log_prob_[0]
	##print repubAndDemMatrix.shape
	##print coefficientsRep.shape
	##print coefficientsRep[1][0]
	sortedindices=[i[0] for i in sorted(enumerate(coefficientsRep),key=lambda x:1*(coefficientsDem[x[0]]-coefficientsRep[x[0]]))]	
	topwords=[word for word in vectorizerRepubDem.get_feature_names()]
	topwords=[topwords[ind] for ind in sortedindices[0:10]]
	print topwords
	##print clf.predict_proba(testMatrix)
	totalScore=totalScore+clf.score(testMatrix,testLabels)
print totalScore/len(repubAndDemMatrix)

##independent data-no data yet
##indMatrix=VF.extractWordCounts(False,False,True);


