import vectorizeFiles as VF
from sklearn.linear_model import LogisticRegression
[RepAndDemMatrix,RepAndDemVectorizer,RepAndDemLabels]=VF.extractWordCounts(True,True,False);
totalScore=0;
for j in range(len(RepAndDemMatrix)):
	print('training, removing example number', j)
	trainSetIndices=[]
	testSetIndices=[]
	testMatrix=[]
	trainMatrix=[]
	trainLabels=[]
	testLabels=[]
	for i in range(len(RepAndDemMatrix)):
		rowSum=RepAndDemMatrix[i].sum()	
		if not i==j:
			trainSetIndices.append(i)
			entry = [float(e)/float(rowSum) for e in RepAndDemMatrix[i]]	
			trainMatrix.append(entry)
			trainLabels.append(RepAndDemLabels[i])	
		else:
			testSetIndices.append(i)
			entry = [float(e)/float(rowSum) for e in RepAndDemMatrix[i]]
			testMatrix.append(entry)
			testLabels.append(RepAndDemLabels[i])				
	logReg=LogisticRegression(penalty='l1',C=1)
	logReg.fit(trainMatrix,trainLabels)	
	totalScore=totalScore+logReg.score(testMatrix,testLabels)

print totalScore/len(RepAndDemMatrix)
