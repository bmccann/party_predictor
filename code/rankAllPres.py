import vectorizeFiles as VF
import sklearn.linear_model as sk
import getFileNames as gf
import numpy as np

rankPres=True
[repubAndDemMatrix,repubAndDemVectorizer,repubAndDemLabels]=VF.extractWordCounts(True,True,False)
presNames=['obama','truman','reagan','clinton','ford','nixon','lbj','bush1','bush2','carter','jfk']
nameToIdx={}
lgMod=sk.LogisticRegression(penalty='l1',C=.1)
fileNames=gf.getFileNames()
print len(fileNames)
print repubAndDemMatrix.shape
for i,fileName in enumerate(fileNames):
	fileName=fileName.lower()
	for name in presNames:
		if name in fileName:
			if name in nameToIdx.keys():
				nameToIdx[name].append(i)		
			else:
				nameToIdx[name]=[i];
repubOrder=[]
demOrder=[]
sortPres=sorted(presNames,key=lambda pres:-len(nameToIdx[pres]))
for pres in sortPres:
	if pres in ['obama','truman','clinton','lbj','carter','jfk']:
		demOrder.append(pres)
	else:
		repubOrder.append(pres)
print demOrder
print repubOrder
##for name in presNames:
##	print len(nameToIdx[name]), name

ranks=[]
if not rankPres:
	for i,speech in enumerate(repubAndDemMatrix):
		trainMat=np.vstack((repubAndDemMatrix[0:i],repubAndDemMatrix[i+1:]))
		trainLabels=repubAndDemLabels[0:i]+repubAndDemLabels[i+1:]
		lgMod.fit(trainMat,trainLabels)
		ranks.append(lgMod.predict_proba(repubAndDemMatrix[i])[0,0])
else:
	for j,pres in enumerate(presNames):
		if j<len(presNames)/2:
			totalScore=0;
			trainRows=[x for x in range(0,repubAndDemMatrix.shape[0]) if ((x not in nameToIdx[demOrder[j]]) and (x not in nameToIdx[repubOrder[j]])) ]
			trainMat=repubAndDemMatrix[trainRows,:]
			trainLabels=[label for i,label in enumerate(repubAndDemLabels) if ((i not in nameToIdx[demOrder[j]]) and (i not in nameToIdx[repubOrder[j]]))]
			lgMod.fit(trainMat,trainLabels)
				
			##Dem
			totalScoreVecDem=lgMod.predict_proba(repubAndDemMatrix[nameToIdx[demOrder[j]],:])
			totalScoreDem=0
			##print 'demVect', totalScoreVecDem
			for row in totalScoreVecDem:
				totalScoreDem+=float(row[0]>=.5)
			##repub
			totalScoreVecRepub=lgMod.predict_proba(repubAndDemMatrix[nameToIdx[repubOrder[j]],:])
			totalScoreRepub=0
			for rowR in totalScoreVecRepub:
				totalScoreRepub+=float(rowR[0]<=.5)
			
			
		##for i in nameToIdx[pres]:
		##	trainMat=np.vstack((repubAndDemMatrix[0:i],repubAndDemMatrix[i+1:]))
		##	trainLabels=repubAndDemLabels[0:i]+repubAndDemLabels[i+1:]
		##	lgMod.fit(trainMat,trainLabels)	
		##	totalScore+=lgMod.predict_proba(repubAndDemMatrix[i])[0,0]	
		##totalScore/=len(nameToIdx[pres])
			totalScoreDem/=len(nameToIdx[demOrder[j]])
			totalScoreRepub/=len(nameToIdx[repubOrder[j]])
			ranks.append(totalScoreDem)
			ranks.append(totalScoreRepub)
			print
			print totalScoreDem,demOrder[j], "(% dem speeches)"
			print np.median(totalScoreVecDem[:, 0]), "(median)"
			print np.mean(totalScoreVecDem[:, 0]), "(mean)"
			print
			print totalScoreRepub,repubOrder[j], "(% dem speeches)"
			print np.median(totalScoreVecRepub[:, 0]), "(median)"
			print np.mean(totalScoreVecRepub[:, 0]), "(mean)"
ranks=np.array(ranks)
##print ranks
order=np.argsort(ranks)
##for i in order:
##	if not rankPres:
##		print fileNames[i],ranks[i]
##	else:
##		print presNames[i], ranks[i]
##print order
