import vectorizeFiles as VF
import sklearn.linear_model as sk
import getFileNames as gf
import numpy as np
from feature_extractor import FeatureExtractor
from collections import Counter


# fe = FeatureExtractor(1)
# featurized = fe.featurizeFiles('../data')
# classNames, repubAndDemMatrix, repubAndDemLabels = featurized[:3]
# fileNames = featurized[5]

[repubAndDemMatrix,vectorizerRepubDem,repubAndDemLabels]=VF.extractWordCounts(True,True,False)
fileNames=gf.getFileNames()

rankPres=True
presNames=['obama','truman','reagan','clinton','ford','nixon','lbj','bush1','bush2','carter','jfk']
nameToIdx={}
lgMod=sk.LogisticRegression(penalty='l1',C=.5)

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
# print demOrder
# print repubOrder
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
	sumOfScores = 0
	demMeanRanks = Counter()
	h2hScore = 0
	totalH2H = 0
	for i, presOne in enumerate(presNames):
		for j, presTwo in enumerate(presNames):
			if presOne == presTwo: continue
			if presOne in demOrder and presTwo in demOrder:continue
			if presOne in repubOrder and presTwo in repubOrder:continue

			totalH2H += 1

			presOneIndices = nameToIdx[presOne]
			presTwoIndices = nameToIdx[presTwo]

			presOneRows = []
			presTwoRows = []

			trainRows = [x for x in range(0,repubAndDemMatrix.shape[0]) 
									if x not in presOneIndices and x not in presTwoIndices]
			testRows1 = [x for x in range(0,repubAndDemMatrix.shape[0]) 
									if x in presOneIndices ]
			testRows2 = [x for x in range(0,repubAndDemMatrix.shape[0]) 
									if x in presTwoIndices]

			trainMatrix = repubAndDemMatrix[trainRows,:]
			testMatrix1 = repubAndDemMatrix[testRows1,:]
			testMatrix2 = repubAndDemMatrix[testRows2,:]

			trainLabels=[label for l,label in enumerate(repubAndDemLabels) 
										if l not in presOneIndices and l not in presTwoIndices]

			testLabels1=[label for l,label in enumerate(repubAndDemLabels) 
										if l in presOneIndices]
			testLabels2=[label for l,label in enumerate(repubAndDemLabels) 
										if l in presTwoIndices]

			lgMod.fit(trainMatrix, trainLabels)
			presOnescore = lgMod.score(testMatrix1, testLabels1)
			presTwoscore = lgMod.score(testMatrix2, testLabels2)

			presOneProbsDem = [p[0] for p in lgMod.predict_proba(testMatrix1)]
			presTwoProbsDem = [p[0] for p in lgMod.predict_proba(testMatrix2)]

			avg_prob1 = sum(presOneProbsDem) / float(len(presOneProbsDem))
			avg_prob2 = sum(presTwoProbsDem) / float(len(presTwoProbsDem))

			demMeanRanks[presOne] = (demMeanRanks[presOne] + avg_prob1) / 2.
			demMeanRanks[presTwo] = (demMeanRanks[presTwo] + avg_prob2) / 2.
			
			if avg_prob1 > avg_prob2:
				if presOne in demOrder:
					h2hScore += 1
				# else:
				# 	h2hScore -= 1
				# print presOne, '>', presTwo
			elif avg_prob2 > avg_prob1:
				if presTwo in demOrder:
					h2hScore += 1
				# else:
				# 	h2hScore -= 1
				# print presTwo, '>', presOne
			# demMeanRanks.append((presName, avg_prob))
			# sumOfScores += score
			##print 'demVect', totalScoreVecDem
			# for row in totalScoreVecDem:
			# 	totalScoreDem += float(row[0]>=.5)
			# ##repub
			# totalScoreVecRepub=lgMod.predict_proba(repubAndDemMatrix[nameToIdx[repubOrder[j]],:])
			# totalScoreRepub=0
			# for rowR in totalScoreVecRepub:
			# 	totalScoreRepub+=float(rowR[0]<=.5)
	print "Score", h2hScore/float(totalH2H)
	print demMeanRanks
	# avgScore = sumOfScores / len(presNames)
	# print avgScore
	# demMeanRanks = sorted(demMeanRanks, key=lambda t: t[1], reverse=True)
	# for rank in demMeanRanks:
	# 	print rank
	##for i in nameToIdx[pres]:
	##	trainMat=np.vstack((repubAndDemMatrix[0:i],repubAndDemMatrix[i+1:]))
	##	trainLabels=repubAndDemLabels[0:i]+repubAndDemLabels[i+1:]
	##	lgMod.fit(trainMat,trainLabels)	
	##	totalScore+=lgMod.predict_proba(repubAndDemMatrix[i])[0,0]	
# 	##totalScore/=len(nameToIdx[pres])
# 		totalScoreDem/=len(nameToIdx[demOrder[j]])
# 		totalScoreRepub/=len(nameToIdx[repubOrder[j]])
# 		ranks.append(totalScoreDem)
# 		ranks.append(totalScoreRepub)
# 		print
# 		print totalScoreDem,demOrder[j], "(% dem speeches)"
# 		print np.median(totalScoreVecDem[:, 0]), "(median)"
# 		print np.mean(totalScoreVecDem[:, 0]), "(mean)"
# 		print
# 		print totalScoreRepub,repubOrder[j], "(% dem speeches)"
# 		print np.median(totalScoreVecRepub[:, 0]), "(median)"
# 		print np.mean(totalScoreVecRepub[:, 0]), "(mean)"
# ranks=np.array(ranks)
# ##print ranks
# order=np.argsort(ranks)
##for i in order:
##	if not rankPres:
##		print fileNames[i],ranks[i]
##	else:
##		print presNames[i], ranks[i]
##print order
