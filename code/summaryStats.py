import vectorizeFiles as VF

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

##independent data-no data yet
##indMatrix=VF.extractWordCounts(False,False,True);


