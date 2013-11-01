import vectorizeFiles as VF

##republican data
[repubMatrix,repubVectorizer]=VF.extractWordCounts(True,False,False);
repubColSums=repubMatrix.sum(axis=0)

sortedIndices=[i[0] for i in sorted(enumerate(repubColSums),key=lambda x:-1*repubColSums[x[0]])]
topWords=[word for word in repubVectorizer.get_feature_names()]
topWords=[topWords[ind] for ind in sortedIndices[10:100]]
print topWords

##democrate data
[demMatrix,demVectorizer]=VF.extractWordCounts(False,True,False);

##independent data-no data yet
##indMatrix=VF.extractWordCounts(False,False,True);


