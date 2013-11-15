import numpy as np
import vectorizeFiles as VF
import PCA
from sklearn.linear_model import LogisticRegression


clf = LogisticRegression()
[repubAndDemMatrix,vectorizerRepubDem, labels]=VF.extractWordCounts(True,True,False)
k = 10
transformed = PCA.getPCAMat(repubAndDemMatrix, k)
totalCorrect = 0
for i in range(len(transformed)):
    trainMat = np.concatenate((transformed[0:i], transformed[i+1:len(transformed)]), axis = 0)
    trainLabels = np.concatenate((labels[0:i], labels[i+1:len(transformed)]), axis = 0)
    clf.fit(trainMat, trainLabels)
    if clf.predict(transformed[i]) == labels[i]:
        totalCorrect = totalCorrect + 1

print('LOOCV test error is')
print(float(totalCorrect) / float(len(transformed)))