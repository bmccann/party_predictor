import numpy as np
import vectorizeFiles as VF
import PCA
from sklearn.linear_model import LogisticRegression


clf = LogisticRegression('l1', False, 1000.0)
[repubAndDemMatrix,vectorizerRepubDem, labels]=VF.extractWordCounts(True,True,False)
for k in xrange(1, 51):
    transformed = PCA.getPCAMat(repubAndDemMatrix, k)
    totalCorrect = 0
    for i in range(len(transformed)):
        trainMat = np.concatenate((transformed[0:i], transformed[i+1:len(transformed)]), axis = 0)
        trainLabels = np.concatenate((labels[0:i], labels[i+1:len(transformed)]), axis = 0)
        clf.fit(trainMat, trainLabels)
        if clf.predict(transformed[i]) == labels[i]:
            totalCorrect = totalCorrect + 1
    print 'LOOCV accuracy when k =', k, 'is', float(totalCorrect) / float(len(transformed))
    
# for k in xrange(1, 31):
#     transformed = PCA.getPCAMat(repubAndDemMatrix, 27)
#     trainMat = np.concatenate((transformed[0:40], transformed[70:len(transformed)]), axis = 0)
#     trainLabels = np.concatenate((labels[0:40], labels[70:len(transformed)]), axis = 0)
#     clf.fit(trainMat, trainLabels)
#     testMat = transformed[40:71]
#     testLabels = labels[40:71]
#     print 'Accuracy is', clf.score(testMat, testLabels), 'when k =', k

