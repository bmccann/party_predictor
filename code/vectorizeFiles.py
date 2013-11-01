from sklearn.feature_extraction.text import CountVectorizer
import os
os.chdir('../data/republican/')
corpus=[]
for fileName in os.listdir('.'):
	if not fileName == 'README.md':
		print(fileName) 
		corpus.append(open(fileName).read())
vectorizer=CountVectorizer(charset_error='ignore');
X=vectorizer.fit_transform(corpus);
Y=X.toarray();
importantIndex=vectorizer.vocabulary_.get('the');
print(Y.sum(axis=0).sum())
