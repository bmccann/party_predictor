from sklearn.feature_extraction.text import CountVectorizer
import os

def extractWordCounts(includeRepublican,includeDemocrat,includeIndependent):
	
	corpus=[];
	
	if includeRepublican:
		os.chdir('../data/republican/')
		for fileName in os.listdir('.'):
			if not fileName == 'README.md':
				##print(fileName) 
				corpus.append(open(fileName).read())
		os.chdir('../../code')
	
	if includeDemocrat:
		os.chdir('../data/democrat/')
		for fileName in os.listdir('.'):
			if not fileName == 'README.md':
				##print(fileName) 
				corpus.append(open(fileName).read())
		os.chdir('../../code')
	
	if includeIndependent:
		os.chdir('../data/independent/')
		for fileName in os.listdir('.'):
			if not fileName == 'README.md':
				##print(fileName) 
				corpus.append(open(fileName).read())	
		
	vectorizer=CountVectorizer(charset_error='ignore');
	X=vectorizer.fit_transform(corpus);
	Y=X.toarray();
	return [Y,vectorizer];
