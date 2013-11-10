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
			if not (fileName == 'README.md' or fileName=='cuomo_dnc_keynote.txt'):
				##print(fileName) 
				corpus.append(open(fileName).read())
		os.chdir('../../code')
	
	if includeIndependent:
		os.chdir('../data/independent/')
		for fileName in os.listdir('.'):
			if not fileName == 'README.md':
				##print(fileName) 
				corpus.append(open(fileName).read())	
		
	vectorizer=CountVectorizer(stop_words='english',charset_error='ignore');
	print vectorizer
	X=vectorizer.fit_transform(corpus);
	Y=X.toarray();
	print len(Y)
	return [Y,vectorizer];
