from sklearn.feature_extraction.text import CountVectorizer
import os
import getFileNames as gf

def extractWordCounts(includeRepublican,includeDemocrat,includeIndependent):	
	corpus=[]
	labels=[]	
	if includeRepublican:
		os.chdir('../data/republican/')
		for fileName in os.listdir('.'):
			if not (fileName == 'README.md' or fileName ==
			'.DS_Store'):# or fileName == 'Ford_6_1976_social_security.txt' or fileName == 'Ford_2_3_1975_black_history.txt'):
				##if fileName=='reagan_sou_1982.txt':
					##print(fileName) 
				corpus.append(open(fileName).read())
				labels.append(1)
		os.chdir('../../code')
	if includeDemocrat:
		os.chdir('../data/democrat/')
		for fileName in os.listdir('.'):
			if not (fileName == 'README.md' or fileName ==
					'.DS_Store' or fileName=='carter_sou_1981.txt' or fileName=='cuomo_dnc_keynote.txt'):# or fileName == 'truman_11_3_1948_victory.txt'):
				##if fileName=='wilson_war_message.txt':
				##print(fileName) print 
				corpus.append(open(fileName).read())
				labels.append(0)
		os.chdir('../../code')
	if includeIndependent:
		os.chdir('../data/independent/')
		for fileName in os.listdir('.'):
			if not fileName == 'README.md':
				##print(fileName) 
				corpus.append(open(fileName).read())	
		
	vectorizer=CountVectorizer(strip_accents='ascii',stop_words='english',charset_error='replace');
	print vectorizer
	X=vectorizer.fit_transform(corpus);	
	Y=X.toarray();
	print len(Y)
	return [Y,vectorizer,labels];
