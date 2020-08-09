import re
import csv
import numpy as np
import pandas as pd
from googletrans import Translator
from textblob import TextBlob
from nltk.corpus import stopwords
from rake_nltk import Metric, Rake

################################################################################################################################################
stop_words = set(stopwords.words('english')) ## ENGLISH STOP-WORD SET
regex = re.compile(r'[\n\r\t]') ## TO REMOVE UNNECESSARY CHARACTERS LIKE SPACE,TABS,etc.


"""
	READING THE 'updated_train_data.csv' FILE AND TAKING THE DATA AND CREATING DATAFRAME
"""

#dict = {'title':[],'link':[],'description':[],'long_description':[],'id':[]}

df1 = pd.DataFrame(columns =['title','link','description','long_description', 'id'])

count = 0



with open(r'E:\prog\analytic_vidya\segmentation\Train_data.csv\Train_data.csv', newline='', encoding='utf-8') as f:
	reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

	for row in reader:
		if count > 2999:
			#print(df1.describe())
			break

		row = ' '.join(row)

		b = [i for i in range(len(row)) if row[i] == '']

		try:

			if len(b) == 0:
				fields = row.split('|')
				df1.at[count,'title'] = fields[0]
				df1.at[count, 'link'] = fields[1]
				df1.at[count, 'description'] = fields[2]
				df1.at[count, 'long_description'] = fields[3]
				df1.at[count, 'id'] = fields[4] #(regex.sub(" ", fields[4]))
				#count = count + 1
		except:

			if len(b) > 0:
				for j in b:
					row[j] = 'NOPE'

				fields = row.split('|')
				df1.at[count, 'title'] = fields[0]
				df1.at[count, 'link'] = fields[1]
				df1.at[count, 'description'] = fields[2]
				df1.at[count,'long_description'] = fields[3]
				df1.at[count, 'id'] = fields[4] #(regex.sub(" ", fields[4]))
				#count = count + 1

		count = count + 1

#df1 = pd.DataFrame.from_dict(dict)
print(df1.tail(10))
print(len(df1))
###############################################################################################################################################

"""
	TRANSLATING THE TEXTS INTO ENGLISH LANGUAGE. TAKING ONLY TITLE AND DESCRIPTION COLUMNS FOR
	EASYNESS. FOR TRANSLATION USING 'GOOGLETRANS' AND 'TEXTBLOB' PACKAGES FOR BETTER PERFORMANCE. THEN EXTRACTING
	KEY PHRASES FROM THEM USING 'RAKE'(NLTK package) AND STORED IN A NEW DATAFRAME COLUMN 'keyword_set'.
"""
columns = ['title','description']
temp = []

translator = Translator()

r = Rake(stopwords=stop_words,ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO,min_length=2, max_length=4)

n = 0
t = 500
dest_t = len(df1)
print(dest_t)
while n < dest_t :

	for i in range(n,t) :
		for j in columns :
		
			print("df1[j][i] = ",df1[j][i],"i==",i)

			if (len(df1[j][i]) >= 3):
				blob = TextBlob(str(df1[j][i]))

				try: 
					language = blob.detect_language()

					if (language != 'en'):
						translated = blob.translate(to='en')
						df1[j][i] = translated

					else:
						df1[j][i] = df1[j][i]

				except:
					df1[j][i] = translator.translate(df1[j][i]).text

			else :
				df1[j][i] = translator.translate(df1[j][i]).text


			r.extract_keywords_from_text(str(df1[j][i]))
			for k in r.get_ranked_phrases_with_scores():
				if k[0] >=4:
					if k[1].find('https :// www') != -1 or k[1].find('https :// ') != -1 or k[1].find('...') != -1 :
						temp.append([i,k[1].replace('https :// ','')])
					else:
						temp.append([i,k[1]])
					print("Keyword-->",k[1])

	n = n + 500
	t = t + 500


######### STORING THE KEY PHRASES INTO DATAFRAME #########
df1["keyword_set"] = ""

for i in range(0,dest_t):
	temp1 = []
	del temp1[:]

	element_index = list(filter(lambda x: temp[x][0] == i, range(0,len(temp)))) ## will give list of matched index

	for j in element_index:
		temp1.append(temp[j][1])

	df1.at[i,'keyword_set'] = (temp1)
###########################################################################################################################################################################

"""
	THE FUNCTION WILL CLEAN BACKLASHES,NONALPHABETS,WHITESPACES AND WILL TRANSFORM INTO LOWERCASE ALPHABETS. THIS FUNCTION HAS BEEN 
	IMPLEMENTED ON 3 DATAFRAME COLUMNS i.e. 'title','description','long_description' .
"""
def clean_text(text):
    # remove backslash-apostrophe 
    text = re.sub("\'", "", text) 

    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text)

    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
 
    # remove whitespaces 
    text = ' '.join(text.split()) 

    # convert text to lowercase 
    text = text.lower() 
    
    return text


df1['title'] = df1['title'].apply(lambda x: clean_text(str(x)))
df1['description'] = df1['description'].apply(lambda x: clean_text(str(x)))
df1['long_description'] = df1['long_description'].apply(lambda x: clean_text(str(x)))
df1['keyword_set'] = df1['keyword_set'].apply(lambda x: clean_text(str(x)))

#################################################################################################################################################################

"""
	THIS FUNCTION WILL REMOVE STOPWORDS FROM 3 DATAFRAME COLUMNS i.e. 'title','description','long_description' .
"""

# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)


df1['title'] = df1['title'].apply(lambda x: remove_stopwords(x))
df1['description'] = df1['description'].apply(lambda x: remove_stopwords(x))
df1['long_description'] = df1['long_description'].apply(lambda x: remove_stopwords(x))

##################################################################################################################################################################
df1[1::].to_csv(r'E:\prog\Train_data\updated_train_data_update1.csv') ## CONVERTING THE DATAFRAME INTO CSV FILE