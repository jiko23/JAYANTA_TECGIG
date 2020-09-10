import re
#from data_preprocess import df1
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance
from textblob import TextBlob


word_dict = {}	## Global dictionary to store word tokenizer.

labels_dict = {}

v = 1	## Variable will be used to give a unique value to each word while tokenizing.



"""
	Funtion to perform tokenizing words from sentences.
"""
def check_dict(word_set):

	global word_dict

	global v

	for i in word_set:

		if i in [i for i in word_dict.keys()]:

			pass

		else:

			word_dict[i] = v

			v = v+1

	return word_dict



"""
	Data formation for ML model, using padding either in begging or end f sentence to match all the sentance length.
"""
def padding(tokenized_list, padding_area):

	max_len = []

	for i in range(len(tokenized_list)):

		max_len.append(len(tokenized_list[i]))

	
	for i in range(len(tokenized_list)):

		if padding_area == 'Front' :

			len_ = max(max_len) - len(tokenized_list[i])

			for j in range(len_):

				tokenized_list[i].insert(j, 0)

		elif padding_area == 'End' :

			len_ = max(max_len) - len(tokenized_list[i])

			for j in range(len_):

				tokenized_list[i].insert(len(tokenized_list[i]) + j, 0)
				
	return tokenized_list


	
########################################################################################################################################################################################		

data = pd.read_csv(r'E:\prog\Train_data\updated_train_data_update1.csv') ## Feature Dataset
#data = df1[1:100]
categories_ = pd.read_csv(r'E:\prog\analytic_vidya\segmentation\cat_levels.csv') ##Label Dataset

catgory_ = [i for i in categories_.level1_categories.unique() if str(i) != 'nan']


########################################################################################################################################################################################
'''
	Creating dictionary for each label/label_set.
'''
for i in range(len(catgory_)) :

	labels_dict[i] = catgory_[i]

########################################################################################################################################################################################

'''
	Making dictionary with the words in column 'Keyword_set'. Then performing tokenization, sentence sequence formation
	and sentence padding to match length of all sentences to be equal.
'''
	
sentences = []


for j in range(len(data)):


	a = [data['keyword_set'].iloc[j]]

	sentences.append(a)



accept_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' ,'CD']

sentences_ = []

for i in range(len(sentences)):

	if len(sentences[i]) < 1:

		continue

	else:

		temp_ = []

		for j in sentences[i]:

			blob_ = TextBlob(str(j))

			for i in blob_.tags :

				if i[1] in accept_tags:

					temp_.append(i[0])
				else:
					continue

			temp_ = list(dict.fromkeys(temp_))

			sentences_.append([' '.join([k for k in temp_])])


del temp_


sentences = [i for i in sentences_ if i != [''] or i != []]


########################################################################################################################################################################################

seperated_words = []	## List will be used to store sentence split. And further it will be used to store the sentence sequence.

words_ = []	## List will be used to store special character striped from sentence words.

l = 0	## For incrementing the while loop


"""
	This loop will perform spliting of sentences from list "sentences" and store the splited words from each sentence. Finally striped words will
	further be stored into list "words_" which will be used for tokenizing words.
"""
while l<1:

	for i in sentences:

		a = str(i[0])

		res = a.split()

		seperated_words.append(res)

		for j in res:

			for k in str(j):

				if re.match("^[@_.!#$%^&*()<>?/\|}{~: 	]*$", k) == None:

					continue

				else:

					j = j.replace(k, "")

			
			words_.append(j)

	l = l+1		



b = check_dict(set(words_))	## Sending the list "words_" for tokenization.

##########################################################################################################################################################################################

"""
	This loop is responsible for generating tokenized word sentence sequence. Sequences will be stored in list "seperated_words" .
"""
for i in range(len(seperated_words)):

	for j in range(len(seperated_words[i])):

		for k in seperated_words[i][j]:

			if re.match("^[@_.!#$%^&*()<>?/\|}{~: 	]*$", k) == None:

				continue

			else:
				seperated_words[i][j] = seperated_words[i][j].replace(k, "")

		if seperated_words[i][j] in [key for key in word_dict.keys()]:

			seperated_words[i][j] = word_dict[seperated_words[i][j]]

		else:
			v = v+1

			word_dict[seperated_words[i][j]] = v

			seperated_words[i][j] = word_dict[seperated_words[i][j]]
			


seperated_words = [i for i in seperated_words if i != []]	## removing empty cells


a = np.array(padding(seperated_words, 'End'))	## Performing padding to match the length of the longet sentence in dataset

#############################################################################################################################################################################################

'''
	Generating random weights for Fuzzy C-Mean algorithm.
'''
np.random.seed(1)

weights_ = [] ##weights for the algorithm.

no_clusters = len(catgory_)


for i in range(0,a.shape[0]):

	temp_ = []

	for j in range(0, no_clusters) :

		w_ = random.randint(0, 20)

		temp_.append(w_)

	weights_.append(temp_)


del temp_


for i in range(len(weights_)) :

	s = sum(weights_[i])

	weights_[i] = [i/s for i in weights_[i]]


weights_ = np.array(weights_)

#########################################################################################################################################################################################

'''
	Predicting the cluster using Fuzzy C-Mean clustering algorithm.
'''

number_iterations = 12

itr_ = 0


while itr_ < number_iterations:

	##print("Updated weights:", '\n', weights_) check if wights are different or not after each iteration
 
	clusters_centroids = []

	weights_i_j = []

	temp_ = []

	y = 0


	for i in range(0,weights_.shape[1]):

		for j in range(0, weights_.shape[0]):

			y = y +  (weights_[j][i]**2) ## p = 2

		weights_i_j.append(y)

		y = 0


	for i in range(0,weights_.shape[1]):

	
		for j in range(0, a.shape[1]):

			x = 0

			for k in range(0, a.shape[0]):

				#print(a[k][j])

				#print(weights_[k][i])

				x = x + (weights_[k][i]**2 * a[k][j])

			y = x / weights_i_j[i]

			temp_.append(y)


	i = 0

	while i < len(temp_) :

		clusters_centroids.append(temp_[i : i + a.shape[1]])

		i = i + a.shape[1]
	
	#print("Cluster Centroids:", '\n', clusters_centroids)


	del temp_ #deleting the temporary list from memory



	'''
		Calculating distance of data with each clusters.
	'''
	euclidean_distance_ = []

	temp_ = []

	for i in range(0, a.shape[0]):

		#temp_.clear()

		for j in range(0, len(clusters_centroids)):

			euclidean_distance = distance.euclidean(a[i], clusters_centroids[j])

			temp_.append(euclidean_distance)

	#print(temp_)

	i = 0

	while i < len(temp_):

		euclidean_distance_.append(temp_[i : i + weights_.shape[1]])

		i = i + weights_.shape[1]


	#print("Distance from cluster are:", '\n', euclidean_distance_)

	del temp_

	updated_weights_ = []

	denominator_ = 0

	temp_ = []


	for i in range(len(euclidean_distance_ )):

		for j in range(len(euclidean_distance_ [i])):

			## p = 2 so 1/p-1 = 1/2-1
			denominator_ = denominator_ + ((1 / euclidean_distance_ [i][j])**(1 / (2-1))) #(1/euclidean_distance_[i][j])

		temp_.append(denominator_)


	updated_weight = []



	for i in range(len(euclidean_distance_ )):


		for j in range(len(euclidean_distance_[i])):

			update_ = ((1 / euclidean_distance_ [i][j])**(1 / (2-1)))  / temp_[i]

			updated_weight.append(update_)




	i = 0

	while i < len(updated_weight) :

		updated_weights_.append(updated_weight[i : i + weights_.shape[1]])

		i = i + weights_.shape[1]



	del temp_
	del updated_weight


	'''
		Checking the sum of every row. If each and all sum equals to 1, then again iterate untill the centroids changes 
		and results in sum not equal to 1. weights_ = updated_weight.
	''' 
	Sum_check = []

	for i in updated_weights_ :

		Sum = sum(i)

		Sum_check.append(Sum)


	if len(set(Sum_check)) == 1 and all(elem == 1 for elem in Sum_check) == True :

		weights_ = updated_weights_

		itr_ = itr_ + 1

		continue

	else:

		data["Label_1"] = ""

		for w_ in range(len(updated_weights_)):

			#print('keyword_set: {}	Label: {}'.format(data['keyword_set'].iloc[w_], labels_dict[np.argmax(updated_weights_[w_], axis=0)]))

			data.at[w_, 'Label_1'] = labels_dict[np.argmax(updated_weights_[w_], axis=0)]
		break #exit()	

####################################################################### END ###################################################################################################################
