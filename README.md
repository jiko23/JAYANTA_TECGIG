# JAYANTA_TECGIG
AUTOMATIC MULTI-LABEL CLASSIFICATION 

This project is based on Automatic Multi-Label Classification. In this project a possible approach to map Keyeword/entity/key_phrases to categories has been also demonstrated. There could be many possible solutions for mapping to categories might be present or can be build. As mapping manually is not possible for a large dataset thus, need to do automatically. The approach shown here is unsupervised method. Any suggestion for changes are always welcome.
The following steps to be followed below ::
(a) Environment Setup for this project:
 	i. Install Anaconda (python_version > 3)
	ii. Install Tensorflow-GPU and Keras. Take help of the below document: https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781
	iii. Install googletrans package from anaconda cloud or using pip.
	iv. Install textblob package from anaconda cloud or using pip.
	v. Install nltk package from anaconda cloud.
	vi. Install rake_nltk.
(b) Project build steps::
	i. 'data_preprocess.py' : It consists of data preprocessing steps,translation of data(text) to english language. For translating to english language two different packages has been used: (i) textblob, (ii) googletrans . To avoid putting much load on one translator try-except method has been used. It is recommended that internet connection must be very strong for large dataset translation. Key_phrase extraction has been also done here. Finally the preprocessed data has been converted to .csv file. (Kindly change the saving address as per yourself)
	ii. 'data_prepare.py' : This module consist of the keyword extraction from the key phrases. Prediction of 1st label .
	iii.  ‘Multilabel_predict.py' :  This consist of the prediction of the 2nd Label using the data from previous module.

(c) Running project Steps: Run scripts in the below order: 1. data_preprocess.py , 2. multilabel_predict.py 
Result files are below::
updated_train_data_update1.csv --> This file contains the cleaned texts and extracted keywords.
Result.csv --> This file contains the proposed approach for automatic multi-label classification of extracted keywords with the categories provided in cat_level.csv file.
point to be noted: these results are for 3000 samples and is presented to prove that the program is running properly.

Note: Kindly note that one must have a good internet to run this project.



###################################################################################################################################################################################

DESCRIPTION REGARDING THE APPROACH::

This project is totally automated and it has been build with minimum involvement of any Natural Language Processing or NLTK libraries. This project has been developed keeping in mind about various challenges like cost effectiveness, automation with minimum human supervision, effective performance with advanced approach. This project has been developed completely using Python programming language, Tensorflow,  Keras,  Platform used is Anaconda.

This project can be divided into 3 models as follows:
1>	Data Preprocessing
2>	1st  Label/Category prediction using Unsupervised algorithm
3>	Predicting 2nd Label/Category on the basis of the 1st predicted Label/Category


Lets talk about the 1st module i.e. Data Preprocessing.  Normally people would be directly apply algorithms like TF-IDF or any NLTK/NLP based packages or libraries to extract keywords from a given sentence or description ,but, in this project the method followed is a bit different.
Method:
Read each row from the provided dataset. While reading the dataset it was kept in mind that a dataset can consist of multiline data within a cell or block. So, this module will ready each row of the dataset, divide the cells having multiline data and the join it with the same sequence. Hence user will get the whole data rather than any missed out data. Then, with these data a dataframe has been created.
While reading the data from dataset user has been given flexible choice of how many rows the user want to create the dataframe.
Next is translation of data from the dataframe. This is the most heaviest method in this module, as no paid service has been used to translate the data. This method has been incorporated so that user could achieve better result with no expenditure. This method needs a very good internet connectivity. In this method rather than using only one translator, two different translators has been used to minimize the load on each translators. Two translator packages are Textblob and Google translator. First preference has been given to TextBlob and if it generates error then automatically Google translator will catch the data and translate it. Target language for translation is English.  To reduce further load translation has been done in a batch size of 500 data at a time. After translating the data has been inserted into the original dataframe by replacing old data.  Rather than extracting keywords directly, Key phrases from the dataframe column(Title,Description) has been done using Rake package from Rake_NLTK.
A clean_text()  function is present to clean unnecessary space, backslash, tabs and then it will convert the text into lowercase.
Finally a function named remove_stopwords() is present to remove all english stopwords.
As this method is time consuming so in the end the cleaned dataframe with extracted key phrases will be saved in a location in the form of a .csv file.


Now lets talk about 2nd module i.e. 1st  Label/Category prediction using Unsupervised algorithm. 
In this module the recently created dataframe from module 1 will be read as well as the category dataset. While in the 1st module key phrases were extracted , now in this module keywords will be extracted. This method will result in deletion of unnecessary phrases or unweighted keywords.
So in this module only keywords that belongs to Noun or Verb POC tags has been accepted but user has been given flexibility to accept any other POC tags too. User just has to mention the tags in the list named accept_tags . This tags has been detected using Texblob one can use any other NLTK library to identify POC tags.
Keywords matching those tags were stored in a dictionary. This dictionary is a global dictionary. These words has been provided with unique values in the dictionary.
With the dictionary further word sequences were created i.e. each keyword has been represented with a unique integer. This method has been done manually. One can also use any sequence creating packages.
Next sentence padding has been done to match the length of all the sequences. Zeros were added in the end to padd. User has been given flexibility to add padding in the beginning also, just put ‘Front’  as value for padding area in the function padding() Finally these padded sequence has been stored into an array.
Steps 2,3,4 are the steps that a tokenizer does just the difference is that in this module it has been done manually without binary values.

Previously, K-Mean Clustering was used to predict the first label. But this time Fuzzy C-Mean Clustering has been used. Lets take a look into Fuzzy C-Mean Clustering algorithm first. Number of clusters were equal to the number of unique labels/categories given in cat_levels.csv which were stored in a dictionary named labels_dict with unique values to each label.

Fuzzy C-Mean Clustering : a. Create a matrix of random weights ‘Wij’, where  0> Wij <1. The sum of a set of weights must be equal to zero(0). So, Wij is the weight with which data Xij belongs to a cluster.  
Let ‘n’  number of data, ‘C’  number of clusters, ‘d’  dimention of the data
So, lowe limit j = 1, uper limit k, Summation(Wij) = 1
b. Each cluster Cj will hold non zero weights. Now , calculate centroid of each cluster using fuzzy pseudo-partition.
c. Calculate the fuzzy pseudo partition i.e. Wij
d. Repeat steps b and c if centroids doesnot change.

Reason for choosing Fuzzy C-Mean clustering : As in our label overlaping is present so Fuzzy Clustering is prefered. If  well seperated clustering were present then K-Mean clustering could be used.
With the C-Mean clustering in this module the 1st label has been predicted i.e. Label_1.
Weights has been assigned randomly using the formula: [a,b,c,…,n], s = sum(a+b+c+..+n)
[a/s , b/s , c/s , .., n /s], sum(a/s + b/s + c/s + …+ n/s) = 1.
Note: Fuzzy C-Mean Clustering has been programmed step by step with mathematical calculations rather than using inbuilt library.
Finally a new column named Label_1 was included in the existing dataframe and was passed to the new module.


Now lets talk about our 3rd module i.e. Predicting 2nd Label/Category on the basis of the 1st predicted Label/Category .
We now have the first label, now lets find the second label.
Note: User has been given the freedom to choose any number of clusters. Just need to change the probability selection process while predicting in this module.
In this module the second label/category has been predicted using LSTM network. 
In this module the target values were the categories from the 2nd module. Activation function used in the final layer of the LSTM network is Softmax. This network has been build using Keras Functional APIs i.e. building each layers as per our choice and style.
Here, ‘keyword_set’ has been taken as feature and Label_1 as target. User has been given the flexibility to predict the 1st label from this module. Just mention the test feature and tokenize the test feature as per the training feature and predict using the test dataset. Also during training mention the validation split.
Now from this layer we will get the 2nd label. A new column named Label_2 has been added to the existing dataframe  and passed to the next module as input data.


Finally we will create a new dataframe with two columns named ‘id’ and ‘category tree’ and place the  id of each feature data and labels as per the guideline to the category_tree column.

SUGGESTIONS ARE ALWAYZ WELLCOME...........................................

