from data_prepare import data
import numpy as np 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Activation, Dense, Dropout
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers.merge import concatenate
#from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras import optimizers as opt

############## READING DATASET CSV FILE ##########################

print(data.describe())

data = data.dropna()

df=data.sample(frac=1) ## SHUFFLING THE DATA TO AVOID A PARTICULAR DATA PATTERN LEARNING AND FOR BETTER PREDICTION

################ DIVIDING THE DATAFRAME DATA INTO TRAIN AND TEST DATA ########################################################
#train_size = int(len(df) * .80) ## DEFINING THE RATIO OF DATA FOR TRAINING i.e.80% DATA FOR TRAINING AND REMAINING 20% FOR TESTING

train1 = df['keyword_set'] ## TRAIN FEATURE 1

train_label = df['Label_1']

#test1 = df['keyword_set'][train_size : ] ## TEST FEATURE 1

#test_label = df['Labels'][train_size : ]


################ TOKENIZING THE TEXT DATA AND CONVERTING INTO MATRIX TO PASS INTO KERAS MODEL #####################################

vocab1 = 100 ## VOCABULARY SIZE FOR FEATURE 1
tokenizer = Tokenizer(num_words=vocab1)
tokenizer.fit_on_texts(train1)
x_train1 = tokenizer.texts_to_matrix(train1, mode='tfidf')
x_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1], 1)

#x_test1 = tokenizer.texts_to_matrix(test1, mode='tfidf')
#x_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1], 1)



################# ENCODING THE LABELS FOR BOTH TRAINING AND TESTING DATA ###########################
encoder = LabelBinarizer()

encoder.fit(train_label)

y_train = encoder.transform(train_label)

#y_test = encoder.transform(test_label)

############################ CREATING THE KERAS MODEL USING KERAS FUNCTIONAL API ##########################################################
x1_in = Input(shape=(vocab1,1),name = 'x_train1') ## DEFINING THE SHAPE FOR TRAIN FEATURE 1

x1 =LSTM(120)(x1_in)
x1 = Dense(120,activation = 'relu')(x1) ## PASSING THE TRAIN FEATURE 1 TO DENSE LAYERS FOR TRAINING
x1 = Dense(120,activation = 'relu')(x1)
x1 = Dropout(0.3)(x1)


output = Dense(y_train.shape[1],activation = 'softmax',name = 'final_out')(x1) ## FINAL OUTPUT LAYER

model = Model(inputs= x1_in, outputs=output) ## DEFINING THE MODEL WITH INPUT AND OUTPUT
print(model.input_shape)
print(model.output_shape)
model.summary() ## GENERATING SUMMARY OF THE MODEL

optimizer = opt.Adam(lr=0.001) ## DEFINING THE OPTIMIZER ALONG WITH LEARNING RATE
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy']) ## COMPILING THE DEFINED MODEL FOR MINIMIZING THE LOSS(error) AND TRACK THE ACCURACY PARAMETER

es = EarlyStopping(monitor='val_acc', mode='max',patience=250) ## EARLY STOPPING FOR TAKING THE BEST MODEL TILL EPOCH WHERE LOSS IS DECREASING
callbacks_list = [es]

#checkpoint = ModelCheckpoint('table_category.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min') ## SAVING THE BEST MODEL ON DISK
#callbacks_list = [checkpoint]

model.fit(x_train1, y_train, batch_size=100, epochs=650, callbacks=callbacks_list, validation_split = 0, shuffle=True, verbose=1) ## PASSING THE TRAINING FEATURES AND LABELS TO THE MODEL

train_score = model.evaluate(x_train1, y_train, batch_size=100, verbose=1) ## TRAINING EVALUATION ALONG WITH LOSS AND ACCURACY SCORE
print('Train[Categorical_LOSS,ACCURACY]:', train_score) ##### train set loss and accuracy



#model.save('table_category.h5')
########################## EVALUATING THE DEFINED MODEL. PREDICTING WITH THE TRAINED MODEL ON TEST DATA. ########################################################################
text_labels = encoder.classes_ ## ORIGINAL LABELS

############### PERFORMING PREDICTION ON TEST DATA USING THE TRAINED MODEL ###########################################

df['Label_2'] = ""

final_df = pd.DataFrame(columns=['id','category_tree'])

for i in range(len(x_train1)) :

	prediction = model.predict([np.array([x_train1[i]])])

	predicted_label= text_labels[np.argmax(prediction[0])]

	df.at[i, 'Label_2'] = predicted_label

	tree_ = "{}^{}-->{}".format(df['Label_1'].loc[i],df['Label_2'].loc[i],df['keyword_set'].iloc[i])

	final_df.at[i,'id'] =  df['id'].loc[i]

	final_df.at[i, 'category_tree'] = tree_


final_df.to_csv(r'E:\prog\analytic_vidya\segmentation\Result.csv') ## CONVERTING THE DATAFRAME INTO CSV FILE


######################################################################################################################## END ##################################################################################################################################	
