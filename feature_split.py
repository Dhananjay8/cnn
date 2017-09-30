# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[2]:

#Step 1: Preprocessing:Encode labels with value between 0 and n_classes-1
def encode(train, test):	
    label_encoder = LabelEncoder().fit(train.species)
	#Fit label encoder	self : returns an instance of self.
    labels = label_encoder.transform(train.species)
	#Transform labels to normalized encoding. Returns:	y : array-like of shape [n_samples]
    classes = list(label_encoder.classes_)
	
	#input dataset to pandas.drop method
    train = train.drop(['species', 'id'], axis=1)
    test = test.drop('id', axis=1)

    return train, labels, test, classes

train, labels, test, classes = encode(train, test)
#calling encode function


# In[3]:

#Step 2: standardize train features

scaler = StandardScaler().fit(train.values)
#fit(X[, y]) 	Compute the mean and std to be used for later scaling.
#fit_transform(X_train) Compute mean, std and transform training data
scaled_train = scaler.transform(train.values)
#transform(X[, y, copy]) 	Perform standardization by centering and scaling


#Step 3: split train data into train and validation

sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
#Stratified ShuffleSplit cross validation iterator
#Provides train/test indices to split data in train test sets.
#This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.
#Note: like the ShuffleSplit strategy, stratified random splits do not guarantee that all folds will be different, although this is still very likely for sizeable datasets.
for train_index, valid_index in sss.split(scaled_train, labels):
	#split(X, y[, groups]) 	Generate indices to split data into training and test set.
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]
	#split method returns train : ndarray-The training set indices for that split. & test : ndarray-The testing set indices for that split.

nb_features = 64 # number of features per features type (shape, texture, margin)   
nb_class = len(classes)

# reshape train data
X_train_r = np.zeros((len(X_train), nb_features, 3))
#it returns a new array of given shape and type, filled with zeros.
X_train_r[:, :, 0] = X_train[:, :nb_features]
X_train_r[:, :, 1] = X_train[:, nb_features:128]
X_train_r[:, :, 2] = X_train[:, 128:]

# reshape validation data
X_valid_r = np.zeros((len(X_valid), nb_features, 3))
#it returns a new array of given shape and type, filled with zeros.
X_valid_r[:, :, 0] = X_valid[:, :nb_features]
X_valid_r[:, :, 1] = X_valid[:, nb_features:128]
X_valid_r[:, :, 2] = X_valid[:, 128:]


# In[ ]:

# Keras model with one Convolution1D layer
model = Sequential()
#The Sequential model is a linear stack of layers.
# Sequential model can be done by passing a list of layer instances to the constructor or by add() method
model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(nb_features, 3)))
#Pass an input_shape argument to the first layer. This is a shape tuple (a tuple of integers or None entries, where None indicates that any positive integer may be expected). In input_shape, the batch dimension is not included.
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_class))
model.add(Activation('softmax'))

#https://keras.io/getting-started/sequential-model-guide/
#https://keras.io/models/about-keras-models/

y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)

sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

nb_epoch = 15
model.fit(X_train_r, y_train, nb_epoch=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=16)


# In[ ]:

