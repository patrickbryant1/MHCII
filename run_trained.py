#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.models import model_from_json
import math
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers, backend
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.layers import Bidirectional,CuDNNLSTM, Dropout, BatchNormalization
from tensorflow.keras.layers import Reshape, Activation, RepeatVector, Permute, multiply, Lambda
from tensorflow.keras.layers import concatenate, add, Conv1D
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import TensorBoard

import pdb

from model_inputs import split_on_h_group
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')

parser.add_argument('json_file', nargs=1, type= str,
                  default=sys.stdin, help = 'path to .json file with keras model to be opened')

parser.add_argument('weights', nargs=1, type= str,
                  default=sys.stdin, help = '''path to .h5 file containing weights for net.''')

parser.add_argument('encodings', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to np array with encoded aa sequences.')

parser.add_argument('dataframe', nargs=1, type= str,
                  default=sys.stdin, help = '''path to data to be used for prediction.''')


#FUNCTIONS
def pad(ar, x):
    '''Pads a 1D array to len x
    '''
    shape = ar.shape

    if max(shape) < x:
        empty = np.zeros((x))
        empty[0:len(ar)]=ar
        ar = empty

    return ar

def load_model(json_file, weights):

	global model

	json_file = open(json_file, 'r')
	model_json = json_file.read()
	model = model_from_json(model_json)
	model.load_weights(weights)
	model._make_predict_function()
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

#MAIN
args = parser.parse_args()
json_file = (args.json_file[0])
weights = (args.weights[0])
encodings = args.encodings[0]
dataframe = args.dataframe[0]

#Assign data and labels
#Read df
df = pd.read_csv(dataframe)

#Assign data and labels
#Read data
aa_enc = np.load(encodings, allow_pickle=True)

#Get pic50 values
pic50 = -np.log10(df['measurement_value'])
bins = np.array([-5.47712125, -4.77712125, -4.07712125, -3.37712125, -2.7,
       -1.97712125, -1.27712125, -0.57712125,  0.12287875,  0.82287875])
y = np.digitize(pic50,bins)

X1 =[]
max_length = 25 #Goes from 15-25
for enc in aa_enc:
    X1.append(pad(enc, max_length))
X1 = np.asarray(X1, dtype = int)
#onehot encode X1
X1 = np.eye(20)[X1]
X2 = np.asarray(df['allele_enc'])
X2 = np.expand_dims(X2, axis=1)
#Split so both groups are represented in train and test
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_valid = X[train_index], X[test_index]
    y_train, y_valid = y[train_index], y[test_index]

#Onehot encode labels
num_classes = max(y)+1
y_train = np.eye(max(y)+1)[y_train]
y_valid = np.eye(max(y)+1)[y_valid]
#Pad X_valid
padded_X_valid = []
for i in range(0,len(X_valid)):
    padded_X_valid.append(pad_cut(X_valid[i], 300, 21))
X_valid = np.asarray(padded_X_valid)


#Load and run model
model = load_model(json_file, weights)
pred = model.predict(X_valid)

argmax_pred = tf.argmax(pred, 1)

sess = tf.Session()
called = sess.run(argmax_pred)

argmax_valid = np.argmax(y_valid, axis = 1)
pdb.set_trace()
