#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
from ast import literal_eval
import pandas as pd
import glob

#Preprocessing
import math
import time
from ast import literal_eval
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

#Keras
import tensorflow as tf
from tensorflow.keras import regularizers,optimizers
from tensorflow.keras.models import model_from_json
import tensorflow.keras as keras
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Reshape, MaxPooling1D, dot, Masking
from tensorflow.keras.layers import Activation, RepeatVector, Permute, multiply, Lambda, GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate, add, Conv1D, BatchNormalization, Flatten, Subtract
from tensorflow.keras.backend import epsilon, clip, get_value, set_value, transpose, variable, square

import pdb
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

parser.add_argument('outdir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

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
outdir = args.outdir[0]

#Assign data and labels
#Read df
df = pd.read_csv(dataframe)

#Assign data and labels
#Read data
aa_enc = np.load(encodings, allow_pickle=True)

#Get pic50 values
pic50 = -np.log10(df['Measurement value'])
bins = np.array([-5.47712125, -4.77712125, -4.07712125, -3.37712125, -2.69897,
       -1.97712125, -1.27712125, -0.57712125,  0.12287875,  0.82287875])
y = np.digitize(pic50,bins)
np.save(outdir+'true.npy', pic50)
np.save(outdir+'true_binned.npy', y)
#onehot y = np.eye(42)[y]


X1 =[]
max_length = 25 #Goes from 15-25
for enc in aa_enc:
    X1.append(pad(enc, max_length))
X1 = np.asarray(X1, dtype = int)
#onehot encode X1
X1 = np.eye(20)[X1]

X2 = np.asarray(df['allele_enc'])
X2 = np.expand_dims(X2, axis=1)
#onehot encode X2
X2 = np.eye(42)[X2]

#Load and run model
bins = np.expand_dims(bins, axis=0)
bins_K = variable(value=bins)
model = load_model(json_file, weights)
pred = model.predict([X1, X2])
np.save(outdir+'pred.npy', pred[:,0])
#argmax_pred = tf.argmax(pred, 1)

#sess = tf.Session()
#called = sess.run(argmax_pred)
#np.save(outdir+'pred.npy', called)
pdb.set_trace()
