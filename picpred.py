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
from sklearn.metrics import roc_auc_score
#Keras
import tensorflow as tf
from tensorflow.keras import regularizers,optimizers
import tensorflow.keras as keras
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Reshape, MaxPooling1D, dot, Masking
from tensorflow.keras.layers import Activation, RepeatVector, Permute, multiply, Lambda, GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate, add, Conv1D, BatchNormalization, Flatten, Subtract
from tensorflow.keras.backend import epsilon, clip, get_value, set_value, transpose, variable, square
from tensorflow.keras.losses import mean_absolute_error


#visualization
from tensorflow.keras.callbacks import TensorBoard
#Custom
from lr_finder import LRFinder
from scipy import stats
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Neural Network for predicting binding of peptides to MHCII variants.''')

parser.add_argument('train_df', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to df.')
parser.add_argument('train_aa_enc', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with amino acid encodings')
parser.add_argument('test_df', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to df.')
parser.add_argument('test_aa_enc', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with amino acid encodings')
# parser.add_argument('params_file', nargs=1, type= str,
#                   default=sys.stdin, help = 'Path to file with net parameters')
parser.add_argument('out_dir', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to output directory. Include /in end')

#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

#FUNCTIONS
def read_net_params(params_file):
    '''Read and return net parameters
    '''
    net_params = {} #Save information for net

    with open(params_file) as file:
        for line in file:
            line = line.rstrip() #Remove newlines
            line = line.split("=") #Split on "="

            net_params[line[0]] = line[1]


    return net_params

def pad(ar, x, y):
    '''Pads a 2D array to len x
    '''
    shape = ar.shape

    if max(shape) < x:
        empty = np.zeros((x,y))
        empty[0:len(ar)]=ar
        ar = empty

    return ar


def convert_to_pic50(y):
    '''Convert log50k vals to pic50'''
    y = np.power(50000, (1-y))
    #Convert to pic50 = -log10(ic50)
    y = -np.log10(y)
    return y

#MAIN
args = parser.parse_args()
train_df = pd.read_csv(args.train_df[0])
train_aa_enc = np.load(args.train_aa_enc[0], allow_pickle = True)
test_df = pd.read_csv(args.test_df[0])
test_aa_enc = np.load(args.test_aa_enc[0], allow_pickle = True)
#params_file = args.params_file[0]
out_dir = args.out_dir[0]

#Assign data and labels
#Get converted ic50 values
#bins = np.array([0.1, 0.2, 0.3, 0.426, 0.5, 0.6, 0.7, 0.8, 0.9])
bins = np.array([-5.47712125, -4.77712125, -4.07712125, -3.37712125, -2.69897,
       -1.97712125, -1.27712125, -0.57712125,  0.12287875,  0.82287875])
bins = np.expand_dims(bins, axis=0) #required for later multiplication

#Get values
y_train = np.asarray(train_df['log50k'])
y_train = convert_to_pic50(y_train)
y_test = np.asarray(test_df['log50k'])
y_test = convert_to_pic50(y_test)
#It is probably easier to convert to pic50 - bigger differences btw binding/non

#Get y_test in binary
#When classifying the peptides into binders and non-binders
#for calculation of the AUC values for instance, a threshold of 500 nM is used.
#This means that peptides with log50k transformed binding
#affinity values greater than 0.426 are classified as binders.

true_binary = []
for i in range(len(y_test)):
    if y_test[i]<=-np.log10(500):
        true_binary.append(1)
    else:
        true_binary.append(0)

#Bin data
#y_train = np.digitize(y_train, bins)
#y_test = np.digitize(y_test, bins)
#Onehot encode
#y_train = np.eye(bins.size+1)[y_train]
#y_test = np.eye(bins.size+1)[y_test]

#onehot encode aa_enc
#train
X1_train =[]
for i in range(len(train_aa_enc)):
    X1_train.append(np.eye(20)[train_aa_enc[i]])
max_length = 37 #Goes from 9-37
for i in range(len(X1_train)):
    X1_train[i] = pad(X1_train[i], max_length, 20)
X1_train = np.array(X1_train)
#test
X1_test =[]
for i in range(len(test_aa_enc)):
    X1_test.append(np.eye(20)[test_aa_enc[i]])
max_length = 37 #Goes from 9-37
for i in range(len(X1_test)):
    X1_test[i] = pad(X1_test[i], max_length, 20)
X1_test = np.array(X1_test)


# #Get allele encodings
X2_train = np.asarray(train_df['allele_enc'])
X2_test = np.asarray(test_df['allele_enc'])

#onehot encode X2 train and test
X2_train = np.eye(80)[X2_train]
X2_train = np.expand_dims(X2_train, axis=2)
X2_test = np.eye(80)[X2_test]
X2_test = np.expand_dims(X2_test, axis=2)
#Tensorboard for logging and visualization
log_name = str(time.time())
tensorboard = TensorBoard(log_dir=out_dir+log_name)

######MODEL######
#Parameters
#net_params = read_net_params(params_file)
input_dim1 = (37, 20) #20 AA*35 residues
input_dim2 = (80, 1) #Shape of allele encodings
num_classes = max(bins.shape) #add +1 if categorical
kernel_size =  9 #The length of the conserved part that should bind to the binding grove

#Variable params
filters =  10#int(net_params['filters']) # Dimension of the embedding vector.
batch_size = 32 #int(net_params['batch_size'])

#lr opt
find_lr = 0
#loss
#loss = 'bin_loss'#'categorical_crossentropy'
#LR schedule
step_size = 5 #should increase alot - maybe 5?
num_cycles = 3
num_epochs = step_size*2*num_cycles
num_steps = int(len(X1_train)/batch_size)
max_lr = 0.01
min_lr = max_lr/10
lr_change = (max_lr-min_lr)/step_size  #(step_size*num_steps) #How mauch to change each batch
lrate = min_lr
#MODEL
in_1 = keras.Input(shape = input_dim1)
in_2 = keras.Input(shape = input_dim2)

#Convolution on aa encoding
x = Conv1D(filters = filters, kernel_size = kernel_size, input_shape=input_dim1, padding ="valid")(in_1) #Same means the input will be zero padded, so the convolution output can be the same size as the input.
#take steps of 1 doing 9+20 convolutions using filters number of filters
x = BatchNormalization()(x) #Bacth normalize, focus on segment
x = Activation('relu')(x)

#Flatten for concatenation
flat1 = Flatten()(x)  #Flatten
flat2 = Flatten()(in_2)  #Flatten

x = concatenate([flat1, flat2])


#Dense final layer for classification
probabilities = Dense(num_classes, activation='softmax')(x)
bins_K = variable(value=bins)
#
#Multiply the probabilities with the bins --> gives larger freedom in assigning values
#I also think it is better to predict a true value to see the correlation
def multiply(x):
    return tf.matmul(x, bins_K,transpose_b=True)
pred_vals = Lambda(multiply)(probabilities)



#Custom loss
def bin_loss(y_true, y_pred):
  #Shold make this a log loss
	g_loss = (y_true-y_pred)**2 #general, compare difference
	kl_loss = keras.losses.kullback_leibler_divergence(y_true, y_pred) #better than comparing to gaussian?
	sum_kl_loss = keras.backend.sum(kl_loss, axis =0)
	sum_g_loss = keras.backend.sum(g_loss, axis =0)
	sum_g_loss = sum_g_loss*5 #This is basically a loss penalty
	loss = sum_g_loss+sum_kl_loss
	return loss


#Model: define inputs and outputs
model = Model(inputs = [in_1, in_2], outputs = pred_vals) # probabilities)#
opt = optimizers.Adam(clipnorm=1., lr = lrate) #remove clipnorm and add loss penalty - clipnorm works better
model.compile(loss=bin_loss, #categorical_crossentropy,
              optimizer=opt)
              #metrics = ['accuracy'])



if find_lr == True:
  lr_finder = LRFinder(model)

  X_train = [X1_train, X2_train]
  lr_finder.find(X_train, y_train, start_lr=0.00000001, end_lr=1, batch_size=batch_size, epochs=1)
  losses = lr_finder.losses
  lrs = lr_finder.lrs
  l_l = np.asarray([lrs, losses])
  np.savetxt(out_dir+'lrs_losses.txt', l_l)
  num_epochs = 0

#LR schedule
class LRschedule(Callback):
  '''lr scheduel according to one-cycle policy.
  '''
  def __init__(self, interval=1):
    super(Callback, self).__init__()
    self.lr_change = lr_change #How mauch to change each batch
    self.lr = min_lr
    self.interval = interval

  def on_epoch_end(self, epoch, logs={}):
    if epoch > 0 and epoch%step_size == 0:
      self.lr_change = self.lr_change*-1 #Change decrease/increase

    self.lr = self.lr + self.lr_change
    keras.backend.set_value(self.model.optimizer.lr, self.lr)

    #Evaluate AUC
    pred = model.predict([X1_test, X2_test])
    #pred = np.argmax(pred, axis = 1)
    pred_binary = []

    for i in range(len(pred)):
        if pred[i]<=-np.log10(500):
            pred_binary.append(1)
        else:
            pred_binary.append(0)

    print('\tAUC',roc_auc_score(true_binary, pred_binary))


#Lrate
lrate = LRschedule()


#Checkpoint
filepath=out_dir+"weights-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)

#Summary of model
print(model.summary())

callbacks=[lrate, tensorboard, checkpoint]

#from tensorflow.keras.models import model_from_json
#serialize model to JSON
model_json = model.to_json()
with open(out_dir+"model.json", "w") as json_file:
    json_file.write(model_json)

#Fit model
model.fit(x = [X1_train, X2_train],
            y = y_train,
            batch_size = batch_size,
            epochs=num_epochs,
            validation_data = [[X1_test, X2_test], y_test],
            shuffle=True, #Dont feed continuously
            callbacks=callbacks)


#For converting binned predictions to binary
if find_lr == False:
    pred = model.predict([X1_test, X2_test])
    true = y_test
    np.save(out_dir+'true.npy', true)
    np.save(out_dir+'pred.npy', pred)
    pdb.set_trace()
