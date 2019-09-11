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
import tensorflow.keras as keras
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Reshape, MaxPooling1D, dot, Masking
from tensorflow.keras.layers import Activation, RepeatVector, Permute, multiply, Lambda, GlobalAveragePooling1D
from tensorflow.keras.layers import concatenate, add, Conv1D, BatchNormalization, Flatten, Subtract
from tensorflow.keras.backend import epsilon, clip, get_value, set_value, transpose, variable, square



#visualization
from tensorflow.keras.callbacks import TensorBoard
#Custom
from lr_finder import LRFinder
from scipy import stats
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Neural Network for predicting binding of peptides to MHCII variants.''')

parser.add_argument('dataframe', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to df.')
parser.add_argument('aa_encodings', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with amino acid encodings')
#parser.add_argument('params_file', nargs=1, type= str,
#                  default=sys.stdin, help = 'Path to file with net parameters')
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

def pad(ar, x):
    '''Pads a 1D array to len x
    '''
    shape = ar.shape

    if max(shape) < x:
        empty = np.zeros((x))
        empty[0:len(ar)]=ar
        ar = empty

    return ar


#MAIN
args = parser.parse_args()
df_path = args.dataframe[0]
aa_enc = np.load(args.aa_encodings[0], allow_pickle = True)
#params_file = args.params_file[0]
out_dir = args.out_dir[0]

#Assign data and labels
#Read df
df = pd.read_csv(df_path)
#Get pic50 values
pic50 = -np.log10(df['measurement_value'])
bins = np.array([-5.47712125, -4.77712125, -4.07712125, -3.37712125, -2.7,
       -1.97712125, -1.27712125, -0.57712125,  0.12287875,  0.82287875])

#Bin the pic50 values
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
#different splits
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(X2, y):
    X1_train, X1_test = X1[train_index], X1[test_index]
    X2_train, X2_test = X2[train_index], X2[test_index]
    y_train, y_test = y[train_index], y[test_index]
bins = np.expand_dims(bins, axis=0)

#onehot encode X2 train and test
X2_train = np.eye(42)[X2_train]
X2_test = np.eye(42)[X2_test]

#onehot encode y train and test
y_train = np.eye(bins.size+1)[y_train]
y_test = np.eye(bins.size+1)[y_test]
#Tensorboard for logging and visualization
log_name = str(time.time())
tensorboard = TensorBoard(log_dir=out_dir+log_name)


######MODEL######
#Parameters
#net_params = read_net_params(params_file)
input_dim1 = (25,20) #20 AA*25 residues
input_dim2 = (1, 42) #42 types of alleles
num_classes = max(bins.shape)+1
kernel_size =  9 #The length of the conserved part that should bind to the binding grove

#Variable params
filters =  10#int(net_params['filters']) # Dimension of the embedding vector.

batch_size = 32 #int(net_params['batch_size'])
#lr opt
find_lr = 0
#LR schedule
step_size = 10 #should increase alot - maybe 5?
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


#Attention layer
#Attention layer - information will be redistributed in the backwards pass
attention = Dense(1, activation='tanh')(x) #Normalize and extract info with tanh activated weight matrix (hidden attention weights)
attention = Flatten()(attention) #Make 1D
attention = Activation('softmax')(attention) #Softmax on all activations (normalize activations)
attention = RepeatVector(212)(attention) #Repeats the input "num_nodes" times.
attention = Permute([2, 1])(attention) #Permutes the dimensions of the input according to a given pattern. (permutes pos 2 and 1 of attention)

sent_representation = multiply([x, attention]) #Multiply input to attention with normalized activations
sent_representation = Lambda(lambda xin: keras.backend.sum(xin, axis=-2), output_shape=(212,))(sent_representation) #Sum all attentions

#Dense final layer for classification
probabilities = Dense(num_classes, activation='softmax')(sent_representation)
#Dense final layer for classification
#probabilities = Dense(num_classes, activation='softmax')(merge)
bins_K = variable(value=bins)

#Multiply the probabilities with the bins --> gives larger freedom in assigning values
def multiply(x):
  return tf.matmul(x, bins_K,transpose_b=True)

#pred_vals = Lambda(multiply)(probabilities)
#The length of the validation data must be a multiple of batch size!
#Other wise you will have shape mismatches

#Custom loss
def bin_loss(y_true, y_pred):
  #Shold make this a log loss
	g_loss = mean_absolute_error(y_true, y_pred) #general, compare difference
	kl_loss = keras.losses.kullback_leibler_divergence(y_true, y_pred) #better than comparing to gaussian?
	sum_kl_loss = keras.backend.sum(kl_loss, axis =0)
	sum_g_loss = keras.backend.sum(g_loss, axis =0)
	sum_g_loss = sum_g_loss*alpha #This is basically a loss penalty

	loss = sum_g_loss+sum_kl_loss
	return loss


#Model: define inputs and outputs
model = Model(inputs = [in_1, in_2], outputs = probabilities)#pred_vals)
opt = optimizers.Adam(clipnorm=1., lr = lrate) #remove clipnorm and add loss penalty - clipnorm works better
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics = ['accuracy'])



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


#Lrate
lrate = LRschedule()


#Checkpoint
filepath=out_dir+"weights-{epoch:02d}-.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)

#Summary of model
print(model.summary())

callbacks=[lrate, tensorboard, checkpoint]

#Fit model
model.fit(x = [X1_train, X2_train],
            y = y_train,
            batch_size = batch_size,
            epochs=num_epochs,
            validation_data = [[X1_test, X2_test], y_test],
            shuffle=True, #Dont feed continuously
            callbacks=callbacks)


#Convert binned predictions to binary
pred = model.predict([X1_test, X2_test])
pred = np.argmax(pred, axis = 1)
true = np.argmax(y_test, axis = 1)
np.save(out_dir+'true.npy', true)
np.save(out_dir+'pred.npy', pred)
