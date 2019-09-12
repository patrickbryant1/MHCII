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
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Reshape, MaxPooling1D, dot, Masking, UpSampling1D
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
parser = argparse.ArgumentParser(description = '''An autoencoder for encoding MHCII alleles into embeddings.''')

parser.add_argument('aa_encodings', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to folder with amino acid encodings')
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


#MAIN
args = parser.parse_args()
aa_enc_files = glob.glob(args.aa_encodings[0]*)
#params_file = args.params_file[0]
out_dir = args.out_dir[0]

#Assign data and labels
alleles = []
allele_enc = []
for file in aa_enc_files:
    name = file.split('/')[-1]
    enc = np.load(name, allow_pickle = True)
    enc = np.eye(20)[enc]
    enc = pad(enc, 266, 20)
    allele_enc.append(enc)

X = np.asarray(allele_enc)
#Tensorboard for logging and visualization
log_name = str(time.time())
tensorboard = TensorBoard(log_dir=out_dir+log_name)


######MODEL######
#Parameters
#net_params = read_net_params(params_file)
input_dim = (266,20) #20 AA*25 residues

kernel_size =  20 #20 aa
stride = 1

filters =  10
batch_size = 32

#Attention size
attention_size = filters*17+42
#lr opt
find_lr = 0
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
input = keras.Input(shape = input_dim)


#Convolution on aa encoding
x = Conv1D(activation='relu', filters = filters, kernel_size = kernel_size, dilation_rate = 2, input_shape=input_dim, padding ="same")(input)
x = AveragePooling1D(data_format='channels_first')(x) #data_format='channels_first'
x = Conv1D(activation='relu', filters = filters/2, kernel_size = kernel_size, dilation_rate = 2, input_shape=input_dim, padding ="same")(x)
encoded = AveragePooling1D(data_format='channels_first')(x) #data_format='channels_first'


x = Conv1D(activation='relu', filters = filters/2, kernel_size = kernel_size, dilation_rate = 2, input_shape=input_dim, padding ="same")(encoded)
x = UpSampling1D(size=2)(x) #Repeats each temporal step size times along the time axis.
x = Conv1D(activation='relu', filters = filters, kernel_size = kernel_size, dilation_rate = 2, input_shape=input_dim, padding ="same")(in_1)
x = UpSampling1D(size=2)(x) #Repeats each temporal step size times along the time axis.
decoded = Conv1D(activation='sigmoid', filters = 20, kernel_size = kernel_size, dilation_rate = 2, input_shape=input_dim, padding ="same")(x)#20 aa

#Model in and outputs
model = Model(inputs = [input], outputs = decoded) #probabilities)#
opt = optimizers.Adam(clipnorm=1., lr = lrate) #remove clipnorm and add loss penalty - clipnorm works better
model.compile(loss=binary_crossentropy',
              optimizer=opt,
              metrics = ['accuracy'])



if find_lr == True:
  lr_finder = LRFinder(model)

  lr_finder.find(X, X, start_lr=0.00000001, end_lr=1, batch_size=batch_size, epochs=1)
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
model.fit(x = [X],
            y = X,
            batch_size = batch_size,
            epochs=num_epochs,
            shuffle=True, #Dont feed continuously
            callbacks=callbacks)


#Convert binned predictions to binary
pred = model.predict([X])
np.save(out_dir+'true.npy', X)
np.save(out_dir+'pred.npy', pred)
pdb.set_trace()
