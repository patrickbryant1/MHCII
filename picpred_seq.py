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
parser.add_argument('train_C1', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with allele encodings for chain 1.')
parser.add_argument('train_C2', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with allele encodings for chain 2.')
parser.add_argument('test_df', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to df.')
parser.add_argument('test_aa_enc', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with amino acid encodings')
parser.add_argument('test_C1', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with allele encodings for chain 1.')
parser.add_argument('test_C2', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with allele encodings for chain 2.')
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
train_df = pd.read_csv(args.train_df[0])
train_aa_enc = np.load(args.train_aa_enc[0], allow_pickle = True)
train_c1 = np.load(args.train_C1[0], allow_pickle = True)
train_c2 = np.load(args.train_C2[0], allow_pickle = True)
test_df = pd.read_csv(args.test_df[0])
test_aa_enc = np.load(args.test_aa_enc[0], allow_pickle = True)
test_c1 = np.load(args.test_C1[0], allow_pickle = True)
test_c2 = np.load(args.test_C2[0], allow_pickle = True)
#params_file = args.params_file[0]
out_dir = args.out_dir[0]

#bins
bins = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
bins = np.expand_dims(bins, axis=0) #required for later multiplication

#Get labels
y_train = np.asarray(train_df['log50k'])
y_test = np.asarray(test_df['log50k'])
#Get y_test in binary
#When classifying the peptides into binders and non-binders
#for calculation of the AUC values for instance, a threshold of 500 nM is used.
#This means that peptides with log50k transformed binding
#affinity values greater than 0.426 are classified as binders.

t = 1-np.log(500)/np.log(50000)
true_binary = []
for i in range(len(y_test)):
    if y_test[i]>=t:
        true_binary.append(1)
    else:
        true_binary.append(0)

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


#Encode alleles
#Each chain of the alleles is represented by 97 aa
X2_train = []
X3_train = []
for i in range(len(train_c1)):
    X2_train.append(np.eye(20)[train_c1[i]])
    X3_train.append(np.eye(20)[train_c2[i]])
X2_train = np.array(X2_train) #convert to arrays
X3_train = np.array(X3_train)

X2_test = []
X3_test = []
for i in range(len(test_c1)):
    X2_test.append(np.eye(20)[test_c1[i]])
    X3_test.append(np.eye(20)[test_c2[i]])
X2_test = np.array(X2_test) #convert to arrays
X3_test= np.array(X3_test)
#Tensorboard for logging and visualization
log_name = str(time.time())
tensorboard = TensorBoard(log_dir=out_dir+log_name)


######MODEL######
#Parameters
#net_params = read_net_params(params_file)
input_dim1 = (37,20) #20 AA*25 residues
input_dim2 = (94, 20) #Shape of allele encodings
input_dim3 = (94, 20) #Shape of allele encodings
num_classes = bins.size #add +1 if categorical
kernel_size =  3 #Should probably vary this also #The length of the conserved part that should bind to the binding grove
dilation_rate = 2
#Variable params
filters =  50#int(net_params['filters']) # Dimension of the embedding vector.
batch_size = 32 #int(net_params['batch_size'])

#Attention size
attention_size = filters*(37-kernel_size+1)+filters*(94-kernel_size+1)
#lr opt
find_lr = 0
#loss
loss = 'bin_loss'#'categorical_crossentropy'
#LR schedule
step_size = 5 #should increase alot - maybe 5?
num_cycles = 3
num_epochs = step_size*2*num_cycles
num_steps = int(len(X1_train)/batch_size)
max_lr = 0.003
min_lr = max_lr/10
lr_change = (max_lr-min_lr)/step_size  #(step_size*num_steps) #How mauch to change each batch
lrate = min_lr
#MODEL
in_1 = keras.Input(shape = input_dim1)
in_2 = keras.Input(shape = input_dim2)
in_3 = keras.Input(shape = input_dim3)

def resnet(x, num_res_blocks):
	"""Builds a resnet with 1D convolutions of the defined depth.
	"""


    	# Instantiate the stack of residual units
    	#Similar to ProtCNN, but they used batch_size = 64, 2000 filters and kernel size of 21
	for res_block in range(num_res_blocks):
		batch_out1 = BatchNormalization()(x) #Bacth normalize, focus on segment
		activation1 = Activation('relu')(x)
		conv_out1 = Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, input_shape=input_dim2, padding ="same")(activation1)
		batch_out2 = BatchNormalization()(conv_out1) #Bacth normalize, focus on segment
		activation2 = Activation('relu')(conv_out1)
        #Downsample - half filters
		conv_out2 = Conv1D(filters = int(filters/2), kernel_size = kernel_size, dilation_rate = 1, input_shape=input_dim2, padding ="same")(activation2)
		x = Conv1D(filters = int(filters/2), kernel_size = kernel_size, dilation_rate = 1, input_shape=input_dim2, padding ="same")(x)
		x = add([x, conv_out2]) #Skip connection



	return x



#Convolution on peptide encoding
p = Conv1D(filters = filters, kernel_size = kernel_size, padding ="valid")(in_1) #Same means the input will be zero padded, so the convolution output can be the same size as the input.
#take steps of 1 doing kernel_size convolutions using filters number of filters
p = BatchNormalization()(p) #Bacth normalize, focus on segment
p = Activation('relu')(p) #try elu activation also?


#Convolution on allele encoding C1
c1 = Conv1D(filters = filters, kernel_size = kernel_size, padding ="valid")(in_2) #Same means the input will be zero padded, so the convolution output can be the same size as the input.
#take steps of 1 doing kernel_size convolutions using filters number of filters
c1 = BatchNormalization()(c1) #Bacth normalize, focus on segment
c1 = Activation('relu')(c1) #try elu activation also?
c1 = MaxPooling1D(pool_size=2)(c1)
#Convolution on allele encoding C2
c2 = Conv1D(filters = filters, kernel_size = kernel_size, padding ="valid")(in_3) #Same means the input will be zero padded, so the convolution output can be the same size as the input.
#take steps of 1 doing kernel_size convolutions using filters number of filters
c2 = BatchNormalization()(c2) #Bacth normalize, focus on segment
c2 = Activation('relu')(c2)
c2 = MaxPooling1D(pool_size=2)(c2)

#Flatten for concatenation
flat1 = Flatten()(p)  #Flatten
flat2 = Flatten()(c1)  #Flatten
flat3 = Flatten()(c2)  #Flatten

x = concatenate([flat1, flat2, flat3])


#Attention layer
#Attention layer - information will be redistributed in the backwards pass
attention = Dense(1, activation='tanh')(x) #Normalize and extract info with tanh activated weight matrix (hidden attention weights)
attention = Flatten()(attention) #Make 1D
attention = Activation('tanh')(attention) #Softmax on all activations (normalize activations)
attention = RepeatVector(attention_size)(attention) #Repeats the input "num_nodes" times.
attention = Permute([2, 1])(attention) #Permutes the dimensions of the input according to a given pattern. (permutes pos 2 and 1 of attention)

sent_representation = multiply([x, attention]) #Multiply input to attention with normalized activations
sent_representation = Lambda(lambda xin: keras.backend.sum(xin, axis=-2), output_shape=(attention_size,))(sent_representation) #Sum all attentions

#Dense final layer for classification
probabilities = Dense(num_classes, activation='softmax')(x)

if loss == 'bin_loss':
    bins_K = variable(value=bins)

    #Multiply the probabilities with the bins --> gives larger freedom in assigning values
    def multiply(x):
        return tf.matmul(x, bins_K,transpose_b=True)

    pred_vals = Lambda(multiply)(probabilities)
    out_vals = pred_vals
else:
    out_vals = probabilities

#Custom loss
def bin_loss(y_true, y_pred):
  #Shold make this a log loss
	g_loss = np.absolute(y_true-y_pred) #general, compare difference
	kl_loss = keras.losses.kullback_leibler_divergence(y_true, y_pred) #better than comparing to gaussian?
	sum_kl_loss = keras.backend.sum(kl_loss, axis =0)
	sum_g_loss = keras.backend.sum(g_loss, axis =0)
	sum_g_loss = sum_g_loss*10 #This is basically a loss penalty
	loss = sum_g_loss+sum_kl_loss
	return loss


#Model: define inputs and outputs
model = Model(inputs = [in_1, in_2, in_3], outputs = out_vals) #probabilities)#
opt = optimizers.Adam(clipnorm=1., lr = lrate) #remove clipnorm and add loss penalty - clipnorm works better
model.compile(loss=bin_loss,
              optimizer=opt)



if find_lr == True:
  lr_finder = LRFinder(model)

  X_train = [X1_train, X2_train, X3_train]
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
    pred = model.predict([X1_test, X2_test, X3_test])
    #pred = np.argmax(pred, axis = 1)
    pred_binary = []

    for i in range(len(pred)):
        if pred[i]>=t:
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
model.fit(x = [X1_train, X2_train, X3_train],
            y = y_train,
            batch_size = batch_size,
            epochs=num_epochs,
            validation_data = [[X1_test, X2_test, X3_test], y_test],
            shuffle=True, #Dont feed continuously
            callbacks=callbacks)
