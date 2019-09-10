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
from tensorflow.keras.layers import Activation, RepeatVector, Permute, Multiply, Lambda, GlobalAveragePooling1D
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
parser.add_argument('params_file', nargs=1, type= str,
                  default=sys.stdin, help = 'Path to file with net parameters')
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
aa_enc = np.load(args.aa_encodings[0])
params_file = args.params_file[0]
out_dir = args.out_dir[0]

#Assign data and labels
#Read df
df = pd.read_csv(df_path)
#Get pic50 values
pic50 = df['pic50']
bins = np.array([-5.47712125, -4.77712125, -4.07712125, -3.37712125, -2.7,
       -1.97712125, -1.27712125, -0.57712125,  0.12287875,  0.82287875])
#Bin the pic50 values
y = np.digitize(pic50,bins)
X1 =[]
max_length = 25 #Goes from 15-25
for enc in aa_enc:
    X1.append(pad_cut(enc, max_length))

X2 = np.asarray(df['allele_enc'])
#5 different splits
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(X, y):
    X1_train, X1_test = X1[train_index], X1[test_index]
    X2_train, X2_test = X2[train_index], X2[test_index]
    y_train, y_test = y[train_index], y[test_index]
bins = np.expand_dims(bins, axis=0)
#Tensorboard for logging and visualization
log_name = str(time.time())
tensorboard = TensorBoard(log_dir=out_dir+log_name)


######MODEL######
#Parameters
net_params = read_net_params(params_file)
input_dim1 = (15,20) #20 AA*15 residues
input_dim2 = (1,42) #42 types of alleles
num_classes = max(bins.shape)
kernel_size =  20 #The number of aa
#The length of the binding grove

#Variable params
base_epochs = 10#int(net_params['base_epochs'])
finish_epochs = 2 #int(net_params['finish_epochs'])
filters =  10#int(net_params['filters']) # Dimension of the embedding vector.

batch_size = 32 #int(net_params['batch_size'])
#lr opt
find_lr = False
#LR schedule
step_size = 5 #should increase alot - maybe 5?
num_cycles = 3
num_epochs = step_size*2*num_cycles
num_steps = int(len(train_df)/batch_size)
max_lr = 0.0015
min_lr = max_lr/10
lr_change = (max_lr-min_lr)/step_size  #(step_size*num_steps) #How mauch to change each batch
lrate = min_lr
#MODEL
in_1 = keras.Input(shape = input_dim1)
in_2 = keras.Input(shape = input_dim2)

#Convolution on aa encoding
in_1_conv = Conv1D(filters = filters, kernel_size = kernel_size, input_shape=input_dim1, padding ="same")(in_1) #Same means the input will be zero padded, so the convolution output can be the same size as the input.

batch_out1 = BatchNormalization()(x) #Bacth normalize, focus on segment
		activation1 = Activation('relu')(batch_out1)
		conv_out1 = Conv1D(filters = filters, kernel_size = kernel_size, dilation_rate = dilation_rate, input_shape=input_dim, padding ="same")(activation1)
		batch_out2 = BatchNormalization()(conv_out1) #Bacth normalize, focus on segment
		activation2 = Activation('relu')(batch_out2)
        #Downsample - half filters
		conv_out2 = Conv1D(filters = int(filters/2), kernel_size = kernel_size, dilation_rate = 1, input_shape=input_dim, padding ="same")(activation2)
		x = Conv1D(filters = int(filters/2), kernel_size = kernel_size, dilation_rate = 1, input_shape=input_dim, padding ="same")(x)
		x = add([x, conv_out2]) #Skip connection



	return x


#Output (batch, steps(len), filters), filters = channels in next
x1 = resnet(in_1_conv, num_res_blocks)
x2 = resnet(in_2_conv, num_res_blocks)

#Maxpool along sequence axis
maxpool1 = MaxPooling1D(pool_size=seq_length)(x1)
maxpool2 = MaxPooling1D(pool_size=seq_length)(x2)
#cat = concatenate([maxpool1, maxpool2]) #Cat convolutions

flat1 = Flatten()(maxpool1)  #Flatten
flat2 = Flatten()(maxpool2)  #Flatten

#Should have sum of two losses:
#1. How closely the predicted lddt matches the real one
#2. How closely the probability distribution of the bins match a gaussian distribution (kl divergence)
#Perhaps I should try to predict the deviation from the mean, since the scores are so centered around the mean.
# Add a customized layer to compute the absolute difference between the encodings
L1_layer = Lambda(lambda tensors:abs(tensors[0] - tensors[1]))
L1_distance = L1_layer([flat1, flat2])

# Add a customized layer to compute the absolute difference between the encodings
#L2_layer = Lambda(lambda tensors:keras.backend.sqrt(keras.backend.square(tensors[0] - tensors[1])))
#L2_distance = L2_layer([flat1, flat2])
#Dense final layer for classification

probabilities = Dense(num_classes, activation='softmax')(L1_distance)
bins_K = variable(value=bins)

def multiply(x):
  return tf.matmul(x, bins_K,transpose_b=True)

pred_vals = Lambda(multiply)(probabilities)
#The length of the validation data must be a multiple of batch size!
#Other wise you will have shape mismatches

#Custom loss
def bin_loss(y_true, y_pred):
  #Shold make this a log loss
        g_loss = mean_absolute_error(y_true, y_pred) #general, compare difference
	#log_g_loss = keras.backend.log(g_loss/100)
  #Gauss for loss
	#gauss = keras.backend.random_normal_variable(shape=(batch_size, 1), mean=0.7, scale=0.3) # Gaussian distribution, scale: Float, standard deviation of the normal distribution.
        kl_loss = keras.losses.kullback_leibler_divergence(y_true, y_pred) #better than comparing to gaussian
        sum_kl_loss = keras.backend.sum(kl_loss, axis =0)
        sum_g_loss = keras.backend.sum(g_loss, axis =0)
        sum_g_loss = sum_g_loss*alpha #This is basically a loss penalty


        # #Normalize due to proportion
        # kl_p = sum_kl_loss/(sum_g_loss+sum_kl_loss)
        # g_p = sum_g_loss/(sum_g_loss+sum_kl_loss)

        # sum_kl_loss = sum_kl_loss/kl_p
        # sum_g_loss = sum_g_loss/g_p
        loss = sum_g_loss+sum_kl_loss
        #Scale with R? loss = loss/R - on_batch_end
  	#Normalize loss by percentage contributions: divide by contribution
  	#Write batch generator to avoid incompatibility in shapes
  	#problem at batch end due to kongruens
        return loss

#Custom validation loss
class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            diff = [y_pred[i]-y_valid[i] for i in range(len(y_valid))]
            score = np.average(np.absolute(diff))
            #Pearson correlation coefficient
            R,pval = stats.pearsonr(y_valid, y_pred.flatten())
            R = np.round(R, decimals = 3)
            score = np.round(score, decimals = 3)
            print('epoch: ',epoch, ' score: ', score, ' R: ', R)
            valid_metrics[0].append(score)
            valid_metrics[1].append(R)
            np.savetxt(out_dir+'validpred_'+str(epoch)+'_'+str(score)+'_'+str(R)+'.txt', y_pred)

#Model: define inputs and outputs
model = Model(inputs = [in_1, in_2], outputs = pred_vals)
opt = optimizers.Adam(clipnorm=1., lr = lrate) #remove clipnorm and add loss penalty - clipnorm works better
model.compile(loss=bin_loss,
              optimizer=opt)



if find_lr == True:
  lr_finder = LRFinder(model)
  #Validation data
  train_enc1 = [pad_cut(np.eye(22)[literal_eval(x)], 300, 22) for x in [*train_df['enc1']]]
  train_enc2 = [pad_cut(np.eye(22)[literal_eval(x)], 300, 22) for x in [*train_df['enc2']]]
  X_train = [np.asarray(train_enc1), np.asarray(train_enc2)]
  y_train = np.asarray(train_df['global_lddt'])
  lr_finder.find(X_train, y_train, start_lr=0.00001, end_lr=1, batch_size=batch_size, epochs=1)
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
  # def on_batch_end(self, batch, logs={}):
  #   self.lr = self.lr + self.lr_change
    keras.backend.set_value(self.model.optimizer.lr, self.lr)


#Lrate
lrate = LRschedule()


#Checkpoint
filepath=out_dir+"weights-{epoch:02d}-.hdf5"
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='max')

#Validation data
ival = IntervalEvaluation(validation_data=(X_valid, y_valid), interval=1)

#Summary of model
print(model.summary())

callbacks=[lrate, tensorboard, checkpoint, ival]
#Fit model
#Should shuffle uid1 and uid2 in X[0] vs X[1]
model.fit_generator(generate(batch_size),
            steps_per_epoch=num_steps,
            epochs=num_epochs,
            #validation_data = [X_valid, y_valid],
            shuffle=True, #Dont feed continuously
            callbacks=callbacks)


#Save validation metrics
valid_metrics = np.asarray(valid_metrics)
np.savetxt('valid_metrics.txt', valid_metrics)
