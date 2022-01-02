# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:21:22 2021

@author: sofiane
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adagrad
#print(tf.__version__)

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from ranking_loss import loss


#Important: script works with Tensorflow 2.7.0



def load_one_batch(Batch_size, Abnormal_train, Normal_train):
    """Inputs:
        All Abnormal and Normal videos features of the training dataset 
        np.array (nb_videos, 32, 512)
        
        Outputs:
            One training Batch 
            np array (Batch_size x 32, 512)
    """
    
    #to store one Batch of training data
    Batch_data = np.zeros((32*Batch_size, 512), dtype = np.float32)
    
    #to store the labels of one Batch
    Batch_labels = np.zeros(32*Batch_size, dtype='uint8')
    
    #Number of abnormal and normal videos in one batch          
    n_exp = Batch_size // 2  
    
    #number of Abnormal videos in the training dataset
    Num_abnormal = len(Abnormal_train)
    
    #number of Normal videos in the training dataset
    Num_Normal = len(Normal_train)
       
    #One Batch: Selecting randomly 30 Abnormal and 30 Normal training videos
    #indexes of the select abnormal and normal videos
    Abnormal_idx = np.random.permutation(Num_abnormal)
    Abnormal_idx = Abnormal_idx[Num_abnormal-n_exp:]
    Normal_idx = np.random.permutation(Num_Normal)
    Normal_idx = Normal_idx[Num_Normal-n_exp:]
      
    #30 abnormal videos randomly selected
    Batch_data_abnormal = Abnormal_train[Abnormal_idx, : , : ]
    Batch_data_abnormal = Batch_data_abnormal.reshape((n_exp*32, 512))
    
    #30 normal videos randomly selected
    Batch_data_normal = Normal_train[Normal_idx, : , : ]
    Batch_data_normal = Batch_data_normal.reshape((n_exp*32, 512))
    
    #Batch: 60 videos randomly selected
    Batch_data[0:n_exp*32,:] = Batch_data_abnormal
    Batch_data[n_exp*32:,:] = Batch_data_normal
    
    #Labels
    # All instances of abnormal videos are labeled 0
    # All instances of Normal videos are labeled 1
    for i in range(0, 32*Batch_size):
        
            if i < n_exp*32:
                Batch_labels[i] = int(0)  
                
            if i > n_exp*32-1:
                Batch_labels[i] = int(1)   

    
    return Batch_data, Batch_labels



def network():
    
    model = Sequential()
    
    initializer = tf.keras.initializers.GlorotNormal()
    #regularization of the weights by L2: add penality to loss during optimization       
    model.add(Dense(128, activation='relu',kernel_initializer = initializer, kernel_regularizer = l2(0.01), input_shape = (512,)))
    model.add(Dropout(0.6))
    model.add(Dense(32, kernel_initializer = initializer, kernel_regularizer = l2(0.01)))
    model.add(Dropout(0.6))
    model.add(Dense(1,activation='sigmoid', kernel_initializer = initializer, kernel_regularizer = l2(0.01)))
    
    optimizer = Adagrad(learning_rate=0.001)
    model.compile(optimizer= optimizer, loss= loss)
    #model.summary()
    
    return model


def train(Batch_size, num_iters, network, Abnormal_train, Normal_train, Abnormal_test, Normal_test):
    
    loss_train_curve = []
    loss_val_curve = []
    iterations = 0
    
    #training per batch
    for it_num in range(num_iters):
        
        inputs, targets = load_one_batch(Batch_size, Abnormal_train, Normal_train)  
        batch_loss = network.train_on_batch(inputs, targets)
        
        #training loss
        loss_train_curve.append(batch_loss)
        
        #validation loss        
        batch_loss_val = compute_val_loss(Batch_size, network, Abnormal_test, Normal_test)
        loss_val_curve.append(batch_loss_val)

        iterations+=1
        
        if iterations % 500 == 1:
            print('iteration: ',iterations,'/',num_iters)
            print('loss train: ', loss_train_curve[it_num])
            print('loss val: ', loss_val_curve[it_num])  
    
    return loss_train_curve, loss_val_curve


def compute_val_loss(Batch_size, network, Abnormal_test, Normal_test):
    """load validation set batch per batch and return validation loss"""
    
    nb_vid_abn = len(Abnormal_test)
    nb_vid_nor = len(Normal_test)
    nb_vid = min(nb_vid_abn, nb_vid_nor)
    
    #30 videos abnormal and 30 videos normal in one batch
    n_exp = Batch_size // 2
    
    loss_val = []
    
    Batch_data = np.zeros((32*Batch_size, 512), dtype = np.float32)
    Batch_labels = np.zeros(32*Batch_size, dtype='uint8')
    
    Batch_labels[0:n_exp*32] = np.zeros(n_exp*32, dtype='uint8')
    Batch_labels[n_exp*32:] = np.ones(n_exp*32, dtype='uint8')
    
    for i in range(0, nb_vid - n_exp + 1, n_exp):
        
        #batch (nb_videos, 32, 512)
        batch_abn = Abnormal_test[i:i+n_exp]
        batch_norm = Normal_test[i:i+n_exp]
        
        #batch (nb_videos*32, 512)
        batch_abn = batch_abn.reshape((n_exp*32, 512))
        batch_norm = batch_norm.reshape((n_exp*32, 512))
        
        Batch_data[0:n_exp*32,:] = batch_abn
        Batch_data[n_exp*32:,:] = batch_norm
        
        outputs_val = network.predict_on_batch(Batch_data)
        
        batch_loss_val = loss(Batch_labels, outputs_val)
        
        loss_val.append(batch_loss_val.numpy())
    
    #print('nb batch validation set: ', len(loss_val))
    
    return np.mean(loss_val)
        

def display_loss_curve(loss_train_curve, loss_val_curve):
    
    x = np.arange(0,len(loss_train_curve))
    plt.figure()
    plt.plot(x[0:-1:100],loss_train_curve[0:-1:100],'b', label = 'training loss')
    plt.plot(x[0:-1:100],loss_val_curve[0:-1:100],'r', label = 'validation loss')
    plt.legend()

