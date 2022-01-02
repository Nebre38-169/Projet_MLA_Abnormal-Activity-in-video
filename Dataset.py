# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:21:22 2021

@author: sofiane
"""

import os
import numpy as np

#function to load the dataset
def load_data(abnormal_features_path, normal_features_path):
    """
    Inputs: 
        folder that contains the abnormal and normal extracted features
    
    Ouputs: 
        array (nb_abnormal_videos, 32, 512): abnormal_data_train that contains all abnormal features of the training dataset UFC Crime
        array (nb_normal_videos, 32, 512): normal_data_train that contains all normal features of the training dataset UFC Crime
        abnormal_data_ignore and normal_data_ignore: list of indexes of the videos too short to have 32 segments
    """
    
    #print('Loading Training Data ...')
    
    #####Load Normal videos features ####
    
    #to store all normal features of the training dataset
    normal_data_train = []
    
    #to ignore videos too short (not enough frames to have 32 segments)
    normal_data_ignore = []
    
    List_Normal_Features = sorted(os.listdir(normal_features_path))
    
    
    print('normal videos loading ...')
    #load normal videos data
    for i in range(len(List_Normal_Features)):
        
        #read features of each video
        one_video = normal_features_path + '/' + List_Normal_Features[i]
        #print('video : ', one_video)
    
        one_video_features = np.loadtxt(one_video)
        one_video_features = np.asarray(one_video_features, dtype= np.float32)
        
        if one_video_features.shape[0] == 0:
            tmp = np.linalg.norm(one_video_features)
        else:
            tmp = np.linalg.norm(one_video_features, axis=1)
            
        #concatenate the videos features
        if len(one_video_features) == 32 and 0 not in tmp :
            
            #normalize data
            #one_video_features = one_video_features/ np.linalg.norm(one_video_features)
            
            assert(0 not in tmp)
            tmp = np.repeat(tmp,512).reshape(32,512)
            feat_norm = one_video_features/tmp
            
            normal_data_train.append(feat_norm)

        #to store index of the normal video too short (ignored)   
        else:
            normal_data_ignore.append(i)

    
    #####Load Abnormal videos features#####
    
    abnormal_data_train =[] #to store training abnormal features 
    
    #to ignore videos too short (not enough frames to have 32 segments)
    abnormal_data_ignore = []
    cpt_abnormal_files = 0

    #list of the sub-directories
    List_Abnormal_Features = sorted(os.listdir(abnormal_features_path))
    
    print('Abnormal videos loading ...')
    for sub_dir in List_Abnormal_Features:
        
        List_sub_dir_abnormal = sorted(os.listdir(abnormal_features_path + '/' + sub_dir))
        
        for i in range(len(List_sub_dir_abnormal)):
        
            #read features of each video
            #print('video : ', one_video)
            one_video = abnormal_features_path + '/' + sub_dir + '/'+ List_sub_dir_abnormal[i]
            one_video_features = np.loadtxt(one_video)
            one_video_features = np.asarray(one_video_features, dtype= np.float32)
            
            if one_video_features.shape[0] == 0:
                tmp = np.linalg.norm(one_video_features)
            else:
                tmp = np.linalg.norm(one_video_features, axis=1)
        
            #concatenate the videos features
            if len(one_video_features) == 32 and 0 not in tmp:
                #normalize data
                #one_video_features = one_video_features/ np.linalg.norm(one_video_features)
                
                assert (0 not in tmp)
                tmp = np.repeat(tmp,512).reshape(32,512)
                feat_norm = one_video_features/tmp
                
                abnormal_data_train.append(feat_norm)

            #to store index of the abnormal video too short (ignored)
            else:
                abnormal_data_ignore.append(cpt_abnormal_files)

            cpt_abnormal_files+=1
   
    abnormal_data_train = np.stack(abnormal_data_train)
    normal_data_train = np.stack(normal_data_train)
    
    
    return abnormal_data_train, normal_data_train, abnormal_data_ignore, normal_data_ignore

