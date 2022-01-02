# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 01:00:11 2021

@author: sofiane
"""


#Inspired by : https://github.com/fluque1995/tfm-anomaly-detection/blob/master/original_model/train_classifier.py


import tensorflow as tf 
import numpy as np
import keras.backend as K

tf.config.run_functions_eagerly(True)

@tf.autograph.experimental.do_not_convert
def loss(y_true, y_pred):

    y_true = K.reshape(y_true, [-1])
    y_pred = K.reshape(y_pred, [-1])
    n_seg = 32
    nvid = 60
    n_exp = int(nvid / 2)

    mu1, mu2 = 0.00008, 0.00008

    max_scores_list = []
    min_scores_list = []

    l1_scores_list = []
    l2_scores_list = []
    l3_scores_list = []
    l4_scores_list = []
    
    #temporal and sparsity constraints
    temporal_constrains_list = []
    sparsity_constrains_list = []
    
    #store abnormal scores sorted by descending order
    desc_scores_abn = []
    
    #Abnormal BAG
    for i in range(0, n_exp, 1):
        
        video_predictions = y_pred[i*n_seg:(i+1)*n_seg]

        max_scores_list.append(K.max(video_predictions))
        min_scores_list.append(K.min(video_predictions))

        temporal_constrains_list.append(
            K.sum(K.pow(video_predictions[1:] - video_predictions[:-1], 2))
        )
        sparsity_constrains_list.append(K.sum(video_predictions))

        #Descending order of the Abnormal instances Group
        liste_score_positive_group = video_predictions.numpy()
        liste_score_positive_group_asc = np.sort(liste_score_positive_group)
        liste_score_positive_group_dsc = liste_score_positive_group_asc[::-1]
        score_positive_groupe_desc = K.stack(liste_score_positive_group_dsc)
        desc_scores_abn.append(score_positive_groupe_desc)
        
    #Normal BAG
    for j in range(n_exp, 2*n_exp, 1):
        
        video_predictions = y_pred[j*n_seg:(j+1)*n_seg]
        max_scores_list.append(K.max(video_predictions))

    max_scores = K.stack(max_scores_list)
    min_scores = K.stack(min_scores_list)

    temporal_constrains = K.stack(temporal_constrains_list)
    sparsity_constrains = K.stack(sparsity_constrains_list)
    
    #compute sub-losses l1,l2,l3,l4
    for ii in range(0, n_exp, 1):

        max_l1 = K.maximum(1 - max_scores[:n_exp] + max_scores[n_exp+ii], 0)
        l1_scores_list.append(K.sum(max_l1))

        max_l2 = K.maximum(1 - max_scores[ii] + min_scores[ii], 0)
        #max_l2 = K.maximum(1 - max_scores[ii] + min_scores[0:n_exp], 0)
        l2_scores_list.append(K.sum(max_l2))

        max_l3 = K.maximum(1 - desc_scores_abn[ii][1] + max_scores[n_exp+ii], 0)
        #max_z3 = K.maximum(1 - score_positive_groupe_desc[1] + max_scores[n_exp+ii], 0)
        l3_scores_list.append(max_l3)

        max_l4 = K.maximum(1 - desc_scores_abn[ii][2] + max_scores[n_exp+ii], 0)
        #max_z4 = K.maximum(1 - score_positive_groupe_desc[2] + max_scores[n_exp+ii], 0)
        l4_scores_list.append(max_l4)


    l1_scores = K.stack(l1_scores_list)
    l1 = K.mean(l1_scores)

    l2_scores = K.stack(l2_scores_list)
    l2 = K.mean(l2_scores)

    l3_scores = K.stack(l3_scores_list)
    l3 = K.mean(l3_scores)

    l4_scores = K.stack(l4_scores_list)
    l4 = K.mean(l4_scores)

    return l1 + l2 + l3 + l4 + mu1*K.sum(temporal_constrains) + mu2*K.sum(sparsity_constrains)
