import tensorflow as tf 
import numpy as np

def custom_objective(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1]) #permet d'applatir les données 
    y_pred = tf.reshape(y_pred, [-1]) #permet d'applatir les données
    n_seg = 32
    nvid = 60 
    n_exp = int(nvid / 2)
    
    min_scores_list = []
    max_scores_list = []
    temporal_constrains_list = []
    sparsity_constrains_list = []
    list_difference = []
    l1_list = []
    l2_list = []
    l3_list = []
    l4_list = []
    
    mu1 = 0.00008
    mu2 = 0.00008

    for i in range(0, n_exp, 1):
        
        video_predictions = y_pred[i*n_seg:(i+1)*n_seg]
        max_scores_list.append(max(video_predictions))
        min_scores_list.append(min(video_predictions))
        for j in range(len(video_predictions)-1) : 
            list_difference.append = video_predictions[j] - video_predictions[j+1]
        temporal_constrains_list.append(sum(tf.pow(list_difference, 2)))
        
    for j in range(n_exp, 2*n_exp, 1):
        
        video_predictions = y_pred[j*n_seg:(j+1)*n_seg]
        max_scores_list.append(max(video_predictions))
        sparsity_constrains_list.append(sum(video_predictions))
        
    #Creation du tenseur de la liste des scores du groupe positif par ordre décroissant
    liste_score_positive_group = y_pred[0:n_exp].numpy()
    liste_score_positive_group_asc = np.sort(liste_score_positive_group)
    liste_score_positive_group_dsc = liste_score_positive_group_asc[::-1]
    score_positive_groupe_desc = tf.stack(liste_score_positive_group_dsc)
    
    
    score_positive_groupe_desc = tf.stack(liste_score_positive_group_dsc)
    max_scores = tf.stack(max_scores_list)
    min_scores = tf.stack(min_scores_list)
    temporal_constrains = tf.stack(temporal_constrains_list)
    sparsity_constrains = tf.stack(sparsity_constrains_list)

    for ii in range(0, n_exp, 1):
        
    #Calcul de la liste des l1 :
        l1_list.append(0,1 - max_scores[ii] + max_scores[n_exp+ii])
    #Calcul de la liste des l2:
        l2_list.append(0,1 - max_scores[ii] + min_scores[ii])        
    #Calcul de la liste des l3:     
        l3_list.append(0,1 - score_positive_groupe_desc[1] + min_scores[n_exp+ii])
    #Calcul de la liste des l4:     
        l4_list.append(0,1 - score_positive_groupe_desc[2] + min_scores[n_exp+ii])       
        
        

    return max(l1_list) + max(l2_list) + max(l3_list) + max(l4_list) + mu1*sum(temporal_constrains) + mu2*sum(sparsity_constrains)
    
    

    
