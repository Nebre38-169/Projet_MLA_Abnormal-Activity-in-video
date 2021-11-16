# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 21:44:09 2021

@author: antoi
"""

import torch

def custom_loss(output, target):
    #output est l'équivalent de w.phi(xk)
    #Target est l'équivalent de Yci (mais étalé sur les 32 segments de chaque vidéo)
    
    n_segment = 32 #Nombre de segment par vidéo
    M = int(output/n_segment) #Nombre de vidéo ou de sachet
    
    #On récupérer les bias :
    bias = m.state_dict()['bias'] #je suis pas sur de ces lignes, mais ici m est le model
    weight = m.state_dict()['weigth'] #Mais je crois que cette méthode ne permet d'acceder qu'a un niveau des poids/bia
    
    sum_value = 0
    for i in range(0,n_segment-1):
        #Pour l'instant je prend le bias le plus grand mais je suis pas sur de ça
        sum_value += hinge_loss(output[n_segmet*i,n_segment*(i+1)], target[n_segmet*i,n_segment*(i+1)],max(bias[n_segmet*i,n_segment*(i+1)])
    mean = sum_value/M
    return mean + 0.5*torch.norm(weight)**2

def hinge_loss(output_bag, target_bag,bia):
    #Prend un seul sac de 32 segment et renvoie la valeur de la fonction a^
    maxWPhiXk = max(output_bag)
    Yci = max(target_bag)
    return max(0,1-Yci*(maxWPhiXk-bia))

    