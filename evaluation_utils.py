# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 01:00:11 2021

@author: sofiane
"""

import numpy as np
import cv2
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

#Script which contains utils function for the evaluation the anomaly detector

#return Temporal Ground truths and predictions necessary to compute RoC curve and AuC
def temporal_gt_pred_abnormal(temporal_annotation_abnormal, abnormal_videos_dir, abnormal_scores, abnormal_ignore_idx):
    
    f = open(temporal_annotation_abnormal, "r")
    test_abnormal_files = f.readlines()
    
    gts_abnormal, preds_abnormal = [], []
    
    idx_files = [x for x in range(len(test_abnormal_files)) if x not in abnormal_ignore_idx]

    for i in range(len(idx_files)):
        #print('i:',i)

        row = test_abnormal_files[idx_files[i]].split("\n")[0].split(" ")
        #print('row: ',row)
        anomaly_start_1 =  int(row[4])
        anomaly_end_1 =  int(row[6])
        anomaly_start_2 =  int(row[8])
        anomaly_end_2 =  int(row[10])

        #compute nmber of frames in the video
        video_path = abnormal_videos_dir + '/' + row[2] + '/' + row[0]
        cap = cv2.VideoCapture(video_path)
        assert(cap.isOpened()== True) # Check if camera opened successfully
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
        #Temporal Ground truth associated to the current abnormal video:
        #1 for the frames with abnormal behaviour
        #0 for the frames with normal behaviour
        curr_gts = np.zeros(nb_frames)
        if anomaly_start_1 != -1 and anomaly_end_1 != -1:
            curr_gts[anomaly_start_1:anomaly_end_1+1] = 1

        if anomaly_start_2 != -1 and anomaly_end_2 != -1:
            curr_gts[anomaly_start_2:anomaly_end_2+1] = 1
  
        gts_abnormal.append(curr_gts)

        #Extrapolate predictions for each frame: one score per frame
        curr_preds = abnormal_scores[i,:]
        preds_abnormal.append(extrapolate(curr_preds,nb_frames))

    gts_abnormal = np.concatenate(gts_abnormal)
    preds_abnormal = np.concatenate(preds_abnormal)
   
    return gts_abnormal, preds_abnormal


def temporal_gt_pred_normal(temporal_annotation_normal, normal_videos_dir, normal_scores, normal_ignore_idx):

    f = open(temporal_annotation_normal, "r")
    test_normal_files = f.readlines()

    gts_normal, preds_normal = [], []
    
    idx_files = [x for x in range(len(test_normal_files)) if x not in normal_ignore_idx]

    for i in range(len(idx_files)):

        row = test_normal_files[idx_files[i]].split("\n")[0].split(" ")
        anomaly_start_1 =  int(row[4])
        anomaly_end_1 =  int(row[6])
        anomaly_start_2 =  int(row[8])
        anomaly_end_2 =  int(row[10])

        #compute nmber of frames in the video
        video_path = normal_videos_dir + '/' + row[0]
        cap = cv2.VideoCapture(video_path)
        assert(cap.isOpened()== True) # Check if camera opened successfully
        nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
        #Temporal Ground truth associated to the current normal video:
        #1 for the frames with abnormal behaviour
        #0 for the frames with normal behaviour
        curr_gts = np.zeros(nb_frames)
        
        if anomaly_start_1 != -1 and anomaly_end_1 != -1:
            curr_gts[anomaly_start_1:anomaly_end_1+1] = 1

        if anomaly_start_2 != -1 and anomaly_end_2 != -1:
            curr_gts[anomaly_start_2:anomaly_end_2+1] = 1
  
        gts_normal.append(curr_gts)

        #Extrapolate predictions for each frame: one score per frame
        curr_preds = normal_scores[i,:]
        preds_normal.append(extrapolate(curr_preds,nb_frames))

    gts_normal = np.concatenate(gts_normal)
    preds_normal = np.concatenate(preds_normal)

    return gts_normal, preds_normal


#ROC Curve
def display_roc_curve(fpr, tpr):
    
    plt.figure()
    plt.title("ROC curve")
    plt.plot(fpr, tpr, 'b', label = "ROC")
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')


#util to have an anomaly score per video frame: score (1,32) -> (1,nb_frames)
def extrapolate(outputs, num_frames):
    
    extrapolated_outputs = []
    extrapolation_indicies = np.round(np.linspace(0, len(outputs) - 1, num=num_frames))
    for index in extrapolation_indicies:
        extrapolated_outputs.append(outputs[int(index)])
    return np.array(extrapolated_outputs)


#Display the anomaly scores predicted for an Abnormal video
def qualitative_test_abnormal(video_name, video_path, features_path, anomaly_start_frame, anomaly_end_frame, anomaly_detector):
    
    #features
    features_video = np.loadtxt(features_path)
    #(32,512) features
    features_video = np.asarray(features_video, dtype= np.float32)
    tmp = np.linalg.norm(features_video, axis=1)
    tmp = np.repeat(tmp,512).reshape(32,512)
    
    #normalize features
    feat_norm = features_video/tmp

    #scores
    score = anomaly_detector.predict_on_batch(feat_norm)
    #score (32,) -> one score per segment video
    score = score.reshape(32)

    #compute nmber of frames in the video
    cap = cv2.VideoCapture(video_path)
    assert(cap.isOpened()== True) # Check if camera opened successfully
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #one anomaly score per frame
    score_frames = extrapolate(score, nb_frames)

    #smooth the curve of the score
    score_frames = savgol_filter(score_frames,301,5)

    plt.figure()
    plt.title('Anomaly detection on an Abnormal Video')
    plt.plot(np.arange(0,nb_frames), score_frames, label='Anomaly Prediction')
    plt.axvline(x=anomaly_start_frame, color='r', label='Ground Truth: Start-End')
    plt.axvline(x=anomaly_end_frame, color='r')
    plt.ylim(0,1)
    plt.legend(loc = 'upper left')
    plt.xlabel('video frames')
    plt.ylabel('Anomaly score')


#Display the anomaly scores predicted for a Normal video
def qualitative_test_normal(video_name, video_path, features_path, anomaly_detector):
    
    #features
    features_video = np.loadtxt(features_path)
    #(32,512) features
    features_video = np.asarray(features_video, dtype= np.float32)
    tmp = np.linalg.norm(features_video, axis=1)
    tmp = np.repeat(tmp,512).reshape(32,512)
    
    #normalize features
    feat_norm = features_video/tmp

    #scores
    score = anomaly_detector.predict_on_batch(feat_norm)
    #score (32,) -> one score per segment video
    score = score.reshape(32)

    #compute nmber of frames in the video
    cap = cv2.VideoCapture(video_path)
    assert(cap.isOpened()== True) # Check if camera opened successfully
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #one anomaly score per frame
    score_frames = extrapolate(score, nb_frames)

    #smooth the curve of the score
    score_frames = savgol_filter(score_frames,301,5)

    plt.figure()
    plt.title('Anomaly detection on a Normal Video ')
    plt.plot(np.arange(0,nb_frames), score_frames, label='Anomaly Prediction')
    plt.ylim(0,1)
    plt.legend(loc = 'upper left')
    plt.xlabel('video frames')
    plt.ylabel('Anomaly score')


def compute_false_alarm_normal(gts_normal, preds_normal, threshold):
    
    #False Positive: model predicts normal activity as an abnormal activity
    nb_FP = len(np.where(preds_normal > threshold))
    
    False_alarm_rate = nb_FP/len(gts_normal)
    
    return False_alarm_rate


def compute_false_alarm_abnormal(gts_abnormal, preds_abnormal, threshold):
    
    #False Negatif: model predicts abnormal activity as an normal activity
    nb_FN = len(np.where(preds_abnormal < threshold))
    
    False_alarm_rate = nb_FN/len(np.where(gts_abnormal == 1))
    
    return False_alarm_rate
    