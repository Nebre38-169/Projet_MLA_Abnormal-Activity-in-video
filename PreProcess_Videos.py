# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:21:22 2021

@author: sofiane
"""

import os
import cv2
import numpy as np
import torch
import ResNet3D
import time

print('current path', os.getcwd())


#Load the PreTrained model ResNet3D-34
resnet3d_model = ResNet3D.load_resnet3d(map_location = 'cpu', PATH = '/Users/sofia/Desktop/Master_ISI/Machine learning&IA/Projet_MLA/PreTrained_ResNet3D/ResNet_PyTorch/resnet-34-kinetics.pth')



#the video rate is already 30fps
def get_videoframes(video_path):
    """resize video frame to 112 x 112 pixels"""
    
    #outputs: list of frames RGB of a video
    
    size = (int(112) , int(112))
    
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_path)

    # Check if camera opened successfully
    assert(cap.isOpened()== True)
    
    img_array = [] #frames BGR
    
    # Read until video is completed
    while(cap.isOpened()):
    
        # Capture frame-by-frame
        ret, frame = cap.read()
 
        if ret == True:
            
            #resize frame
            resize_img = cv2.resize(frame, size)
            
            #BGR to RGB image
            resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
            img_array.append(resize_img)
            
            # Display the resulting frame
            #cv2.imshow('Resize',resize_img)
            # Press Q on keyboard to  exit
            #if cv2.waitKey(25) & 0xFF == ord('q'):
                #break
                
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    #cv2.destroyAllWindows()
    
    return img_array


#extract spatio-temporal features with ResNet3D of one Video
def extract_features_resnet3d(video_path, out_dir = '/Features/Anormal/'):
    """extract 32 x 512D features from a video,
    one segment video -> one 512D features vector
    
    video_path: path of a video .mp4
    out_dir: the directory where features will be stored
    
    output: .txt file that contains the 32x512D features of the video
    """
    
    
    #numbers of segments for each video
    nb_segments = int(32)
    
    #all the frames of the video
    frames = get_videoframes(video_path)
    
    #numbers of frames for each segment
    nb_frames_seg = len(frames)//nb_segments
    
    #to store the 32 x 512D features of the video
    features_video = []
    
    
    #loop each segment of the video
    for i in range (0, len(frames) - nb_frames_seg, nb_frames_seg):
        
        #one segment video
        segment = frames[i:i+nb_frames_seg]
        
        #to store features 512D of one clip 16-frames
        list_features_clip = []
        
        #print('nb_frames_seg = ', nb_frames_seg )
        
        #split the segment into 16-frames clip: inputs of the resnet3d
        for j in range(0, nb_frames_seg - 16, 16):
            
            #inputs of the resnet3d
            input_clip = segment[j:j+16]
            
            #convert list of images to numpy of images
            input_clip = np.stack(input_clip) #array of shape (16,112,112,3)
            input_clip = np.asarray(input_clip, dtype=np.float32)
            
            #input shape must be (batch_size, nb_channels, depth, height, width) -> conv3d
            input_clip = input_clip.reshape((1,3,16,112,112))
            
            #convert numpy of images to torch tensor of images
            input_clip = torch.from_numpy(input_clip)
            
            #use ResNet3D-34 to extract spatio-temporal features
            features_clip = resnet3d_model(input_clip) #tensor (1,512)
            
            list_features_clip.append(features_clip.detach().numpy()[0])
        
        #features of the segment -> average of clips features
        feature_segment = np.stack(list_features_clip)
        feature_segment = np.mean(feature_segment,axis=0)
        
        #store the 512D features of the video
        features_video.append(feature_segment)
        
        
    #save features_video to binary file in /Features folder with the same name as video_path
    features_video = np.stack(features_video)
    
    #save the file with the same name as the video
    filename = video_path.split("/")[-1]
    filename = filename.split(".")[0]
    save_path = out_dir + filename
    np.savetxt(save_path + '.txt', features_video, fmt='%1.3f')


#to extract features of all the abnormal videos in the dataset UCF
def extract_all_features_abnormal(train_abnormal_split, abnormal_videos_dir, out_dir):
    """
    train_abnormal_split: path of the file that contains all abnormals videos names (split train abnormal)
    abnormal_videos_dir: path of the directory where all abnormals videos are stored
    out_dir: directory where the fearures will be saved
    """
    
    #store in a list all videos names of abnormal class
    f = open(train_abnormal_split, "r")
    list_train_abnormal = f.readlines()
    
    for i in range (len(list_train_abnormal)):
    
        print("extract features of the ",i ,"th video abnormal")
    
        #get one video abnormal in the dataset
        video_path = abnormal_videos_dir + '/' + list_train_abnormal[i].split("\n")[0]
        
        start_time = time.time()
        
        #extract the features and save it into file
        extract_features_resnet3d(video_path = video_path, out_dir = out_dir)
        
        print("--- %s seconds ---" % (time.time() - start_time))


 #to extract features of all the normal videos in the dataset UCF
def extract_all_features_normal(train_normal_split, normal_videos_dir, out_dir):
    
    #store in a list all videos names of normal class
    f = open(train_normal_split, "r")
    list_train_normal = f.readlines()
    
    for i in range (len(list_train_normal)):
    
        print("extract features of the ",i ,"th video normal")
    
        #get one video abnormal in the dataset
        video_path = normal_videos_dir + '/' + list_train_normal[i].split("\n")[0]
    
        #extract the features and save it into file
        extract_features_resnet3d(video_path = video_path, out_dir = out_dir)

