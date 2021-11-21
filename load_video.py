# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:12:22 2021

@author: antoi
"""

import numpy as np
import moviepy as mv


def read_source():
    f = open("source.txt")
    lines = f.readlines()
    normale_video = []
    abnormale_video = []
    for line in lines:
        content = line.strip().split(' ')
        video_dict = {'file':content[0],'type':content[2],'start':int(content[4]),'end':int(content[6]),'start2':int(content[8]),'end2':int(content[10])}
        if(video_dict['type']=='Normal'):
            normale_video.append(video_dict)
        else:
            abnormale_video.append(video_dict)
    return normale_video,abnormale_video


def read_video(batch_size):
    nb_abnormal_video = int(batch_size/2)
    nb_normal_video = nb_abnormal_video
    
    normal_video,abnormal_video = read_source()
    tot_normal_video = len(normal_video)
    tot_abnormal_video = len(abnormal_video)
    
    indexs_normal_video = np.random.permutation(nb_normal_video)
    indexs_abnormal_video = np.random.permutation(nb_abnormal_video)
    
    normal_video_full = []
    for index in indexs_normal_video:
        full_video = mv.editor.VideoFileClip('Video/Normal/'+normal_video[index]['file'])
        full_video_duration = full_video.duration
        segment_duration = full_video_duration/32
        segmented_video = []
        for i in range(31):
            segmented_video.append(full_video.subclip(segment_duration*i,segment_duration*(i+1)))
        normal_video_full.append({ 'info' : normal_video[index],'segment':segmented_video})
    
    abnormal_video_full = []
    for index in indexs_abnormal_video:
        full_video = mv.editor.VideoFileClip('Video/{0}/'.format(abnormal_video[index]['type'])+abnormal_video[index]['file'])
        full_video_duration = full_video.duration
        segment_duration = full_video_duration/32
        segmented_video = []
        for i in range(31):
            segmented_video.append(full_video.subclip(segment_duration*i,segment_duration*(i+1)))
        abnormal_video_full.append({ 'info' : abnormal_video[index],'segment':segmented_video})
    return normal_video_full,abnormal_video_full

normal,abnormal = read_video(4)


    