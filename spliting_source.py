# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 09:08:46 2021

@author: antoi
"""

f = open('source.txt')
lines = f.readlines()
f.close()

abnormal_lines, normal_lines = [],[]
for line in lines:
    line_split = line.strip().split(' ')
    if(line_split[2]=='Normal'):
        normal_lines.append(line)
    else:
        abnormal_lines.append(line)

f = open('Normal_Train.txt','w')
f.writelines(normal_lines)
f.close()
f = open('Abnormal_Train.txt','w')
f.writelines(abnormal_lines)
f.close()