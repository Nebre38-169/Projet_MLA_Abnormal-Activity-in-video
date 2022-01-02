# 3D ResNet with Ranking Loss Function for Abnormal Activity Detection in Videos
Simple overview of use

## Description

This project consisted in reproducing the results obtained by Shikha Dubey, Abhijeet Boragule,
Moongu Jeon <https://arxiv.org/abs/2002.01132>. It consists to use the model ResNet3D pre-trained in the Kinetics dataset as features extractor of the videos from UCF-Crime dataset and implement a neural network to predict anomaly behaviour in videos. The new approach that consists to use Multiple Instance Learning allows to predict temporal anomaly behaviour (in frame-level) without using frame-level labels in the training, only by using video-level labels (video is annotated abnormal or normal). Moreover with a new ranking loss, the false alarm rate in abnormal videos is reduced.

## Getting Started

### Dependencies

* Tensorflow (2.7.0)
* pytorch (1.6.0), only if you want to extract features with ResNet3D from (https://github.com/kenshohara/3D-ResNets-PyTorch) but you can download our features in folder /Features.
* numpy
* scikit-learn
* scipy

### Pre-Trained ResNet3D-34
As features extractor, we use the model ResNet3D-34 which was pre-trained on the dataset Kinetics in order to extract human action. The pre-trained model that we have used is stored here https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M. Download the file resnet-34-kinetics.pth.
In our work, we load this pre-trained model in the script ResNet3D.py and use it in the script PreProcess_Videos.py in order to extract features in the videos UCF-Crime.

### Preparation of the dataset

* After download the UCF-Crime dataset from https://webpages.charlotte.edu/cchen62/dataset.html, you have first to :

1) separate the normal videos .mp4 and the abnormal videos .mp4

2) separate the Anomaly_train.txt and Anomaly_test.txt files in Abnormal_train.txt, Normal_train.txt and Abnormal_test.txt, Normal_test.txt respectively. 

3) then you have to separate the file Temporal_Anomaly_Annotation_for_Testing_Videos.txt in temporal_annotation_normal_videos.txt and temporal_annotation_normal_videos.txt.





### Executing program

* To run our project, just execute the Notebook Demo.ipynb


## Authors


* Pichereau Victorien : victorien.pichereau@ensam.eu
* Bouzidi Sofiane : sofiane.bouzidi@hotmail.fr
* Doste Antoine : antoine.doste@ensam.eu
* Mbaira Laila : laila.mbaira@ensam.eu

## Acknowledgments


<https://arxiv.org/abs/2002.01132>

<https://arxiv.org/pdf/1801.04264.pdf>

<https://arxiv.org/abs/1705.06950>

<https://image-net.org/challenges/LSVRC/>

<https://arxiv.org/pdf/1708.07632v1.pdf>

<https://paperswithcode.com/paper/real-world-anomaly-detection-in-surveillance>


