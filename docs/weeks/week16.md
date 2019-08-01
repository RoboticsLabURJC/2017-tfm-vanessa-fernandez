---
layout: default
---
# Week 16: Driving videos, Classification network, LSTM-Tinypilotnet


## Driving videos

I've used the predictions of the classification network according to w (7 classes) and constant v to driving a formula 1: 

[![Follow line with classification network (7 classes for w, Test4)](https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/images/follow_blue.png)](https://www.youtube.com/watch?v=o1QfHrzEHm4)



## Problems with Ubuntu and Gazebo

This week, I had problems with Ubutu, I had to reinstall it several times, and when installing Gazebo the worlds look different. For this reason, the red line hasn't stripes and looks darker. Now, the trained models don't work correctly for these reasons. I have tried to put several lights in the world and it seems that the line looks more clear, but not being so similar to the training data causes the car crashes. Probably, the BGR color space is not the best to train. It may be that HSV is better, because it is more invariant in light changes. 


* BGR:

![bgr](https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/images/bgr.png)


* HSV:

![hsv](https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/images/hsv.png)



## Driving analysis

I've relabelled the images from [1](https://github.com/RoboticsURJC-students/2017-tfm-vanessa-fernandez/tree/master/Follow%20Line/Failed_driving/4v_7w) (classification network 4v+7w) at my criteria. And I've compared the relabelled data with the driving data. A 82% accuracy is obtained for w and 77% for v.

I've relabelled the images from [2](https://github.com/RoboticsURJC-students/2017-tfm-vanessa-fernandez/tree/master/Follow%20Line/Failed_driving/7w) (classification network 7w) at my criteria. And I've compared the relabelled data with the driving data. A 64% accuracy is obtained for w. 


## LSTM-Tinypilotnet

I've trained a new model: LSTM-Tinypilotnet: 

![model_lstm_tinypilotnet](https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/images/model_lstm_tinypilotnet.png)



## Classification network

I've imbalanced the classification network for w. The result can be watched in diving videos. 

