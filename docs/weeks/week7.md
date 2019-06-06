---
layout: default
---
# Week 7: Dataset generator and driver node

## Dataset generator

The final goal of the project is to make a follow-line using Deep Learning. For this it is necessary to collect data. For this reason, I have based on the [code](http://vanessavisionrobotica.blogspot.com/2018/05/practica-1-follow-line-prueba-2.html) created for the follow-line practice of JdeRobot Academy in order to create a dataset. The created [dataset](https://github.com/RoboticsURJC-students/2017-tfm-vanessa-fernandez/tree/master/Follow%20Line/Dataset) contains the input images with the corresponding linear and angular speeds (in a json file). To create this dataset I have made a Python [file](https://github.com/RoboticsURJC-students/2017-tfm-vanessa-fernandez/blob/master/Follow%20Line/generator.py) that contains functions that allow to create it. 


## Driver node

In addition, I've created a driver node based on the objectdetector node, which allows to connect neural networks. For now the initial gui looks like this: 

![GUI](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/gui_inicial.png)

