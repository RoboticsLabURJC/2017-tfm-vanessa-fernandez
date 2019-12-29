# dl-driver (TensorFlow + Keras)

### Contents

1. [Introduction](#introduction)
2. [Getting started](#getting-started)
3. [Requirements](#requirements)
4. [How to use](#how-to-use)
5. [Framework choice](#framework-choice)
6. [Dataset](#dataset)
7. [Models](#models)
8. [Classification Network](#classification-network)
9. [Regression Network](#regression-network)
10. [Information](#info)

## Introduction

Dl-driver is a visual control application with the necessary infrastructure to load and use self-driving neural networks. The system solves several functionalities: (1) it offers a graphical user interface that helps you debug the code; (b) it offers access to sensors and actuators with simple methods (it hides communications middleware); (c) it includes auxiliary code to send the estimated orders by the networks (either classification or regression) to the engines. Through this system, the user must only include his network and retouch a file where the predicted speed orders by the network are provided to the vehicle.


## Getting started

First of all, we need to install the JdeRobot packages, Python (2.7 for the moment, due to ROS compatibility), and a few Python packages, installable via `python-pip`. We need two JdeRobot packages: JdeRobot-base and JdeRobot-assets. You can follow this [tutorial](https://github.com/JdeRobot/base#getting-environment-ready) for the complete installation. See the [Requirements](#requirements) for more details.

* Clone the repository:

`git clone https://github.com/RoboticsLabURJC/2017-tfm-vanessa-fernandez.git`

`cd 2017-tfm-vanessa-fernandez/Follow\ Line/dl-driver/`


* Create a virtual environment:

`virtualenv -p python2.7 virtualenv --system-site-packages`


## Requirements

## How to Use

Launch Gazebo with the f1 world through the command

```
roslaunch /opt/jderobot/share/jderobot/gazebo/launch/f1.launch
```

Then you have to execute the application, which will incorporate your code:

```
python2 driver.py driver.yml
```

## Framework choice

## Datasets

There are currently **two sets** of data to train the neural network that resolves the circuit. One contains **images of all types** such as straights and curves and the other contains **only the curves** of the circuit. The second one is smaller and the results are good enough to solve a lap of the circuit.

- [Complete dataset](http://wiki.jderobot.org/store/jmplaza/uploads/deeplearning-datasets/vision-based-end2end-learning/complete_dataset.zip).
- [Curve dataset](http://wiki.jderobot.org/store/jmplaza/uploads/deeplearning-datasets/vision-based-end2end-learning/curves_only.zip).

## Models

The models used in this repository are the following:

| Model                     | Links                                                        | Image                                                       |
| ------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| PilotNet                  | [Paper](https://arxiv.org/pdf/1704.07911.pdf). [Nvidia source.](https://devblogs.nvidia.com/explaining-deep-learning-self-driving-car/) | [Structure](/Net/Keras/models/model_pilotnet.png)                 |
| TinyPilotNet              | [Javier del Egido Sierra](https://ebuah.uah.es/dspace/bitstream/handle/10017/33946/TFG_Egido_Sierra_2018.pdf?sequence=1&isAllowed=y) TFG's. | -                                                           |
| LSTM                      | [Info](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | -                                                           |
| LSTM TinyPilotNet         | -                                                            | [Structure](/Net/Keras/models/model_lstm_tinypilotnet.png)        |
| Deepest LSTM TinyPilotNet | [Javier del Egido Sierra](https://ebuah.uah.es/dspace/bitstream/handle/10017/33946/TFG_Egido_Sierra_2018.pdf?sequence=1&isAllowed=y) TFG's. | [Structure](/Net/Keras/models/model_deepestlstm_tinypilotnet.png) |
| ControlNet                | -                                                            | [Structure](/Net/Keras/models/model_controlnet.png)               |
| Stacked                   | -                                                            | [Structure](/Net/Keras/models/model_stacked.png)                  |
| Stacked Dif or Temporal   | -                                                            | -                                                           |

The models are available in the [following repository](http://wiki.jderobot.org/store/jmplaza/uploads/deeplearning-models/models.zip).


## Classification Network

### Train Classification Network

To train the classification network to run the file:

```
cd 2017-tfm-vanessa-fernandez/Follow Line/dl-driver/Net/Keras/
python classification_train.py
```

When the program is running it will ask for data to know the characteristics of the network to be trained:

1. **Choose one of the options for the number of classes:** Choose the number of classes you want, typically 4-5 for v parameter and 2, 7 or 9 for w parameter. Depending on the dataset created, there are different classifications in the json depending on the number of classes for each screen.

2. **Choose the variable you want to train:** v or w: here you put v or w depending on the type of speed you want to train (traction or turn).

3. **Choose the type of image you want:** normal or cropped: you can choose normal or cropped. If you want to train with the full image you have to choose normal and if you want to train with the
cropped image choose cropped.

4. **Choose the type of image you want:** normal or cropped : you can choose normal or cropped. If you want to train with the full image you have to choose normal and if you want to train with the
cropped image choose cropped.

The documentation talks about having on the one hand a training with the whole dataset without any type of treatment of the number of images for each class (there were many more straight lines than
curves) or using a balanced datased that we created keeping the same number of images for each class (v and w).

To train with that configuration, set the normal option. To train with balanced dataset, set the option: balanced. And finally, biased refers to you training with the full dataset (unbalanced) but in training you put weights to each class with class_weigth type.

```
class_weight = {0: 4., 1: 2., 2: 2., 3: 1., 4:2., 5: 2., 6: 3.}
```

then with this option you can give more weight to the kinds of curves than straight lines. In that example the class 0 is the class radically_left and the 6 would be radically_right. The option
that worked best was that of 'biased'.

5. **Choose the model you want to use:** lenet, smaller_vgg or other: Here you have to choose the model you want to train. The option that offers the best results is smaller_vgg . The lenet model
gave very bad results because it was very basic. The other model loaded another model that gives worse results. The files containing the network models as such are in the folder models/ . For
classification you have them in classification_model.py for regression in model_nvidia.py.



## Regression Network


## Information

- More detailed info at my [Github-pages](https://roboticslaburjc.github.io/2017-tfm-vanessa-fernandez/).



