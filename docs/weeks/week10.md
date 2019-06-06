---
layout: default
---
# Week 10: Adding new class, Classification network

## Adding new class

The files data.json, train.json and test.json have been modified to add a new classification that divides the angles of rotation into 7 classes. The classes are the following: 

<pre>
radically_right: if the rotation's angle is w <= -1.
moderately_right: if the rotation's angle is -1 < w <= -0.5.
slightly_right: if the rotation's angle is -0.5 < w <= -0.1.
slight: if the rotation's angle is -0.1 < w < 0.1.
slightly_left: if the rotation's angle is 0.1 <= w < 0.5.
moderately_left: if the rotation's angle is 0.5 <= w < 1.
radically_left: if the rotation's angle is w >= 1.
</pre>

## Multiclass classification network

I've followed this [blog](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras)/ as example to make the classification network. In this case I have trained a model with the 7 classes mentioned above.

The CNN architecture I am using is SmallerVGGNet, a simplified version of VGGNet. The VGGNet model was first introduced by Simonyan and Zisserman in their 2014 paper, [Very Deep Convolutional Networks for Large Scale Image Recognition](https://arxiv.org/pdf/1409.1556/). In this case we keep an image of the model with plot_model to see the architecture of the network. The model (classification_model.py) is as follows: 

![model_smaller_vgg](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/model_smaller_vgg.png)

After training the network, we save the model (models/model_smaller_vgg_7classes_w.h5) and evaluate the model with the validation set: 

![smaller_vgg_7_train](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/smaller_vgg_7_train.png)

We also show the graphs of loss and accuracy for training and validation according to the epochs: 

![smaller_vgg_7_train_loss](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/smaller_vgg_7_train_loss.png)

![smaller_vgg_7_train_accuracy](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/smaller_vgg_7_train_accuracy.png)



In addition, we evaluate the accuracy, precision, recall, F1-score and we paint the confusion matrix. The results are the following: 

![smaller_vgg_7_test](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/smaller_vgg_7_test.png)

![smaller_vgg_7_confusion_matrix](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/smaller_vgg_7_confusion_matrix.png)

