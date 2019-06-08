---
layout: default
---
# Week 14: Classification Network, Regression Network, Reading information


This week, I've retrained the models with the new dataset. This new dataset is divided into 11275 pairs of images and speed data for training, and 4833 pairs of images and data for testing. 


## Driving videos

I've used the predictions of the classification network according to w (7 classes) and v (4 classes) to driving a formula 1: 

[![Follow line with classification network (7 classes for w and 4 classes for v, dataset2)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/follow_blue.png)](https://www.youtube.com/watch?v=mTQYu0gdxNY)


I've used the predictions of the regression network (223 epochs for v and 212 for w) to driving a formula 1:

[![Follow line with regression network (dataset2 simpleCircuit.world)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/follow_blue.png)](https://www.youtube.com/watch?v=CxkbHCt1gaI)


[![Follow line with regression network (dataset2 nurburgrinLineROS.world)](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/follow_blue.png)](https://www.youtube.com/watch?v=XBYubp2SChA)



## Data statistics

To analyze the data, a new statistic was created (analysis_vectors.py). I've analyzed two lines of each image and calculated the centroids of the corresponding lines (row 250 and row 360). On the x-axis of the graph, the centroid of row 350 is represented and the y-axis represents the centroid of row 260 of the image. In the following image we can see the representation of this statistic of the training set (new dataset) (red circles) iagainst the driving data (blue crosses). 

![L1_L2_dataset2](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/L1_L2_dataset2.png)



## Reading information

I've read some information about self-driving. I've read about different architectures: 

* [Pilotnet:](https://arxiv.org/pdf/1704.07911.pdf), [2](https://medium.com/pharos-production/behavioral-cloning-project-3-6b7163d2e8f9).

* [TinyPilotNet](https://ebuah.uah.es/dspace/handle/10017/33946): developed as a reduction from NVIDIA PilotNet CNN.

* [DeeperLSTM-TinyPilotNet](https://ebuah.uah.es/dspace/handle/10017/33946).

* [DeepestLSTM-TinyPilotNet](https://ebuah.uah.es/dspace/handle/10017/33946).

* [C-LSTM](https://arxiv.org/pdf/1710.03804.pdf).

* [ControlNet](https://pdfs.semanticscholar.org/ec17/ec40bb48ec396c626506b6fe5386a614d1c7.pdf).


## Classification network for w

I've retrained the classification network for w with the new dataset. The test results are: 

![test_7_w_dataset2](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/test_7_w_dataset2.png)

![matrix_conf_7_w_dataset2](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/matrix_conf_7_w_dataset2.png)



## Classification network for v

I've retrained the classification network for v with the new dataset. The test results are: 

![test_4_v_dataset2](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/test_4_v_dataset2.png)

![matrix_conf_4_v_dataset2](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/matrix_conf_4_v_dataset2.png)

