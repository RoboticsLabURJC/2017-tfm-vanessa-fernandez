---
layout: default
---
# Week 24: Reading information, Temporal difference network, Results

## Results(cropped image)

|                                                      Driving results (classification networks)                     ||||||||
|                           |        Manual         |      5v+7w biased     |     5v+7w balanced    |    5v+7w imbalanced   | 
|         :---:             |        :---:         ||        :---:         ||        :---:         ||        :---:         ||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |    100%    | 1min 35s |    100%    | 1min 41s |     75%    |          |    100%    | 1min 42s |
|Simple (anti-clockwise)    |    100%    | 1min 32s |    100%    | 1min 39s |    100%    | 1min 39s |    100%    | 1min 43s |
|  Monaco (clockwise)       |    100%    | 1min 15s |    100%    | 1min 20s |     70%    |          |     85%    |          |
|Monaco (anti-clockwise)    |    100%    | 1min 15s |    100%    | 1min 18s |      8%    |          |    100%    | 1min 20s |
| Nurburgrin (clockwise)    |    100%    | 1min 02s |    100%    | 1min 03s |    100%    | 1min 03s |    100%    | 1min 05s |
|Nurburgrin (anti-clockwise)|    100%    | 1min 02s |    100%    | 1min 05s |     80%    |          |     80%    |          |




## Results(whole image)

|                                                      Driving results (classification networks)                     ||||||||
|                           |        Manual         |      5v+7w biased     |     5v+7w balanced    |    5v+7w imbalanced   | 
|         :---:             |        :---:         ||        :---:         ||        :---:         ||        :---:         ||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |    100%    | 1min 35s |     35%    |          |     10%    |          |     90%    |          |
|Simple (anti-clockwise)    |    100%    | 1min 32s |    100%    | 1min 49s |    100%    | 1min 46s |     90%    |          |
|  Monaco (clockwise)       |    100%    | 1min 15s |    100%    | 1min 24s |      5%    |          |    100%    | 1min 23s |
|Monaco (anti-clockwise)    |    100%    | 1min 15s |    100%    | 1min 29s |      8%    |          |    100%    | 1min 24s |
| Nurburgrin (clockwise)    |    100%    | 1min 02s |    100%    | 1min 10s |      8%    |          |     90%    |          |
|Nurburgrin (anti-clockwise)|    100%    | 1min 02s |    100%    | 1min 07s |      8%    |          |    100%    | 1min 09s |



## Reading information

### End to End Learning for Self-Driving Cars

In this [paper](https://www.researchgate.net/publication/301648615_End_to_End_Learning_for_Self-Driving_Cars) (https://github.com/Kejie-Wang/End-to-End-Learning-for-Self-Driving-Cars), a convolutional neural network (CNN) is trained to map raw pixels from a single front-facing camera directly to steering commands. The system automatically learns internal representations of the necessary processing steps such as detecting useful road features with only the human steering angle as the training signal.

Images are fed into a CNN which then computes a proposed steering command. The proposed command is compared to the desired command for that image and the weights of the CNN are adjusted to bring the CNN output closer to the desired output. The weight adjustment is accomplished using back propagation. Once trained, the network can generate steering from the video images of a single center camera.

Training data was collected by driving on a wide variety of roads and in a diverse set of lighting and weather conditions. Most road data was collected in central New Jersey, although highway data was also collected from Illinois, Michigan, Pennsylvania, and New York. Other road types include two-lane roads (with and without lane markings), residential roads with parked cars, tunnels, and unpaved roads. Data was collected in clear, cloudy, foggy, snowy, and rainy weather, both day and night. 72 hours of driving data was collected.

They train the weights of their network to minimize the mean squared error between the steering command output by the network and the command of either the human driver, or the adjusted steering command for off-center and rotated images. The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The input image is split into YUV planes and passed to the network.

The ﬁrst layer of the network performs image normalization. The convolutional layers were designed to perform feature extraction and were chosen empirically through a series of experiments that varied layer conﬁgurations. Theye use strided convolutions in the ﬁrst three convolutional layers with a 2×2 stride and a 5×5 kernel and a non-strided convolution with a 3×3 kernel size in the last two convolutional layers. They follow the ﬁve convolutional layers with three fully connected layers leading to an output control value which is the inverse turning radius. The fully connected layers are designed to function as a controller for steering, but it is not possible to make a clean break between which parts of the network function primarily as feature extractor and which serve as controller.

To train a CNN to do lane following they only select data where the driver was staying in a lane and discard the rest. They then sample that video at 10 FPS. A higher sampling rate would result in including images that are highly similar and thus not provide much useful information. After selecting the ﬁnal set of frames they augment the data by adding artiﬁcial shifts and rotations to teach the network how to recover from a poor position or orientation.

Before road-testing a trained CNN, they ﬁrst evaluate the networks performance in simulation. 


### VisualBackProp: efficient visualization of CNNs

[Paper](https://arxiv.org/pdf/1611.05418.pdf) 


### Target-driven Visual Navigation in Indoor Scenesusing Deep Reinforcement Learning

[Paper](https://arxiv.org/pdf/1609.05143.pdf, https://www.youtube.com/watch?v=SmBxMDiOrvs) 



## Temporal difference network

In this method I test a new network with the difference image of it and it-5. The results are: 


|                                                      Driving results (regression networks)                                                          ||||||||
|                           | Temporal_dif const v whole image | Temporal_dif whole image   | Temporl_dif const v cropped image | Temporal_dif cropped image |
|         :---:             |              :---:              ||           :---:           ||               :---:               ||           :---:          ||
|      Circuits             |    Percentage     |     Time     |    Percentage   |   Time   |      Percentage      |     Time    |   Percentage   |   Time   |
|  Simple (clockwise)       |       100%        |   3min 37s   |       100%      | 1min 43s |          100%        |   3min 37s  |      100%      | 1min 39s |
|Simple (anti-clockwise)    |       100%        |   3min 38s   |       100%      | 1min 44s |          100%        |   3min 38s  |      100%      | 1min 42s |
|  Monaco (clockwise)       |        45%        |              |         5%      |          |          45%         |             |       5%       |          |
|Monaco (anti-clockwise)    |        45%        |              |         5%      |          |           8%         |             |       5%       |          |
| Nurburgrin (clockwise)    |         8%        |              |         8%      |          |           8%         |             |       8%       |          |
|Nurburgrin (anti-clockwise)|        90%        |              |         8%      |          |          90%         |             |       8%       |          |




Difference image: 

![dif_im](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/dif_im.png)


