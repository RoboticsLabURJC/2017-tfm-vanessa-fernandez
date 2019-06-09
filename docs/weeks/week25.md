---
layout: default
---
# Week 25: Study of difference images, Results, Controlnet

## Results table (regression, cropped image)

|                                                      Driving results (regression networks)                                                                                              ||||||||||||||
|                           |        Manual        ||    Pilotnet v + w    ||  TinyPilotnet v + w  ||        Stacked v+w   ||  Stacked (diff) v+w  || LSTM-Tinypilotnet v + w ||DeepestLSTM-Tinypilot.||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |  Percentage   |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |    100%    | 1min 35s |     100%   | 1min 37s |     100%   | 1min 41s |     100%   | 1min 41s |     100%   | 1min 39s |      100%     | 1min 40s |    100%    | 1min 37s | 
|Simple (anti-clockwise)    |    100%    | 1min 32s |     100%   | 1min 38s |     100%   | 1min 41s |     10%    |          |     100%   | 1min 38s |      100%     | 1min 38s |    100%    | 1min 38s |
|  Monaco (clockwise)       |    100%    | 1min 15s |     100%   | 1min 20s |     100%   | 1min 19s |     85%    |          |     45%    |          |       50%     |          |     55%    |          |
|Monaco (anti-clockwise)    |    100%    | 1min 15s |     100%   | 1min 19s |     100%   | 1min 18s |     15%    |          |     5%     |          |       35%     |          |     55%    |          |
| Nurburgrin (clockwise)    |    100%    | 1min 02s |     100%   | 1min 04s |     100%   | 1min 04s |      8%    |          |     8%     |          |       40%     |          |    100%    | 1min 04s |
|Nurburgrin (anti-clockwise)|    100%    | 1min 02s |     100%   | 1min 06s |     100%   | 1min 05s |     80%    |          |     50%    |          |       50%     |          |     80%    |          |



## Results table (regression, whole image)

|                                                      Driving results (regression networks)                                               ||||||||||
|                           |        Manual        ||    Pilotnet v + w    ||  TinyPilotnet v + w  ||        Stacked v+w   ||  Stacked (diff) v+w  ||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |    100%    | 1min 35s |     100%   | 1min 41s |     100%   | 1min 39s |     100%   | 1min 40s |     100%   | 1min 43s |
|Simple (anti-clockwise)    |    100%    | 1min 32s |     100%   | 1min 39s |     100%   | 1min 38s |     100%   | 1min 46s |     10%    |          |
|  Monaco (clockwise)       |    100%    | 1min 15s |     100%   | 1min 21s |     100%   | 1min 19s |     50%    |          |     5%     |          |
|Monaco (anti-clockwise)    |    100%    | 1min 15s |     100%   | 1min 23s |     100%   | 1min 20s |      7%    |          |     5%     |          |
| Nurburgrin (clockwise)    |    100%    | 1min 02s |     100%   | 1min 03s |     100%   | 1min 05s |     50%    |          |     8%     |          |
|Nurburgrin (anti-clockwise)|    100%    | 1min 02s |     100%   | 1min 06s |     100%   | 1min 06s |     80%    |          |     50%    |          |


|                     Driving results (regression networks, continuation)                         ||||||
|                           | LSTM-Tinypilotnet v + w ||DeepestLSTM-Tinypilot.||      Controlnet      || 
|      Circuits             |  Percentage   |   Time   | Percentage |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |      100%     | 1min 39s |    100%    | 1min 39s |    100%    | 1min 46s |
|Simple (anti-clockwise)    |       10%     |          |    100%    | 1min 41s |    100%    | 1min 37s |
|  Monaco (clockwise)       |      100%     | 1min 27s |     50%    |          |      5%    |          | 
|Monaco (anti-clockwise)    |       50%     |          |    100%    | 1min 21s |      5%    |          |
| Nurburgrin (clockwise)    |      100%     | 1min 08s |    100%    | 1min 05s |      8%    |          |
|Nurburgrin (anti-clockwise)|       50%     |          |    100%    | 1min 07s |      8%    |          |




## Study of temporal images

I've tried to create a difference image with only two channels: HV. First, I made the absolute difference of the two images (separated 5 frames) for each channel. Then I normalized the difference between 0 and 255. It isn't a good solution for driving.

Straight line: 

![straight_hs](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/straight_hs.png)


Curve: 

![curve_hs](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/curve_hs.png)


I've create a sum image using numpy.add(x1, x2). The image result is:

![add_imgs](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/add_imgs.png)



## Controlnet

![model_controlnet](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/model_controlnet.png)


|            Driving results (Controlnet network, whole image)             |||||
|                           |          Manual         ||      Controlnet      || 
|      Circuits             |  Percentage   |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |      100%     | 1min 35s |    100%    | 1min 46s |
|Simple (anti-clockwise)    |      100%     | 1min 33s |    100%    | 1min 37s |
|  Monaco (clockwise)       |      100%     | 1min 15s |      5%    |          | 
|Monaco (anti-clockwise)    |      100%     | 1min 15s |      5%    |          |
| Nurburgrin (clockwise)    |      100%     | 1min 02s |      8%    |          |
|Nurburgrin (anti-clockwise)|      100%     | 1min 02s |      8%    |          |


