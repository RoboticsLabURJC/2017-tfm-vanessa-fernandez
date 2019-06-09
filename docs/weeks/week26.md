---
layout: default
---
# Week 26: Tests with other circuit, Controlnet, Temporal difference network


## Tests with other circuit

I've done tests with a circuit that hasn't been used for training. 

![small_cirtuit](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/small_cirtuit.png)



## Results table (cropped image)

|                                                      Driving results (regression networks)                                                                                              ||||||||||||||
|                           |        Manual        ||    Pilotnet v + w    ||  TinyPilotnet v + w  ||        Stacked v+w   ||  Stacked (diff) v+w  || LSTM-Tinypilotnet v + w ||DeepestLSTM-Tinypilot.||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |  Percentage   |   Time   | Percentage |   Time   |
| Small (clockwise)         |    100%    | 1min 00s |      10%   |          |     100%   | 1min 14s |     100%   | 1min 08s |      10%   |          |       10%     |          |    100%    | 1min 09s | 
| Small (anti-clockwise)    |    100%    |    59s   |      20%   |          |     100%   | 1min 17s |     100%   | 1min 08s |      20%   |          |       80%     |          |    100%    | 1min 07s |


|                                                      Driving results (classification networks)                     ||||||||
|                           |        Manual        ||      5v+7w biased    ||    5v+7w balanced    ||   5v+7w imbalanced   || 
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|   Small (clockwise)       |    100%    | 1min 00s |    100%    | 1min 02s |    100%    | 1min 03s |    100%    | 1min 07s |
| Small (anti-clockwise)    |    100%    |    59s   |    100%    | 1min 05s |    100%    | 1min 02s |    100%    | 1min 08s |



## Results table (whole image)

|                                                      Driving results (regression networks)                                               ||||||||||
|                           |        Manual        ||    Pilotnet v + w    ||  TinyPilotnet v + w  ||        Stacked v+w   ||  Stacked (diff) v+w  ||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|   Small (clockwise)       |    100%    | 1min 00s |      85%   |          |     100%   | 1min 09s |      80%   |          |     100%   | 1min 03s |
| Small (anti-clockwise)    |    100%    |    59s   |     100%   | 1min 08s |     100%   | 1min 13s |      20%   |          |     100%   | 1min 04s |


|                     Driving results (regression networks, continuation)                         ||||||
|                           | LSTM-Tinypilotnet v + w ||DeepestLSTM-Tinypilot.||      Controlnet      || 
|      Circuits             |  Percentage   |   Time   | Percentage |   Time   | Percentage |   Time   |
|   Small (clockwise)       |       10%     |          |    100%    | 1min 01s |     20%    |          |
| Small (anti-clockwise)    |       20%     |          |     20%    |          |     20%    |          |



|                                                      Driving results (classification networks)                     ||||||||
|                           |        Manual        ||      5v+7w biased    ||    5v+7w balanced    ||   5v+7w imbalanced   || 
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|   Small (clockwise)       |    100%    | 1min 00s |    100%    | 1min 10s |     80%    |          |    100%    | 1min 07s |
| Small (anti-clockwise)    |    100%    |    59s   |    100%    | 1min 07s |     15%    |          |     75%    |          |




## Controlnet

|            Driving results (Controlnet network, whole image)             |||||
|                           |          Manual         ||      Controlnet      || 
|      Circuits             |  Percentage   |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |      100%     | 1min 35s |    100%    | 1min 46s |
|Simple (anti-clockwise)    |      100%     | 1min 33s |    100%    | 1min 38s |
|  Monaco (clockwise)       |      100%     | 1min 15s |      5%    |          | 
|Monaco (anti-clockwise)    |      100%     | 1min 15s |      5%    |          |
| Nurburgrin (clockwise)    |      100%     | 1min 02s |      8%    |          |
|Nurburgrin (anti-clockwise)|      100%     | 1min 02s |     75%    |          |



## Temporal difference network

I've tested a network that takes a gray scale difference image as the input image, but I've made a preprocess: 

<pre>
margin = 10
i1 = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
i2 = cv2.cvtColor(imgs[i - (margin + 1)], cv2.COLOR_BGR2GRAY)
i1 = cv2.GaussianBlur(i1, (5, 5), 0)
i2 = cv2.GaussianBlur(i2, (5, 5), 0)
difference = np.zeros((i1.shape[0], i1.shape[1], 1))
difference[:, :, 0] = cv2.absdiff(i1, i2)
_, difference[:, :, 0] = cv2.threshold(difference[:, :, 0], 15, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
difference[:, :, 0] = cv2.morphologyEx(difference[:, :, 0], cv2.MORPH_CLOSE, kernel)
</pre>


 I've used a margin of 10 images between the 2 images. The result is: 

![dif_gray](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/dif_gray.png)


[![Follow line with Temporal difference network](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/image_simple_circuit.png)](https://www.youtube.com/watch?v=E8Z50k3hRpw)


|        Driving results (Temporal difference network, whole image)        |||||
|                           |          Manual         ||      Controlnet      || 
|      Circuits             |  Percentage   |   Time   | Percentage |   Time   |
|  Simple (clockwise)       |      100%     | 1min 35s |     25%    |          |
|Simple (anti-clockwise)    |      100%     | 1min 33s |     10%    |          |
|  Monaco (clockwise)       |      100%     | 1min 15s |      5%    |          | 
|Monaco (anti-clockwise)    |      100%     | 1min 15s |      3%    |          |
| Nurburgrin (clockwise)    |      100%     | 1min 02s |      8%    |          |
|Nurburgrin (anti-clockwise)|      100%     | 1min 02s |      3%    |          |


