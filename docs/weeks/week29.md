---
layout: default
---
# Week 29: Problems with the circuits, Tests with CurveGP circuit, Controlnet statistics, Circuit

## Problems with circuits

* Simple circuit:

![simple](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/simple.png)


* Monaco circuit:

![monaco](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/monaco.png)


* Nurburgrin circuit:

![nurburgrin](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/nurburgrin.png)


* Small circuit:

![small](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/small.png)


* CurveGP circuit:

![curveGP](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/curve.png)



## Tests with CurveGP circuit

I've done tests with a CurveGP circuit: 


### Results table (cropped image)

|                                                      Driving results (regression networks)                                                                                              ||||||||||||||
|                           |        Manual        ||    Pilotnet v + w    ||  TinyPilotnet v + w  ||        Stacked v+w   ||  Stacked (diff) v+w  || LSTM-Tinypilotnet v + w ||DeepestLSTM-Tinypilot.||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |  Percentage   |   Time   | Percentage |   Time   |
| CurveGP (clockwise)       |    100%    | 2min 13s |      50%   |          |      25%   |          |      25%   |          |       2%   |          |       50%     |          |     25%    |          | 
| CurveGP  (anti-clockwise) |    100%    | 2min 09s |       2%   |          |       2%   |          |       2%   |          |       1%   |          |        2%     |          |      2%    |          |


|                                                      Driving results (classification networks)                     ||||||||
|                           |        Manual        ||      5v+7w biased    ||    5v+7w balanced    ||   5v+7w imbalanced   || 
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|   CurveGP (clockwise)     |    100%    | 2min 13s |    100%    | 2min 11s |    100%    | 2min 04s |    100%    | 2min 11s |
| CurveGP (anti-clockwise)  |    100%    | 2min 09s |    100%    | 2min 07s |    100%    | 2min 03s |    100%    | 2min 09s |



### Results table (whole image)

|                                                      Driving results (regression networks)                                               ||||||||||
|                           |        Manual        ||    Pilotnet v + w    ||  TinyPilotnet v + w  ||        Stacked v+w   ||  Stacked (diff) v+w  ||
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|   CurveGP (clockwise)     |    100%    | 2min 13s |       2%   |          |      1%    |          |      1%    |          |      2%    |          |
| CurveGP (anti-clockwise)  |    100%    | 2min 09s |       1%   |          |      1%    |          |      1%    |          |      1%    |          |


|                     Driving results (regression networks, continuation)                         ||||||
|                           | LSTM-Tinypilotnet v + w ||DeepestLSTM-Tinypilot.||      Controlnet      || 
|      Circuits             |  Percentage   |   Time   | Percentage |   Time   | Percentage |   Time   |
|   CurveGP (clockwise)     |        1%     |          |     15%    |          |      1%    |          |
| CurveGP (anti-clockwise)  |        2%     |          |      2%    |          |      1%    |          |



|                                                      Driving results (classification networks)                     ||||||||
|                           |        Manual        ||      5v+7w biased    ||    5v+7w balanced    ||   5v+7w imbalanced   || 
|      Circuits             | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   | Percentage |   Time   |
|   CurveGP (clockwise)     |    100%    | 2min 13s |     2%     |          |      2%    |          |      1%    |          |
| CurveGP (anti-clockwise)  |    100%    | 2min 09s |     1%     |          |      1%    |          |      1%    |          |



## Controlnet statistics

* Train:

![difference_controlnet_train_curves](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/difference_controlnet_train_curves.png)


<pre>
v results:

MSE: 1.308563

MAE: 0.433917
</pre>

<pre>
w results:

MSE: 0.009177

MAE: 0.053194
</pre>


* Test:

![difference_controlnet_test_curves](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/difference_controlnet_test_curves.png)


<pre>
v results:

MSE: 4.017514

MAE: 0.816513
</pre>

<pre>
w results:

MSE: 0.055743

MAE: 0.173289
</pre>


## Circuit

![circuit_curves](https://roboticsurjc-students.github.io/2017-tfm-vanessa-fernandez/images/circuit_curves.png)


