---
layout: default
---
# Week 34: Reading information, Controlnet

## Reading information

This week, I used [1](http://drivendata.co/blog/pri-matrix-factorization-benchmark/) for a better understanding of LRCN networks. 


## Controlnet

I've have problems with v's predictions: 

|            Driving results (Controlnet, whole image)                  |||||
|                           |        Manual        ||      ControlNet      || 
|      Circuits             | Percentage |   Time   | Percentage |   Time   | 
|  Simple (clockwise)       |    100%    | 1min 35s |      10%   |          |
|Simple (anti-clockwise)    |    100%    | 1min 32s |      12%   |          |
|  Monaco (clockwise)       |    100%    | 1min 15s |       5%   |          |
|Monaco (anti-clockwise)    |    100%    | 1min 15s |       5%   |          |
| Nurburgrin (clockwise)    |    100%    | 1min 02s |       8%   |          |
|Nurburgrin (anti-clockwise)|    100%    | 1min 02s |       3%   |          |
|   CurveGP (clockwise)     |    100%    | 2min 13s |      12%   |          |
| CurveGP (anti-clockwise)  |    100%    | 2min 09s |       3%   |          |
|   Small (clockwise)       |    100%    | 1min 00s |       5%   |          |
| Small (anti-clockwise)    |    100%    |    59s   |      10%   |          |

