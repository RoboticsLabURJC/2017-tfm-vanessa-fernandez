#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import glob
import cv2


def parse_json(data):
    array_v = []
    array_w = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array_v.append(float(v))
        array_w.append(float(w))

    return array_v, array_w


def analysis_w(array):
    num_neg_2_9 = 0
    num_neg_2_6 = 0
    num_neg_2_3 = 0
    num_neg_2_0 = 0
    num_neg_1_7 = 0
    num_neg_1_4 = 0
    num_neg_1_1 = 0
    num_neg_0_8 = 0
    num_neg_0_5 = 0
    num_neg_0_2 = 0
    num_neg_0_0 = 0
    num_pos_0_0 = 0
    num_pos_0_2 = 0
    num_pos_0_5 = 0
    num_pos_0_8 = 0
    num_pos_1_1 = 0
    num_pos_1_4 = 0
    num_pos_1_7 = 0
    num_pos_2_0 = 0
    num_pos_2_3 = 0
    num_pos_2_6 = 0
    num_pos_2_9 = 0
    for w in array:
        if w < -2.9:
           num_neg_2_9 += 1
        elif -2.9 <= w and w < -2.6: 
           num_neg_2_6 += 1
        elif -2.6 <= w and w < -2.3: 
           num_neg_2_3 += 1
        elif -2.2 <= w and w < -2.0: 
           num_neg_2_0 += 1
        elif -2.0 <= w and w < -1.7: 
           num_neg_1_7 += 1
        elif -1.7 <= w and w < -1.4: 
           num_neg_1_4 += 1
        elif -1.4 <= w and w < -1.1: 
           num_neg_1_1 += 1
        elif -1.1 <= w and w < -0.8: 
           num_neg_0_8 += 1
        elif -0.8 <= w and w < -0.5: 
           num_neg_0_5 += 1
        elif -0.5 <= w and w < -0.2: 
           num_neg_0_2 += 1
        elif -0.2 <= w and w < -0.0: 
           num_neg_0_0 += 1
        elif 0.0 <= w and w < 0.2: 
           num_pos_0_0 += 1
        elif 0.2 <= w and w < 0.5: 
           num_pos_0_2 += 1
        elif 0.5 <= w and w < 0.8: 
           num_pos_0_5 += 1
        elif 0.8 <= w and w < 1.1: 
           num_pos_0_8 += 1
        elif 1.1 <= w and w < 1.4: 
           num_pos_1_1 += 1
        elif 1.4 <= w and w < 1.7: 
           num_pos_1_4 += 1
        elif 1.7 <= w and w < 2.0: 
           num_pos_1_7 += 1
        elif 2.0 <= w and w < 2.3: 
           num_pos_2_0 += 1
        elif 2.3 <= w and w < 2.6: 
           num_pos_2_3 += 1
        elif 2.6 <= w and w < 2.9: 
           num_pos_2_6 += 1
        elif 2.9 <= w: 
           num_pos_2_9 += 1
    print('Negative: < -2.9', num_neg_2_9)
    print('Negative: -2.9 <= w < -2.6', num_neg_2_6)
    print('Negative: -2.6 <= w < -2.3', num_neg_2_3)
    print('Negative: -2.3 <= w < -2.0', num_neg_2_0)
    print('Negative: -2.0 <= w < -1.7', num_neg_1_7)
    print('Negative: -1.7 <= w < -1.4', num_neg_1_4)
    print('Negative: -1.4 <= w < -1.1', num_neg_1_1)
    print('Negative: -1.1 <= w < -0.8', num_neg_0_8)
    print('Negative: -0.8 <= w < -0.5', num_neg_0_5)
    print('Negative: -0.5 <= w < -0.2', num_neg_0_2)
    print('Negative: -0.2 <= w < -0.0', num_neg_0_0)
    print('Positive: 0.0 <= w < 0.2', num_pos_0_0)
    print('Positive: 0.2 <= w < 0.5', num_pos_0_2)
    print('Positive: 0.5 <= w < 0.8', num_pos_0_5)
    print('Positive: 0.8 <= w < 1.1', num_pos_0_8)
    print('Positive: 1.1 <= w < 1.4', num_pos_1_1)
    print('Positive: 1.4 <= w < 1.7', num_pos_1_4)
    print('Positive: 1.7 <= w < 2.0', num_pos_1_7)
    print('Positive: 2.0 <= w < 2.3', num_pos_2_0)
    print('Positive: 2.3 <= w < 2.6', num_pos_2_3)
    print('Positive: 2.6 <= w < 2.9', num_pos_2_6)
    print('Positive: 2.9 <= w', num_pos_2_9)


def analysis_v(array):
    num_neg_v = 0
    num_min_v = 0
    num_5_v = 0
    num_6_v = 0
    num_7_v = 0
    num_8_v = 0
    num_9_v = 0
    num_10_v = 0
    num_11_v = 0
    num_12_v = 0
    num_13_v = 0
    for v in array:
        if v < 0:
            num_neg_v += 1
        elif v >= 0 and v < 5:
            num_min_v += 1
        elif v >= 5 and v < 6:
            num_5_v += 1
        elif v >= 6 and v < 7:
            num_6_v += 1
        elif v >= 7 and v < 8:
            num_7_v += 1
        elif v >= 8 and v < 9:
            num_8_v += 1
        elif v >= 9 and v < 10:
            num_9_v += 1
        elif v >= 10 and v < 11:
            num_10_v += 1
        elif v >= 11 and v < 12:
            num_11_v += 1
        elif v >= 12 and v < 13:
            num_12_v += 1
        elif v >= 13:
            num_13_v += 1

    print('Negative <= 0: ', num_neg_v)
    print('Positive 0 <= v < 5: ', num_min_v)
    print('Positive 5 <= v < 6: ', num_5_v)
    print('Positive 6 <= v < 7: ', num_6_v)
    print('Positive 7 <= v < 8: ', num_7_v)
    print('Positive 8 <= v < 9: ', num_8_v)
    print('Positive 9 <= v < 10: ', num_9_v)
    print('Positive 10 <= v < 11: ', num_10_v)
    print('Positive 11 <= v < 12: ', num_11_v)
    print('Positive 12 <= v < 13: ', num_12_v)
    print('Positive 13 <= v: ', num_13_v)


if __name__ == "__main__":
    # Load data
    file = open('dl-driver/Net/Dataset/data.json', 'r')
    data = file.read()
    file.close()

    array_v, array_w = parse_json(data)

    print(len(array_w))
    print('=====ANALYSYS OF W=====')
    analysis_w(array_w)
    print('=====ANALYSYS OF V=====')
    analysis_v(array_v)

