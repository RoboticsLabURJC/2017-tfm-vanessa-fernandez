#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import glob
import cv2


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        array_imgs.append(img)

    return array_imgs


def parse_json(data):
    array_w = []
    array_v = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array_v.append(format(float(v), '.6f'))
        array_w.append(float(w))

    return array_w, array_v


def calculate_accuracy(labels, y_predict):
    top_k = 0
    for i in range(0, len(labels)):
        if labels[i] == y_predict[i]:
            top_k += 1
    top_k = top_k * 100 / len(labels)
    return top_k


if __name__ == "__main__":
    # Load data
    file = open('Failed_driving/data.json', 'r')
    data = file.read()
    file.close()
    file= open('Failed_driving/corrected_data.json', 'r')
    data_corrected = file.read()
    file.close()
    
    # We preprocess json
    array_w, array_v = parse_json(data)
    array_w_corrected, array_v_corrected = parse_json(data_corrected)

    accuracy = calculate_accuracy(array_w_corrected, array_w)
    print('Accuracy: ' + str(accuracy) + '%')

