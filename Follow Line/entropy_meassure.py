#!/usr/bin/python
#-*- coding: utf-8 -*-

import glob
import cv2
import numpy as np

from math import *

def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        array_imgs.append(img)
    return array_imgs


def filter_image(image):
    # RGB model change to HSV
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Minimum and maximum values ​​of the red
    value_min_HSV = np.array([0, 235, 60])
    value_max_HSV = np.array([180, 255, 255])

    # Filtering images
    image_HSV_filtered = cv2.inRange(image_HSV, value_min_HSV, value_max_HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    image_HSV_filtered = cv2.morphologyEx(image_HSV_filtered, cv2.MORPH_CLOSE, kernel)
    return image_HSV_filtered



def calculate_position_vectors(img):
    # We look for the position on the x axis of the pixels that have value 1 in different positions and
    position_x_down = np.where(img[350, :])
    position_x_middle = np.where(img[310, :])
    position_x_above = np.where(img[260, :])
    return position_x_down, position_x_middle, position_x_above


def calculate_position_points(positionx):
    if (len(positionx[0]) > 1):
        x_points = [positionx[0][0], positionx[0][len(positionx[0])-1]]
        not_found = False
    else:
        x_points = [None, None]
        not_found = True
    return x_points, not_found


def get_dataset_vectors(array_images):
    dataset = []
    for i in range(0, len(array_images)):
        img = filter_image(array_images[i])
    
        # We calculate vectors
        position_x_down, position_x_middle, position_x_above = calculate_position_vectors(img)

        # We see that white pixels have been located and we look if the vector is located
        x_middle_down, not_found_down = calculate_position_points(position_x_down)
        x_middle_middle, not_found_middle = calculate_position_points(position_x_middle)
        x_middle_above, not_found_above = calculate_position_points(position_x_above)

        dataset.append([x_middle_down[0], x_middle_down[1], x_middle_middle[0], x_middle_middle[1], x_middle_above[0], x_middle_above[1]])

    return dataset


def calculate_shannon_entropy(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0 
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
	return shannonEnt


if __name__ == "__main__":
    # Load data
    list_images_driving = glob.glob('Failed_driving/Images/' + '*')
    images_driving = sorted(list_images_driving, key=lambda x: int(x.split('/')[2].split('.png')[0]))
    list_images_dataset = glob.glob('Dataset/Train/Images/' + '*')
    images_dataset = sorted(list_images_dataset, key=lambda x: int(x.split('/')[3].split('.png')[0]))

    # Read images
    array_driving = get_images(list_images_driving)
    array_dataset = get_images(list_images_dataset)

    # Get dataset of vectors
    dataset_driving = get_dataset_vectors(array_driving)
    dataset_dataset = get_dataset_vectors(array_dataset)

    # Shannon entropy
    shannon_entropy_driving = calculate_shannon_entropy(dataset_driving)
    shannon_entropy_dataset = calculate_shannon_entropy(dataset_dataset)

    print('Shannon entropy of driving: ' + str(shannon_entropy_driving))
    print('Shannon entropy of dataset: ' + str(shannon_entropy_dataset))

