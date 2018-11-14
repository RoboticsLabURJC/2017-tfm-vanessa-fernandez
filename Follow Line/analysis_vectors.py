#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def calculate_postion_vectors(img):
    # We look for the position on the x axis of the pixels that have value 1 in different positions and
    position_x_down = np.where(img[350, :])
    position_x_above = np.where(img[260, :])
    return position_x_down, position_x_above


def calculate_centroid(positionx):
    if (len(positionx[0]) > 1):
        x_middle = (positionx[0][0] + positionx[0][len(positionx[0]) - 1]) / 2
        not_found = False
    else:
        x_middle = None
        not_found = True
    return x_middle, not_found


def draw_centroids(array_images, marker, ax1, ax2, ax3):
    for i in range(0, len(array_images)):
        img = filter_image(array_images[i])
    
        # We calculate vectors
        position_x_down, position_x_above = calculate_postion_vectors(img)

        # We see that white pixels have been located and we look if the center is located
        x_middle_down, not_found_down = calculate_centroid(position_x_down)
        x_middle_above, not_found_above = calculate_centroid(position_x_above)

        print(x_middle_down, not_found_down, x_middle_above, not_found_above)

        if not_found_down:
            ax3.plot([0.5], [x_middle_above], marker)
        elif not_found_above:
            ax1.plot([x_middle_down], [0.5], marker)
        else:
            ax2.plot([x_middle_down], [x_middle_above], marker)

    return ax1, ax2, ax3


if __name__ == "__main__":
    # Load data
    list_images_dataset = glob.glob('Dataset/Train/Images/' + '*')
    images_dataset = sorted(list_images_dataset, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    list_images_driving = glob.glob('Failed_driving/Images/' + '*')
    images_driving = sorted(list_images_driving, key=lambda x: int(x.split('/')[2].split('.png')[0]))

	# We preprocess images
    array_images_dataset = get_images(images_dataset)
    array_images_driving = get_images(images_driving)

    # We create the figure and subplots
    fig = plt.figure()
    plt.suptitle('Datatset against Driving')

    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4])

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[1])
    ax1.set_title('Nan values of L1')
    ax2.set_title('Represent pairs of L1-L2')
    ax3.set_title('Nan values of L2')
    ax4.set_title('Legend')

    ax1, ax2, ax3 = draw_centroids(array_images_dataset, 'ro', ax1, ax2, ax3)
    ax1, ax2, ax3 = draw_centroids(array_images_driving, 'bx', ax1, ax2, ax3)

    ax1.axis([0, 640, 0, 1])
    ax2.axis([0, 640, 0, 640])
    ax2.set_xlabel('L2 (Row 350)')
    ax2.set_ylabel('L1 (Row 260)')
    ax3.axis([0, 1, 0, 640])
    ax4.axis([0, 1, 0, 1])
    ax4.plot([-1], [-1], 'ro', label='Dataset')
    ax4.plot([-1], [-1], 'bx', label='Driving')
    plt.legend()
    plt.show()

