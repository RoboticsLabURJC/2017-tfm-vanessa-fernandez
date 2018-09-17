"""
	Based on: https://github.com/navoshta/behavioral-cloning/blob/master/model.py
"""

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def readFiles(path, typeFormat):
	"""
	This function return the list of a certain type format

	"""
	# We read path's files and return them
	list = [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if file.endswith(typeFormat)]
	return list


def read_images(path):
	"""
	This function return the list of images

	"""
	list_imgs = []
	list = readFiles(path, ".png")
	for img in list:
		image = cv2.imread(img)
		list_imgs.append(image)
	data = np.array(list_imgs)
	return data


def process_txt(txt):
	data = txt.split('\n')[:-1]
	return data


def cnn_model(features, labels, mode):
	"""
	Model function for CNN
	"""

	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(features["x"], [-1, 320, 239, 3])

	# Convolutional Layer 1
	conv1 = tf.layers.conv2d(inputs=input_layer,
		filters=16,	kernel_size=[3, 3], padding="same",
		activation=tf.nn.relu)

	# Pooling Layer 1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer 2
	conv2 = tf.layers.conv2d(inputs=pool1,
		filters=32, kernel_size=[3, 3], padding="same",
		activation=tf.nn.relu)

	# Pooling Layer 2
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2

	# Flatten tensor into a batch of vectors
	pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 32])

	# Dense Layer
	dense = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu)

	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(inputs=dense, rate=0.5,
		training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits Layer
	# Input Tensor Shape: [batch_size, 500]
	# Output Tensor Shape: [batch_size, 100]
	logits = tf.layers.dense(inputs=dropout, units=100)


if __name__ == '__main__':
	data_img = read_images('Dataset/Images/')
	fileTxt = open('Dataset/angles.txt', 'r')
	txt = fileTxt.read()
	data_txt = process_txt(txt)

	# We separate the data
	# 60% training data and 40% testing data
	x_train = data_img[:int(0.6*len(data_img))]
	x_test = data_img[int(0.6*len(data_img)):]
	y_train = data_txt[:int(0.6*len(data_txt))]
	y_test = data_txt[int(0.6*len(data_txt)):]

