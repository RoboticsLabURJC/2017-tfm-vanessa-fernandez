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
	# For normalizing dataset
	scaler = MinMaxScaler()

	# We want to predict angle(txt) value
	X_train = scaler.fit_transform(x_train.reshape(-1, 1))
	y_train = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
	X_test = scaler.fit_transform(x_test.reshape(-1, 1))
	y_test = scaler.fit_transform(np.array(y_test).reshape(-1, 1))

	# tf.placeholder() will define gateway for data to graph
	xs = tf.placeholder("float")
	ys = tf.placeholder("float")

