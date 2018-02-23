# It trains and tests a convolutional neural network 
# MNIST dataset

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend
from keras.utils import np_utils, io_utils
from cnn import CNN


# Seed for the computer pseudorandom number generator
np.random.seed(123)


if __name__ == '__main__':
	batch_size = 128
	num_classes = 10
	nb_epochs = 12
	ngpus =  4

	# Input image dimensions
	img_rows, img_cols = 28, 28

	# The data shuffled and split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	# Check the backend
	if backend.image_dim_ordering() == "th":
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape, 'x_test.shape:', x_test.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# Convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, num_classes)
	y_test = np_utils.to_categorical(y_test, num_classes)

	# Build the model
	model = CNN(num_classes, input_shape, ngpus)
	model.fit(x_train, y_train, batch_size=batch_size*ngpus, epochs=nb_epochs, verbose=1, validation_data=(x_test, y_test))

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
