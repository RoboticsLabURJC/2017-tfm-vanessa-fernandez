#  Based on: https://arxiv.org/pdf/1604.07316.pdf
#
#  Authors :
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, ConvLSTM2D, Reshape
from keras.optimizers import Adam


def pilotnet_model(img_shape):
    '''
    Model of End to End Learning for Self-Driving Cars (NVIDIA)
    '''
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.00001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def tinypilotnet_model(img_shape):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


def lstm_tinypilotnet_model(img_shape):
    model = Sequential()
    model.add(BatchNormalization(epsilon=0.001, axis=-1, input_shape=img_shape))
    model.add(Conv2D(8, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(16, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Conv2D(32, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Reshape((1, img_shape[0], img_shape[1], img_shape[2], 32)))
    model.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3, border_mode='same', return_sequences=True))
    model.add(Reshape((img_shape[0], img_shape[1], img_shape[2], 40)))
    model.add(Conv2D(1, (3, 3), strides=(2, 2), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model