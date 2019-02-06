"""

Example of LSTM (images)

"""

import glob
import cv2
import numpy as np

from time import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, BatchNormalization, Dropout, Reshape, MaxPooling2D, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1] / 6, img.shape[0] / 6))
        array_imgs.append(img)

    return array_imgs


def lstm_model(img_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=img_shape, activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(128, (3, 3), padding='same', activation="relu"))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Reshape((1024, 1)))
    model.add(LSTM(10, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(10))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation="relu"))
    model.add(Dense(1))
    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse", metrics=['accuracy', 'mse', 'mae'])
    return model


if __name__ == "__main__":

    # Load data
    list_images = glob.glob('Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[1].split('.png')[0]))

    y = [71.71, 56.19, -44.51, 61.90, 67.86, -61.52, -75.73, 44.75, -89.51, 44.75]
    # We preprocess images
    x = get_images(images)

    X_train = x
    y_train = y
    X_t, X_val, y_t, y_val = train_test_split(x, y, test_size=0.20, random_state=42)

    # Variables
    batch_size = 8
    nb_epoch = 200
    img_shape = (39, 53, 3)


    # We adapt the data
    X_train = np.stack(X_train, axis=0)
    y_train = np.stack(y_train, axis=0)
    X_val = np.stack(X_val, axis=0)
    y_val = np.stack(y_val, axis=0)


    # Get model
    model = lstm_model(img_shape)

    model_history_v = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=2,
                              validation_data=(X_val, y_val))
    print(model.summary())


    # We evaluate the model
    score = model.evaluate(X_val, y_val, verbose=0)
    print('Evaluating v')
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('Test mean squared error: ', score[2])
    print('Test mean absolute error: ', score[3])

