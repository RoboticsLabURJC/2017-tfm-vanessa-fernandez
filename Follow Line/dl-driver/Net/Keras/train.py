import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')

from time import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from models.model_nvidia import *


def parse_json(data):
    array_v = []
    array_w = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        array_v.append(format(float(v), '.6f'))
        array_w.append(float(w))

    return array_v, array_w


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        #img = cv2.resize(img, (img.shape[1] / 8, img.shape[0] / 8))
        array_imgs.append(img)

    return array_imgs


def add_extreme_data(array_w, imgs_w, array_v, imgs_v):
    for i in range(0, len(array_w)):
        if abs(array_w[i]) >= 1:
            for j in range(0, 5):
                array_w.append(array_w[i])
                imgs_w.append(imgs_w[i])
        if float(array_v[i]) <= 2:
            for j in range(0, 5):
                array_v.append(array_v[i])
                imgs_v.append(imgs_v[i])
    return array_w, imgs_w, array_v, imgs_v


def choose_model(type_net, img_shape):
    model_png = 'models/model_' + type_net + '.png'
    model_file_v = 'models/model_' + type_net + '_v.h5'
    model_file_w = 'models/model_' + type_net + '_w.h5'
    if type_net == 'pilotnet':
        model_v = pilotnet_model(img_shape)
        model_w = pilotnet_model(img_shape)
        batch_size_v = 64#16
        batch_size_w = 64
        nb_epoch_v = 223
        nb_epoch_w = 212
    elif type_net == 'tinypilotnet':
        model_v = tinypilotnet_model(img_shape)
        model_w = tinypilotnet_model(img_shape)
        batch_size_v = 64#16
        batch_size_w = 64
        nb_epoch_v = 223
        nb_epoch_w = 212
    elif type_net == 'lstm_tinypilotnet':
        model_v = lstm_tinypilotnet_model(img_shape)
        model_w = lstm_tinypilotnet_model(img_shape)
        batch_size_v = 12 #8
        batch_size_w = 12 #8
        nb_epoch_v = 350#223
        nb_epoch_w = 350#212
    elif type_net == 'lstm':
        model_v = lstm_model(img_shape)
        model_w = lstm_model(img_shape)
        batch_size_v = 12 #8
        batch_size_w = 12 #8
        nb_epoch_v = 200#223
        nb_epoch_w = 200#212
    return model_v, model_w, model_file_v, model_file_w, model_png, batch_size_v, nb_epoch_v, batch_size_w, nb_epoch_w


if __name__ == "__main__":
    # Choose options
    type_net = raw_input('Choose the type of network you want: pilotnet, tinypilotnet, lstm_tinypilotnet or lstm: ')
    print('Your choice: ' + type_net)

    # Load data
    if type_net == 'pilotnet' or type_net == 'tinypilotnet':
        #list_images = glob.glob('../Dataset/Train/Images/' + '*')
        #images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))
        #file = open('../Dataset/Train/train.json', 'r')
        list_images = glob.glob('../Dataset/Images/' + '*')
        images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
        file = open('../Dataset/data.json', 'r')
    elif type_net == 'lstm_tinypilotnet' or type_net == 'lstm':
        list_images = glob.glob('../Dataset/Images/' + '*')
        images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
        file = open('../Dataset/data.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x = get_images(images)
    # We preprocess json
    y_v, y_w = parse_json(data)

    # Split data into 80% for train and 20% for validation
    if type_net == 'pilotnet' or type_net == 'tinypilotnet':
        # We adapt the data
        x_w = x[:]
        x_v = x[:]
        y_w, x_w, y_v, x_v = add_extreme_data(y_w, x_w, y_v, x_v)
        X_train_v, X_validation_v, y_train_v, y_validation_v = train_test_split(x_v, y_v, test_size=0.20, random_state=42)
        X_train_w, X_validation_w, y_train_w, y_validation_w = train_test_split(x_w, y_w, test_size=0.20, random_state=42)
    elif type_net == 'lstm_tinypilotnet' or type_net == 'lstm':
        X_train_v = x
        X_train_w = x
        y_train_v = y_v
        y_train_w = y_w
        X_t_v, X_validation_v, y_t_v, y_validation_v = train_test_split(x, y_v, test_size=0.20, random_state=42)
        X_t_w, X_validation_w, y_t_w, y_validation_w = train_test_split(x, y_w, test_size=0.20, random_state=42)

    # Variables
    img_shape = (120, 160, 3)
    #img_shape = (60, 80, 3)


    # We adapt the data
    X_train_v = np.stack(X_train_v, axis=0)
    y_train_v = np.stack(y_train_v, axis=0)
    X_validation_v = np.stack(X_validation_v, axis=0)
    y_validation_v = np.stack(y_validation_v, axis=0)
    # print(X_train_v.shape)
    # print(type(X_train_v))
    # X_train_v = np.reshape(X_train_v, (len(X_train_v), 10, img_shape[0], img_shape[1], img_shape[2]))
    # y_train_v = np.reshape(y_train_v, (len(X_train_v), 10, img_shape[0], img_shape[1], img_shape[2]))
    # X_validation_v = np.reshape(X_validation_v, (len(X_validation_v), 10, img_shape[0], img_shape[1], img_shape[2]))
    # y_validation_v = np.reshape(y_validation_v, (len(X_validation_v), 10, img_shape[0], img_shape[1], img_shape[2]))
    # print(X_train_v.shape)


    X_train_w = np.stack(X_train_w, axis=0)
    y_train_w = np.stack(y_train_w, axis=0)
    X_validation_w = np.stack(X_validation_w, axis=0)
    y_validation_w = np.stack(y_validation_w, axis=0)
    # X_train_w = np.reshape(X_train_w, (len(X_train_w), 10, img_shape[0], img_shape[1], img_shape[2]))
    # y_train_w = np.reshape(y_train_w, (len(X_train_w), 10, img_shape[0], img_shape[1], img_shape[2]))
    # X_validation_w = np.reshape(X_validation_w, (len(X_validation_w), 10, img_shape[0], img_shape[1], img_shape[2]))
    # y_validation_w = np.reshape(y_validation_w, (len(X_validation_w), 10, img_shape[0], img_shape[1], img_shape[2]))

    #img_shape = (len(X_train_v), 10, img_shape[0], img_shape[1], img_shape[2])

    # Get model
    model_v, model_w, model_file_v, model_file_w, model_png, batch_size_v, nb_epoch_v, batch_size_w, \
    nb_epoch_w = choose_model(type_net, img_shape)

    # Print layers
    print(model_v.summary())
    # Plot layers of model
    plot_model(model_v, to_file=model_png)

    #  We train
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model_history_v = model_v.fit(X_train_v, y_train_v, epochs=nb_epoch_v, batch_size=batch_size_v, verbose=2,
                              validation_data=(X_validation_v, y_validation_v), callbacks=[tensorboard])

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    model_history_w = model_w.fit(X_train_w, y_train_w, epochs=nb_epoch_w, batch_size=batch_size_w, verbose=2,
                                  validation_data=(X_validation_w, y_validation_w), callbacks=[tensorboard])

    # We evaluate the model
    score = model_v.evaluate(X_validation_v, y_validation_v, verbose=0)
    print('Evaluating v')
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('Test mean squared error: ', score[2])
    print('Test mean absolute error: ', score[3])

    score = model_w.evaluate(X_validation_w, y_validation_w, verbose=0)
    print('Evaluating w')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Test mean squared error: ', score[2])
    print('Test mean absolute error: ', score[3])

    # We save the model
    model_v.save(model_file_v)
    model_w.save(model_file_w)

    # Plot the training and validation loss for each epoch
    # plt.plot(model_history.history['loss'])
    # plt.plot(model_history.history['val_loss'])
    # plt.title('mse')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # plt.ylim([0, 0.1])
    # plt.show()
    #
    # # Accuracy Curves
    # plt.figure(figsize=[8, 6])
    # plt.plot(model_history.history['acc'], 'r', linewidth=3.0)
    # plt.plot(model_history.history['val_acc'], 'b', linewidth=3.0)
    # plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    # plt.xlabel('Epochs ', fontsize=16)
    # plt.ylabel('Accuracy', fontsize=16)
    # plt.title('Accuracy Curves', fontsize=16)
    # plt.show()
