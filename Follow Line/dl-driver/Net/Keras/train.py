import glob
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')

from time import time
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
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
        array_v.append(float(v))
        array_w.append(float(w))

    return array_v, array_w


def get_images(list_images, type_image):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        if type_image == 'cropped':
            img = img[220:480, 0:640]
        if type_net == 'lstm':
            img = cv2.resize(img, (img.shape[1] / 10, img.shape[0] / 10))
        else:
            img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


def add_extreme_data(array_w, imgs_w, array_v, imgs_v):
    for i in range(0, len(array_w)):
        if abs(array_w[i]) >= 1:
            if abs(array_w[i]) >= 2:
                num_iter = 10
            else:
                num_iter = 5
            for j in range(0, num_iter):
                array_w.append(array_w[i])
                imgs_w.append(imgs_w[i])
        if float(array_v[i]) <= 2:
            for j in range(0, 1):
                array_v.append(array_v[i])
                imgs_v.append(imgs_v[i])
    return array_w, imgs_w, array_v, imgs_v


def preprocess_data(array_w, array_v, imgs):
    # We take the image and just flip it and negate the measurement
    flip_imgs = []
    array_flip_w = []
    for i in range(len(array_w)):
        flip_imgs.append(cv2.flip(imgs[i], 1))
        array_flip_w.append(-array_w[i])
    new_array_w = array_w + array_flip_w
    new_array_v = array_v + array_v
    new_array_imgs = imgs + flip_imgs
    return new_array_w, new_array_v, new_array_imgs


def stack_frames(imgs, type_net):
    new_imgs = []
    margin = 10
    for i in range(0, len(imgs)):
        # if i - 2*(margin+1) < 0:
        #     index1 = 0
        # else:
        #     index1 = i - 2*(margin+1)
        if i - (margin + 1) < 0:
            index2 = 0
        else:
            index2 = i - (margin + 1)
        #im1 =  np.concatenate([imgs[index1], imgs[index2]], axis=2)
        #im2 = np.concatenate([im1, imgs[i]], axis=2)
        if type_net == 'stacked_dif':
            im = imgs[i] - imgs[index2]
            im2 = np.concatenate([im, imgs[i]], axis=2)
        else:
            im2 = np.concatenate([imgs[index2], imgs[i]], axis=2)
        new_imgs.append(im2)
    return new_imgs


def choose_model(type_net, img_shape, type_image):
    model_png = 'models/model_' + type_net + '.png'
    if type_image == 'cropped':
        model_file_v = 'models/model_' + type_net + '_' + type_image + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_' + type_image + '_w.h5'
    else:
        model_file_v = 'models/model_' + type_net + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_w.h5'
    if type_net == 'pilotnet':
        model_v = pilotnet_model(img_shape)
        model_w = pilotnet_model(img_shape)
        batch_size_v = 64#16
        batch_size_w = 64
        nb_epoch_v = 200#223
        nb_epoch_w = 212#212
    elif type_net == 'tinypilotnet':
        model_v = tinypilotnet_model(img_shape)
        model_w = tinypilotnet_model(img_shape)
        batch_size_v = 64#16
        batch_size_w = 64
        nb_epoch_v = 250 #223
        nb_epoch_w = 1000 #212
    elif type_net == 'stacked' or type_net == 'stacked_dif':
        model_v = pilotnet_model(img_shape)
        model_w = pilotnet_model(img_shape)
        batch_size_v = 64
        batch_size_w = 64
        nb_epoch_v = 300
        nb_epoch_w = 250
    elif type_net == 'lstm_tinypilotnet':
        model_v = lstm_tinypilotnet_model(img_shape, type_image)
        model_w = lstm_tinypilotnet_model(img_shape, type_image)
        batch_size_v = 12 #8
        batch_size_w = 12 #8
        nb_epoch_v = 150#223
        nb_epoch_w = 150#212
    elif type_net == 'deepestlstm_tinypilotnet':
        model_v = deepestlstm_tinypilotnet_model(img_shape, type_image)
        model_w = deepestlstm_tinypilotnet_model(img_shape, type_image)
        batch_size_v = 12 #8
        batch_size_w = 12 #8
        nb_epoch_v = 150#223
        nb_epoch_w = 100#212

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
    type_image = raw_input('Choose the type of image you want: normal or cropped: ')
    type_net = raw_input('Choose the type of network you want: pilotnet, tinypilotnet, lstm_tinypilotnet, lstm, '
                         'deepestlstm_tinypilotnet, stacked or stacked_dif: ')
    print('Your choice: ' + type_net + ', ' + type_image)

    # Load data
    list_images = glob.glob('../Dataset/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))
    file = open('../Dataset/data.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x = get_images(images, type_image)
    # We preprocess json
    y_v, y_w = parse_json(data)


    # Split data into 80% for train and 20% for validation
    if type_net == 'pilotnet' or type_net == 'tinypilotnet':
        # We adapt the data
        #x_w = x[:]
        #x_v = x[:]
        #y_w, x_w, y_v, x_v = add_extreme_data(y_w, x_w, y_v, x_v)
        y_w, y_v, x = preprocess_data(y_w, y_v, x)
        x_w = x[:]
        x_v = x[:]
        y_w, x_w, y_v, x_v = add_extreme_data(y_w, x_w, y_v, x_v)
        X_train_v, X_validation_v, y_train_v, y_validation_v = train_test_split(x_v, y_v, test_size=0.20,
                                                                                random_state=42)
        X_train_w, X_validation_w, y_train_w, y_validation_w = train_test_split(x_w, y_w, test_size=0.20,
                                                                                random_state=42)
        #X_train_v, X_validation_v, y_train_v, y_validation_v = train_test_split(x_v,y_v,test_size=0.20,random_state=42)
        #X_train_w, X_validation_w, y_train_w, y_validation_w = train_test_split(x_w,y_w,test_size=0.20,random_state=42)
    elif type_net == 'stacked' or type_net == 'stacked_dif':
        # We stack frames
        y_w, y_v, x = preprocess_data(y_w, y_v, x)
        x = stack_frames(x, type_net)
        X_train_v, X_validation_v, y_train_v, y_validation_v = train_test_split(x, y_v, test_size=0.20, random_state=42)
        X_train_w, X_validation_w, y_train_w, y_validation_w = train_test_split(x, y_w, test_size=0.20, random_state=42)
    elif type_net == 'lstm_tinypilotnet' or type_net == 'lstm' or type_net == 'deepestlstm_tinypilotnet':
        y_w, y_v, x = preprocess_data(y_w, y_v, x)
        X_train_v = x
        X_train_w = x
        y_train_v = y_v
        y_train_w = y_w
        X_t_v, X_validation_v, y_t_v, y_validation_v = train_test_split(x, y_v, test_size=0.20, random_state=42)
        X_t_w, X_validation_w, y_t_w, y_validation_w = train_test_split(x, y_w, test_size=0.20, random_state=42)

    # Variables
    if type_net == 'stacked' or type_net == 'stacked_dif':
        if type_image == 'cropped':
            #img_shape = (65, 160, 9)
            img_shape = (65, 160, 6)
        else:
            #img_shape = (120, 160, 9)
            img_shape = (120, 160, 6)
    elif type_net == 'lstm':
        if type_image == 'cropped':
            img_shape = (26, 64, 3)
        else:
            img_shape = (48, 64, 3)
    else:
        if type_image == 'cropped':
            img_shape = (65, 160, 3)
        else:
            img_shape = (120, 160, 3)

    # We adapt the data
    X_train_v = np.stack(X_train_v, axis=0)
    y_train_v = np.stack(y_train_v, axis=0)
    X_validation_v = np.stack(X_validation_v, axis=0)
    y_validation_v = np.stack(y_validation_v, axis=0)

    X_train_w = np.stack(X_train_w, axis=0)
    y_train_w = np.stack(y_train_w, axis=0)
    X_validation_w = np.stack(X_validation_w, axis=0)
    y_validation_w = np.stack(y_validation_w, axis=0)

    # Get model
    model_v, model_w, model_file_v, model_file_w, model_png, batch_size_v, nb_epoch_v, batch_size_w, \
    nb_epoch_w = choose_model(type_net, img_shape, type_image)

    # Print layers
    print(model_v.summary())
    # Plot layers of model
    plot_model(model_v, to_file=model_png)

    #  We train
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    if not os.path.exists('csv'): os.makedirs('csv')
    filename = 'csv/' + type_net + '_' + type_image + '_v.csv'
    csv_logger = CSVLogger(filename=filename, separator = ',', append = True)

    model_checkpoint = ModelCheckpoint(model_file_v,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       monitor='val_loss',
                                       verbose=1)

    model_history_v = model_v.fit(X_train_v, y_train_v, epochs=nb_epoch_v, batch_size=batch_size_v, verbose=2,
                                  validation_data=(X_validation_v, y_validation_v),
                                  callbacks=[tensorboard, model_checkpoint, csv_logger])


    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

    filename = 'csv/' + type_net + '_' + type_image + '_w.csv'
    csv_logger = CSVLogger(filename=filename, separator=',', append=True)

    model_checkpoint = ModelCheckpoint(model_file_w,
                                       save_best_only=True,
                                       save_weights_only=False,
                                       monitor='val_loss',
                                       verbose=1)

    model_history_w = model_w.fit(X_train_w, y_train_w, epochs=nb_epoch_w, batch_size=batch_size_w, verbose=2,
                                  validation_data=(X_validation_w, y_validation_w),
                                  callbacks=[tensorboard, model_checkpoint, csv_logger])

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
    #model_v.save(model_file_v)
    #model_w.save(model_file_w)

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
