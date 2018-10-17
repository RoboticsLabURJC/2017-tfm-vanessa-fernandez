import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras

from sklearn.model_selection import train_test_split
from keras.utils import plot_model, np_utils
from keras.preprocessing.image import img_to_array
from keras.applications.mobilenet import MobileNet
from models.classification_model import cnn_model
from models.lenet5 import lenet5


def parse_json_2_classes(data):
    array_class = []
    array_w = []
    # We process json
    data_parse = data.split('"classification": ')[1:]
    for d in data_parse:
        classification = d.split(', "w":')[0]
        d_parse = d.split(', "w": ')[1]
        w = float(d_parse.split(', "v":')[0])
        array_class.append(classification)
        array_w.append(w)

    return array_class, array_w


def parse_json_7_classes(data):
    array_class = []
    array_w = []
    # We process json
    data_parse = data.split('"class2": ')[1:]
    for d in data_parse:
        classification = d.split(', "classification":')[0]
        d_parse = d.split(', "w": ')[1]
        w = float(d_parse.split(', "v":')[0])
        array_class.append(classification)
        array_w.append(w)

    return array_class, array_w


def parse_json(data, num_classes):
    if num_classes == 2:
        array_class, array_w = parse_json_2_classes(data)
    elif num_classes == 7:
        array_class, array_w = parse_json_7_classes(data)
    return array_class, array_w


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
        #img = img_to_array(img)
        array_imgs.append(img)

    return array_imgs


def remove_values_aprox_zero(list_imgs, list_data, list_w):
    index = [i for i,x in enumerate(list_w) if np.isclose([x], [0.0], atol=0.08)[0] == True]
    for i in range(len(index)-1, 0, -1):
        list_data.pop(index[i])
        list_imgs.pop(index[i])
    return list_imgs, list_data


def adapt_labels(array_labels, num_classes):
    for i in range(0, len(array_labels)):
        if num_classes == 2:
            if array_labels[i] == '"left"':
                array_labels[i] = 0
            else:
                array_labels[i] = 1
        elif num_classes == 7:
            if array_labels[i] == 'radically_left':
                array_labels[i] = 0
            elif array_labels[i] == 'moderately_left':
                array_labels[i] = 1
            elif array_labels[i] == 'slightly_left':
                array_labels[i] = 2
            elif array_labels[i] == 'slight':
                array_labels[i] = 3
            elif array_labels[i] == 'slightly_right':
                array_labels[i] = 4
            elif array_labels[i] == 'moderately_right':
                array_labels[i] = 5
            elif array_labels[i] == 'radically_right':
                array_labels[i] = 6

    return array_labels


# def adapt_data(x_train, y_train):
#     num = 16
#     for i in range(0, 3):
#         d2 = np.array(x_train[i*(len(x_train)/num):(i+1)*len(x_train)/num]).astype(np.float32) / 255.0
#         d4 = np.array(y_train[i*(len(y_train)/num):(i+1)*len(y_train)/num])
#         if i != 0:
#             new_x_train = np.hstack((d1, d2))
#             new_y_train = np.hstack((d3, d4))
#             d1 = new_x_train
#             d3 = new_y_train
#         else:
#             d1 = d2
#             d3 = d4
#
#     return new_x_train, new_y_train


def choose_model(name, input_shape, num_classes):
    if name == "mobilenet":
        base_model = MobileNet(weights=None, include_top='avg', input_shape=input_shape, classes=num_classes)
        # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        # x = keras.layers.AveragePooling2D((7, 7))(base_model.output)
        x = keras.layers.Dropout(0.3)(base_model.output)
        output = keras.layers.Dense(1)(x)
        model = keras.models.Model(inputs=[base_model.input], outputs=[output])
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model_png = 'models/model_mobilenet.png'
        model_file = 'models/model_mobilenet.h5'
        batch_size = 128
        nb_epochs = 20
    elif name == "lenet":
        model = lenet5(input_shape, num_classes)
        model_png = 'models/model_lenet5_7classes.png'
        model_file = 'models/model_lenet5_7classes.h5'
        batch_size = 128
        nb_epochs = 20
    elif name == "other":
        model = cnn_model(input_shape)
        model_png = 'models/model_classification.png'
        model_file = 'models/model_classification.h5'
        batch_size = 32
        nb_epochs = 12
    return model, model_file, model_png, batch_size, nb_epochs


if __name__ == "__main__":

    # Choose options
    num_classes = int(input('Choose one of the two options for the number of classes: 2 or 7: '))
    name_model = raw_input('Choose the model you want to use: mobilenet, lenet, or other: ')
    print('Your choice: ' + str(num_classes) + ' and ' + name_model)

    # Load data
    list_images = glob.glob('../Dataset/Train/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    file = open('../Dataset/Train/train.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x = get_images(images)
    # We preprocess json
    y, array_w = parse_json(data, num_classes)

    # We delete values close to zero
    x_train, y_train = remove_values_aprox_zero(x, y, array_w)

    # We adapt string labels to int labels
    y_train = adapt_labels(y_train, num_classes)

    # We adapt the data
    # print(len(x_train), len(y_train))
    # print(type(x_train))
    #
    # x_train, y_train = adapt_data(x_train, y_train)
    # print(len(x_train), len(y_train))

    # https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

    # d1 = np.array(x_train[:len(x_train)/4]).astype(np.float32) / 255.0
    # d2 = np.array(x_train[len(x_train)/4:len(x_train)/2]).astype(np.float32) / 255.0
    # d3 = np.array(x_train[len(x_train)/2:len(x_train)*3/4]).astype(np.float32) / 255.0
    # d4 = np.array(x_train[len(x_train)*3/4:]).astype(np.float32) / 255.0
    # y_train = np.array(y_train)
    # x_train = np.hstack(d1, d2, d3, d4)
    #x_train = np.array(x_train)
    #y_train = np.array(y_train)

    # Split data into 80% for train and 20% for validation
    X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    # Convert the labels from integers to vectors
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_validation = np_utils.to_categorical(y_validation, num_classes)


    # Variables
    #batch_size = 32
    #nb_epochs = 12
    img_shape = (240, 320, 3)


    # Get model
    model, model_file, model_png, batch_size, nb_epochs = choose_model(name_model, img_shape, num_classes)

    # We adapt the data
    X_train = np.stack(X_train, axis=0)
    y_train = np.stack(y_train, axis=0)
    X_validation = np.stack(X_validation, axis=0)
    y_validation = np.stack(y_validation, axis=0)

    print('X train',  X_train.shape)
    print('y train',  y_train.shape)
    print('X validation',  X_validation.shape)
    print('y val',  y_validation.shape)


    # Print layers
    print(model.summary())
    # Plot layers of model
    plot_model(model, to_file=model_png)

    #  We train
    model_history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, verbose=2,
                                   validation_data = (X_validation, y_validation))

    # We evaluate the model
    score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # We save the model
    model.save(model_file)


    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(model_history.history['loss'], 'r', linewidth=3.0)
    plt.plot(model_history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(model_history.history['acc'], 'r', linewidth=3.0)
    plt.plot(model_history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()
