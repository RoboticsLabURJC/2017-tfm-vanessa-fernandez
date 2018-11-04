import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from time import time
from sklearn.model_selection import train_test_split
from keras.utils import plot_model, np_utils
from keras.callbacks import TensorBoard
from models.classification_model import cnn_model, lenet5, SmallerVGGNet


def parse_json_2_classes_w(data):
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


def parse_json_other_classes_w(data):
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


def parse_json_other_classes_v(data):
    array_class = []
    array_w = []
    # We process json
    data_parse = data.split('"class3": ')[1:]
    for d in data_parse:
        classification = d.split(', "class2":')[0]
        d_parse = d.split(', "w": ')[1]
        w = float(d_parse.split(', "v":')[0])
        array_class.append(classification)
        array_w.append(w)

    return array_class, array_w


def parse_json(data, num_classes, name_variable):
    if num_classes == 2 and name_variable == 'w':
        array_class, array_w = parse_json_2_classes_w(data)
    elif name_variable == 'w':
        array_class, array_w = parse_json_other_classes_w(data)
    elif name_variable == 'v':
        array_class, array_w = parse_json_other_classes_v(data)
    return array_class, array_w


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


def remove_values_aprox_zero(list_imgs, list_data, list_w):
    index = [i for i,x in enumerate(list_w) if np.isclose([x], [0.0], atol=0.08)[0] == True]
    for i in range(len(index)-1, 0, -1):
        list_data.pop(index[i])
        list_imgs.pop(index[i])
    return list_imgs, list_data


def adapt_labels(array_labels, num_classes, name_variable):
    for i in range(0, len(array_labels)):
        if name_variable == 'w':
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

        elif name_variable == 'v':
            if array_labels[i] == 'slow':
                array_labels[i] = 0
            elif array_labels[i] == 'moderate':
                array_labels[i] = 1
            elif array_labels[i] == 'fast':
                array_labels[i] = 2
            elif array_labels[i] == 'very_fast':
                array_labels[i] = 3
    return array_labels


def choose_model(name, input_shape, num_classes, name_variable):
    if name == "lenet":
        model = lenet5(input_shape, num_classes)
        model_png = 'models/model_lenet5.png'
        model_file = 'models/model_lenet5_' + str(num_classes) + 'classes_ ' + name_variable + '.h5'
        batch_size = 64
        nb_epochs = 20
    elif name == "smaller_vgg":
        model = SmallerVGGNet(input_shape, num_classes)
        model_png = 'models/model_smaller_vgg.png'
        model_file = 'models/model_smaller_vgg_' + str(num_classes) + 'classes_' + name_variable + '.h5'
        if num_classes == 7:
            batch_size = 64
            nb_epochs = 20
        else:
            batch_size = 32
            nb_epochs = 22
    elif name == "other" and num_classes == 2:
        model = cnn_model(input_shape)
        model_png = 'models/model_binary_classification.png'
        model_file = 'models/model_binary_classification.h5'
        batch_size = 32
        nb_epochs = 12
    return model, model_file, model_png, batch_size, nb_epochs


if __name__ == "__main__":

    # Choose options
    num_classes = int(input('Choose one of the two options for the number of classes: '))
    name_variable = raw_input('Choose the variable you want to train: v or w: ')
    name_model = raw_input('Choose the model you want to use: lenet, smaller_vgg or other: ')
    print('Your choice: ' + str(num_classes) + ', ' + name_variable + ' and ' + name_model)

    # Load data
    list_images = glob.glob('../Dataset/Train/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    file = open('../Dataset/Train/train.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x = get_images(images)
    # We preprocess json
    y, array_w = parse_json(data, num_classes, name_variable)

    # We delete values close to zero
    #x_train, y_train = remove_values_aprox_zero(x, y, array_w)
    x_train = x
    y_train = y

    # We adapt string labels to int labels
    y_train = adapt_labels(y_train, num_classes, name_variable)

    # https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

    # Split data into 80% for train and 20% for validation
    X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    # Convert the labels from integers to vectors
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_validation = np_utils.to_categorical(y_validation, num_classes)


    # Variables
    img_shape = (120, 160, 3)


    # Get model
    model, model_file, model_png, batch_size, nb_epochs = choose_model(name_model, img_shape, num_classes, name_variable)

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

    # Tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    #  We train
    model_history = model.fit(X_train, y_train, epochs=nb_epochs, batch_size=batch_size, verbose=2,
                                   validation_data = (X_validation, y_validation), callbacks=[tensorboard])

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
