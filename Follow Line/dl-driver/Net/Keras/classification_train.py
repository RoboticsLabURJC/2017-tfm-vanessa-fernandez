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
        classification = d.split(', "class2":')[0]
        d_parse = d.split(', "w": ')[1]
        w = float(d_parse.split(', "v":')[0])
        array_class.append(classification)
        array_w.append(w)

    return array_class, array_w


def parse_json_7_classes_w(data):
    array_class = []
    array_w = []
    # We process json
    data_parse = data.split('"class2": ')[1:]
    for d in data_parse:
        classification = d.split(', "class3":')[0]
        d_parse = d.split(', "w": ')[1]
        w = float(d_parse.split(', "v":')[0])
        array_class.append(classification)
        array_w.append(w)

    return array_class, array_w


def parse_json_9_classes_w(data):
    array_class = []
    array_w = []
    # We process json
    data_parse = data.split('"class_w_9": ')[1:]
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
        classification = d.split(', "w":')[0]
        d_parse = d.split(', "w": ')[1]
        w = float(d_parse.split(', "v":')[0])
        array_class.append(classification)
        array_w.append(w)

    return array_class, array_w


def parse_json(data, num_classes, name_variable):
    if num_classes == 2 and name_variable == 'w':
        array_class, array_w = parse_json_2_classes_w(data)
    elif num_classes == 7 and name_variable == 'w':
        array_class, array_w = parse_json_7_classes_w(data)
    elif num_classes == 9 and name_variable == 'w':
        array_class, array_w = parse_json_9_classes_w(data)
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


def adapt_label_7_w(label):
    if label == '"radically_left"' or label == 'radically_left':
        label = 0
    elif label == '"moderately_left"' or label == 'moderately_left':
        label = 1
    elif label == '"slightly_left"' or label == 'slightly_left':
        label = 2
    elif label == '"slight"' or label == 'slight':
        label = 3
    elif label == '"slightly_right"' or label == 'slightly_right':
        label = 4
    elif label == '"moderately_right"' or label == 'moderately_right':
        label = 5
    elif label == '"radically_right"' or label == 'radically_right':
        label = 6
    return label


def adapt_label_9_w(label):
    if label == '"radically_left"' or label == 'radically_left':
        label = 0
    elif label == '"strongly_left"' or label == 'strongly_left':
        label = 1
    elif label == '"moderately_left"' or label == 'moderately_left':
        label = 2
    elif label == '"slightly_left"' or label == 'slightly_left':
        label = 3
    elif label == '"slight"' or label == 'slight':
        label = 4
    elif label == '"slightly_right"' or label == 'slightly_right':
        label = 5
    elif label == '"moderately_right"' or label == 'moderately_right':
        label = 6
    elif label == '"strongly_right"' or label == 'strongly_right':
        label = 7
    elif label == '"radically_right"' or label == 'radically_right':
        label = 8
    return label


def adapt_label_4_v(label):
    if label == '"slow"' or label == 'slow':
        label = 0
    elif label == '"moderate"' or label == 'moderate':
        label = 1
    elif label == '"fast"' or label == 'fast':
        label = 2
    elif label == '"very_fast"' or label == 'very_fast':
        label = 3
    return label


def adapt_labels(array_labels, num_classes, name_variable):
    for i in range(0, len(array_labels)):
        if name_variable == 'w':
            if num_classes == 2:
                if array_labels[i] == '"left"':
                    array_labels[i] = 0
                else:
                    array_labels[i] = 1
            elif num_classes == 7:
                array_labels[i] = adapt_label_7_w(array_labels[i])
            elif num_classes == 9:
                array_labels[i] = adapt_label_9_w(array_labels[i])
        elif name_variable == 'v':
            array_labels[i] = adapt_label_4_v(array_labels[i])
    return array_labels


def choose_model(name, input_shape, num_classes, name_variable, type_net):
    if name == "lenet":
        model = lenet5(input_shape, num_classes)
        model_png = 'models/model_lenet5.png'
        model_file = 'models/model_lenet5_' + str(num_classes) + 'classes_' + type_net + '_' + name_variable + '.h5'
        batch_size = 64
        nb_epochs = 20
        if type_net == "biased":
            class_weight = {0: 3., 1: 2., 2: 1., 3: 1., 4: 1., 5: 2., 6: 3.}
        else:
            class_weight = None
    elif name == "smaller_vgg":
        model = SmallerVGGNet(input_shape, num_classes)
        model_png = 'models/model_smaller_vgg.png'
        model_file = 'models/model_smaller_vgg_' + str(num_classes) + 'classes_' + type_net + '_' + \
                     name_variable + '.h5'
        if num_classes == 7:
            batch_size = 64
            nb_epochs = 45
            if type_net == "biased":
                nb_epochs = 55
                class_weight = {0: 4., 1: 2., 2: 2., 3: 1., 4:2., 5: 2., 6: 3.}
            else:
                class_weight = None
        elif num_classes == 9:
            batch_size = 64
            nb_epochs = 34
            if type_net == "biased":
                class_weight = {0: 6., 1: 5., 2: 2., 3: 1., 4: 1., 5: 1., 6: 2., 7: 5., 8: 6.}
            else:
                class_weight = None
        else:
            batch_size = 64
            nb_epochs = 55
            if type_net == "biased":
                class_weight = {0: 2., 1: 3., 2: 3., 3: 4.}
            else:
                class_weight = None
    elif name == "other" and num_classes == 2:
        model = cnn_model(input_shape)
        model_png = 'models/model_binary_classification.png'
        model_file = 'models/model_binary_classification.h5'
        batch_size = 32
        nb_epochs = 12
        if type_net == "biased":
            class_weight = {0: 1., 1: 1.}
        else:
            class_weight = None
    return model, model_file, model_png, batch_size, nb_epochs, class_weight


if __name__ == "__main__":

    # Choose options
    num_classes = int(input('Choose one of the options for the number of classes: '))
    name_variable = raw_input('Choose the variable you want to train: v or w: ')
    type_net = raw_input('Choose the type of network you want: normal, biased or balanced: ')
    name_model = raw_input('Choose the model you want to use: lenet, smaller_vgg or other: ')
    print('Your choice: ' + str(num_classes) + ', ' + name_variable + ', ' + type_net + ' and ' + name_model)

    # Load data
    if type_net == 'balanced':
        if name_variable == 'w':
            list_images = glob.glob('../Dataset/Train_balanced_bbdd_w/Images/' + '*')
            images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))
            file = open('../Dataset/Train_balanced_bbdd_w/train.json', 'r')
            data = file.read()
            file.close()
        elif name_variable == 'v':
            list_images = glob.glob('../Dataset/Train_balanced_bbdd_v/Images/' + '*')
            images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))
            file = open('../Dataset/Train_balanced_bbdd_v/train.json', 'r')
            data = file.read()
            file.close()
    else:
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
    if type_net == 'balanced':
        X_train = x_train
        y_train = y_train
        X_t, X_validation, y_t, y_validation = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
    else:
        X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.20,
                                                                        random_state=42)

    # Convert the labels from integers to vectors
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_validation = np_utils.to_categorical(y_validation, num_classes)


    # Variables
    img_shape = (120, 160, 3)


    # Get model
    model, model_file, model_png, batch_size, nb_epochs, class_weight = choose_model(name_model, img_shape,
                                                                                     num_classes, name_variable,
                                                                                     type_net)

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
                              class_weight=class_weight, validation_data=(X_validation, y_validation),
                              callbacks=[tensorboard])

    #step_epoch = 15

    #model_history = model.fit(X_train, y_train, epochs=nb_epochs, verbose=2, class_weight=class_weight,
    #                          validation_data = (X_validation, y_validation), steps_per_epoch= step_epoch,
    #                          validation_steps = 15, callbacks=[tensorboard])


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
