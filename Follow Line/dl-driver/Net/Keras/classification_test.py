import glob
import numpy as np
import cv2
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.models import load_model
from keras.utils import np_utils
from sklearn import metrics


def parse_json_2_classes_w(data):
    array_class = []
    # We process json
    data_parse = data.split('"classification": ')[1:]
    for d in data_parse:
        classification = d.split(', "class2":')[0]
        array_class.append(classification)

    return array_class


def parse_json_other_classes_w(data):
    array_class = []
    # We process json
    data_parse = data.split('"class2": ')[1:]
    for d in data_parse:
        classification = d.split(', "class3":')[0]
        array_class.append(classification)

    return array_class


def parse_json_other_classes_v(data):
    array_class = []
    # We process json
    data_parse = data.split('"class3": ')[1:]
    for d in data_parse:
        classification = d.split(', "w":')[0]
        array_class.append(classification)

    return array_class


def parse_json(data, num_classes, name_variable):
    if num_classes == 2 and name_variable == 'w':
        array_class = parse_json_2_classes_w(data)
    elif num_classes == 7 and name_variable == 'w':
        array_class = parse_json_other_classes_w(data)
    elif name_variable == 'v':
        array_class = parse_json_other_classes_v(data)
    return array_class



def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


def adapt_labels(array_labels, num_classes, name_variable):
    for i in range(0, len(array_labels)):
        if name_variable == 'w':
            if num_classes == 2:
                if array_labels[i] == '"left"':
                    array_labels[i] = 0
                else:
                    array_labels[i] = 1
            elif num_classes == 7:
                if array_labels[i] == '"radically_left"':
                    array_labels[i] = 0
                elif array_labels[i] == '"moderately_left"':
                    array_labels[i] = 1
                elif array_labels[i] == '"slightly_left"':
                    array_labels[i] = 2
                elif array_labels[i] == '"slight"':
                    array_labels[i] = 3
                elif array_labels[i] == '"slightly_right"':
                    array_labels[i] = 4
                elif array_labels[i] == '"moderately_right"':
                    array_labels[i] = 5
                elif array_labels[i] == '"radically_right"':
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


def choose_model(name, num_classes, name_variable):
    if name == "lenet":
        model_file = 'models/model_lenet5_7classes.h5'
    elif name == "smaller_vgg":
        model_file = 'models/model_smaller_vgg_' + str(num_classes) + 'classes_' + name_variable + '.h5'
    elif name == "other":
        model_file = 'models/model_binary_classification.h5'
    return model_file



def make_predictions(data):
    """
    Function to make the predictions over a data set
    :param data: np.array - Images to predict
    :return: np.array - Labels of predictions
    """
    predictions = model.predict(data)
    predicted = [np.argmax(prediction) for prediction in predictions]

    return np.array(predicted)


def top_k_accuracy(labels, y_predict, k):
    top_k = 0
    for i in range(0, len(labels)):
        if (labels[i] -(k-1)) <= y_predict[i] and y_predict[i] <= (labels[i] + (k-1)):
            top_k += 1
    top_k = top_k * 100 / len(labels)
    return top_k


def plot_confusion_matrix(cm, cmap=plt.cm.Blues):
    """
    Function to plot the confusion matrix
    :param cm: np.array - Confusion matrix to plot
    :param cmap: plt.cm - Color map
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()


if __name__ == "__main__":

    # Choose options
    num_classes = int(input('Choose the number of classes: '))
    name_variable = raw_input('Choose the variable you want to train: v or w: ')
    name_model = raw_input('Choose the model you want to use: mobilenet, lenet, smaller_vgg or other: ')
    print('Your choice: ' + str(num_classes) + ', ' + name_variable + ' and ' + name_model)

    # Load data
    list_images = glob.glob('../Dataset/Test/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    file = open('../Dataset/Test/test.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x_test = get_images(images)
    # We preprocess json
    y_test = parse_json(data, num_classes, name_variable)

    # We adapt string labels to int labels
    y_test = adapt_labels(y_test, num_classes, name_variable)
    labels = y_test

    # Convert the labels from integers to vectors
    y_test = np_utils.to_categorical(y_test, num_classes)

    # We adapt the data
    X_test = np.stack(x_test, axis=0)
    y_test = np.stack(y_test, axis=0)

    # Get model
    model_file = choose_model(name_model, num_classes, name_variable)

    # Load model
    print('Loading model...')
    model = load_model(model_file)

    # Make predictions
    print('Making predictions...')
    y_predict = make_predictions(X_test)

    # Evaluation
    print('Making evaluation...')
    score = model.evaluate(X_test, y_test)

    evaluation = metrics.classification_report(labels, y_predict)

    # Test loss and accuracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # We calculate top 2 accuracy
    top_2_accuracy = top_k_accuracy(labels, y_predict, 2)
    print('Top 2 accuracy: ' + str(top_2_accuracy) +'%')

    # Precision, recall, F1 score for each class
    print("Evaluation's metrics: ")
    print(evaluation)

    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(labels, y_predict)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)