import cv2
import glob
import numpy as np

from keras.models import load_model
from keras.utils import np_utils


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


def parse_json_7_classes_w(data):
    array_class = []
    # We process json
    data_parse = data.split('"class2": ')[1:]
    for d in data_parse:
        classification = d.split(', "classification":')[0]
        array_class.append(classification)

    return array_class


def parse_json_4_classes_v(data):
    array_class = []
    # We process json
    data_parse = data.split('"class3": ')[1:]
    for d in data_parse:
        classification = d.split(', "class2":')[0]
        array_class.append(classification)

    return array_class


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


def adapt_labels(array_labels, variable):
    for i in range(0, len(array_labels)):
        if variable == 'w':
            array_labels[i] = adapt_label_7_w(array_labels[i])
        elif variable == 'v':
            array_labels[i] = adapt_label_4_v(array_labels[i])
    return array_labels


def make_predictions(data):
    """
    Function to make the predictions over a data set
    :param data: np.array - Images to predict
    :return: np.array - Labels of predictions
    """
    predictions = model.predict(data)
    predicted = [np.argmax(prediction) for prediction in predictions]

    return np.array(predicted)


def adapt_predictions_w(predictions):
    predicts = []
    for i in range(0, len(predictions)):
        if predictions[i] == 0:
            predicts.append('radically_left')
        elif predictions[i] == 1:
            predicts.append('moderately_left')
        elif predictions[i] == 2:
            predicts.append('slightly_left')
        elif predictions[i] == 3:
            predicts.append('slight')
        elif predictions[i] == 4:
            predicts.append('slightly_right')
        elif predictions[i] == 5:
            predicts.append('moderately_right')
        elif predictions[i] == 6:
            predicts.append('radically_right')
    return predicts


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


def adapt_predictions_v(predictions):
    predicts = []
    for i in range(0, len(predictions)):
        if predictions[i] == 0:
            predicts.append('slow')
        elif predictions[i] == 1:
            predicts.append('moderate')
        elif predictions[i] == 2:
            predicts.append('fast')
        elif predictions[i] == 3:
            predicts.append('very_fast')
    return predicts


def calculate_accuracy(labels, y_predict):
    top_k = 0
    for i in range(0, len(labels)):
        if labels[i] == y_predict[i]:
            top_k += 1
    top_k = top_k * 100 / len(labels)
    return top_k


if __name__ == "__main__":
    # Load data
    file = open('4v_7w/corrected_data.json', 'r')
    data_corrected = file.read()
    file.close()

    # We preprocess json
    array_w_corrected = parse_json_7_classes_w(data_corrected)
    array_v_corrected = parse_json_4_classes_v(data_corrected)

    # We process images
    list_images = glob.glob('4v_7w/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[2].split('.png')[0]))
    li = get_images(images)


    # We adapt string labels to int labels
    lab_w_str = array_w_corrected[:]
    labels_w = adapt_labels(array_w_corrected, 'w')
    lab_v_str = array_v_corrected[:]
    labels_v = adapt_labels(array_v_corrected, 'v')

    # We adapt the data
    X_test = np.stack(li, axis=0)

    # Load w model
    print('Loading model...')
    model = load_model('model_smaller_vgg_7classes_w.h5')

    # Make predictions
    print('Making predictions...')
    y_predict_w = make_predictions(X_test)
    predictions_w = adapt_predictions_w(y_predict_w)

    # Load v model
    print('Loading model...')
    model = load_model('model_smaller_vgg_4classes_v.h5')

    # Make predictions
    print('Making predictions...')
    y_predict_v = make_predictions(X_test)
    predictions_v = adapt_predictions_v(y_predict_v)

    for i in range(0, len(y_predict_w)):
        print('Data corrected w: ' + str(lab_w_str[i]))
        print('Prediction w: ' + predictions_w[i])
        print('Data corrected v: ' + str(lab_v_str[i]))
        print('Prediction v: ' + predictions_v[i])
        cv2.imshow('img', li[i])
        cv2.waitKey(0)

    accuracy = calculate_accuracy(labels_w, y_predict_w)
    print('Accuracy of w: ' + str(accuracy) + '%')
    accuracy = calculate_accuracy(labels_v, y_predict_v)
    print('Accuracy of v: ' + str(accuracy) + '%')

