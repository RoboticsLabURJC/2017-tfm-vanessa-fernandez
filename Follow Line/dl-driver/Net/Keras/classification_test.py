import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn import metrics


def parse_json(data):
    array_class = []
    # We process json
    data_parse = data.split('{"classification": ')[1:]
    for d in data_parse:
        classification = d.split(', "w":')[0]
        array_class.append(classification)

    return array_class


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
        array_imgs.append(img)

    return array_imgs


def adapt_labels(array_labels):
    for i in range(0, len(array_labels)):
        if array_labels[i] == '"left"':
            array_labels[i] = 0
        else:
            array_labels[i] = 1

    return array_labels


def make_predictions(data):
    """
    Function to make the predictions over a data set
    :param data: np.array - Images to predict
    :return: np.array - Labels of predictions
    """
    predictions = model.predict(data)
    predicted = [int(prediction[0]) for prediction in predictions]

    return np.array(predicted)


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

    # Load data
    list_images = glob.glob('../Dataset/Test/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    file = open('../Dataset/Test/test.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x_test = get_images(images)
    # We preprocess json
    y_test = parse_json(data)

    # We adapt string labels to int labels
    y_test = adapt_labels(y_test)

    # We adapt the data
    X_test = np.stack(x_test, axis=0)
    y_test = np.stack(y_test, axis=0)

    # Load model
    print('Loading model...')
    model = load_model('models/model_classification.h5')

    # Make predictions
    print('Making predictions...')
    y_predict = make_predictions(X_test)

    # Evaluation
    print('Making evaluation...')
    score = model.evaluate(X_test, y_test)

    evaluation = metrics.classification_report(y_test, y_predict)

    # Test loss and accuracy
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Precision, recall, F1 score for each class
    print("Evaluation's metrics: ")
    print(evaluation)

    # Confusion matrix
    conf_matrix = metrics.confusion_matrix(y_test, y_predict)
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix)