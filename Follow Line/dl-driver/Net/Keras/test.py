import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import keras.backend as K

from keras.models import load_model
from sklearn import metrics


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


def get_images(list_images):
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


def choose_model(type_net, type_image):
    if type_image == 'cropped':
        model_file_v = 'models/model_' + type_net + '_' + type_image + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_' + type_image + '_w.h5'
    else:
        model_file_v = 'models/model_' + type_net + '_v.h5'
        model_file_w = 'models/model_' + type_net + '_w.h5'
    return model_file_v, model_file_w


def make_predictions(data, model):
    """
    Function to make the predictions over a data set
    :param data: np.array - Images to predict
    :return: np.array - Labels of predictions
    """
    predictions = model.predict(data)
    predicted = [float(prediction[0]) for prediction in predictions]

    return np.array(predicted)


# def myAccuracy_regression(y_true, y_pred):
#     # Absolute difference between correct and predicted values
#     diff = K.abs(y_true-y_pred)
#     # Tensor with 0 for false values and 1 for true values
#     correct = K.less(diff,0.05)
#     # Sum all 1's and divide by the total
#     return K.mean(correct)



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
    type_image = raw_input('Choose the type of image you want: normal or cropped: ')
    type_net = raw_input('Choose the type of network you want: pilotnet, tinypilotnet, lstm_tinypilotnet, lstm, '
                         'deepestlstm_tinypilotnet or stacked: ')
    print('Your choice: ' + type_net + ', ' +type_image)

    # Load data
    list_images = glob.glob('../Dataset/Test/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    file = open('../Dataset/Test/test.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x_test = get_images(images, type_image)
    # We preprocess json
    y_test_v, y_test_w = parse_json(data)

    # We adapt the data
    X_test = np.stack(x_test, axis=0)
    y_test_v = np.stack(y_test_v, axis=0)
    y_test_w = np.stack(y_test_w, axis=0)

    # Get model
    model_file_v, model_file_w = choose_model(type_net, type_image)

    # Load model
    print('Loading model...')
    model_v = load_model(model_file_v)
    model_w = load_model(model_file_w)
    #model_v = load_model('models/model_pilotnet_v.h5', custom_objects={'myAccuracy_regression': myAccuracy_regression})
    #model_w = load_model('models/model_pilotnet_w.h5', custom_objects={'myAccuracy_regression': myAccuracy_regression})

    # Make predictions
    print('Making predictions...')
    y_predict_v = make_predictions(X_test, model_v)
    y_predict_w = make_predictions(X_test, model_w)

    for i in range(0, len(y_predict_w)):
        print('w: ', y_test_w[i], y_predict_w[i])
        print('v: ', y_test_v[i], y_predict_v[i])

    # Evaluation
    print('Making evaluation...')
    score_v = model_v.evaluate(X_test, y_test_v)
    score_w = model_w.evaluate(X_test, y_test_w)

    # Test loss, accuracy, mse and mae
    print('Evaluation v:')
    print('Test loss:', score_v[0])
    print('Test accuracy:', score_v[1])
    print('Test mean squared error: ', score_v[2])
    print('Test mean absolute error: ', score_v[3])

    print('Evaluation w:')
    print('Test loss:', score_w[0])
    print('Test accuracy:', score_w[1])
    print('Test mean squared error: ', score_w[2])
    print('Test mean absolute error: ', score_w[3])