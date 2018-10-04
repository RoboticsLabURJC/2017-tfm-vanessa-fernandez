import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from models.classification_model import cnn_model


def parse_json(data):
    array_class = []
    array_w = []
    # We process json
    data_parse = data.split('{"classification": ')[1:]
    for d in data_parse:
        classification = d.split(', "w":')[0]
        d_parse = d.split(', "w": ')[1]
        w = float(d_parse.split(', "v":')[0])
        array_class.append(classification)
        array_w.append(w)

    return array_class, array_w


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
        array_imgs.append(img)

    return array_imgs


def remove_values_aprox_zero(list_imgs, list_data, list_w):
    index = [i for i,x in enumerate(list_w) if np.isclose([x], [0.0], atol=0.08)[0] == True]
    for i in range(len(index)-1, 0, -1):
        list_data.pop(index[i])
        list_imgs.pop(index[i])
    return list_imgs, list_data


def adapt_array(array):
    new_array = []
    num_array = 100
    for i in range(0, num_array):
        if i == 0:
            array_split = array[:len(array)/num_array]
        elif i == num_array - 1:
            array_split = array[i*len(array)/num_array:]
        else:
            array_split = array[i*len(array)/num_array:len(array)*(i+1)/num_array]
        new_array.append(array_split)
    return new_array


if __name__ == "__main__":

    # Load data
    list_images = glob.glob('../Dataset/Train/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[4].split('.png')[0]))

    file = open('../Dataset/Train/train.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x = get_images(images)
    # We preprocess json
    y, array_w = parse_json(data)

    # We delete values close to zero
    x_train, y_train = remove_values_aprox_zero(x, y, array_w)

    # Split data into 80% for train and 20% for validation
    X_train, X_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    # Split a array in a array with small arrays
    array_x_train = adapt_array(X_train)
    array_y_train = adapt_array(y_train)
    array_x_validation = adapt_array(X_validation)
    array_y_validation = adapt_array(y_validation)

    # Variables
    batch_size = 128
    num_classes = 2
    nb_epochs = 12
    img_shape = (240, 320, 3)


    # Get model
    model = cnn_model(num_classes, img_shape)

    for i in range(0, len(array_x_train)):
        # We adapt the data
        X_train = np.stack(array_x_train[i], axis=0)
        y_train = np.stack(array_y_train[i], axis=0)
        X_validation = np.stack(array_x_validation[i], axis=0)
        y_validation = np.stack(array_y_validation[i], axis=0)

        #  We train
        model_history = model.fit(X_train, y_train, epochs=nb_epochs, validation_data=(X_validation, y_validation),
                                  batch_size=batch_size)

        # We evaluate the model
        score = model.evaluate(X_validation, y_validation, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    # We save the model
    model.save('models/model_classification.h5')


    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(model_history.history['loss'], 'r', linewidth=3.0)
    plt.plot(model_history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(model_history.history['acc'], 'r', linewidth=3.0)
    plt.plot(model_history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)