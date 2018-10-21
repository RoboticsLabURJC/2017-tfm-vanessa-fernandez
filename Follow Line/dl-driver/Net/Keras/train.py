import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils import plot_model
from models.model_nvidia import pilotnet_model


def parse_json(data):
    array = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v": ')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split(('"w": '))[1]
        #array.append([float(v), float(w)])
        array.append(float(w))

    return array


def get_images(list_images):
    # We read the images
    array_imgs = []
    for name in list_images:
        img = cv2.imread(name)
        img = cv2.resize(img, (img.shape[1] / 4, img.shape[0] / 4))
        array_imgs.append(img)

    return array_imgs


# def adapt_array(array):
#     new_array = []
#     num_array = 300
#     for i in range(0, num_array):
#         if i == 0:
#             array_split = array[:len(array)/num_array]
#         elif i == num_array - 1:
#             array_split = array[i*len(array)/num_array:]
#         else:
#             array_split = array[i*len(array)/num_array:len(array)*(i+1)/num_array]
#         new_array.append(array_split)
#     return new_array


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
    y = parse_json(data)

    # Split data into 80% for train and 20% for validation
    X_train, X_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=42)

    # # Split a array in a array with small arrays
    # array_x_train = adapt_array(X_train)
    # array_y_train = adapt_array(y_train)
    # array_x_test = adapt_array(X_validation)
    # array_y_test = adapt_array(y_validation)

    # Variables
    batch_size = 16
    nb_epoch = 2000
    img_shape = (120, 160, 3)

    # Get model
    model = pilotnet_model(img_shape)

    X_train = np.stack(X_train, axis=0)
    y_train = np.stack(y_train, axis=0)
    X_validation = np.stack(X_validation, axis=0)
    y_validation = np.stack(y_validation, axis=0)

    # Print layers
    print(model.summary())
    # Plot layers of model
    plot_model(model, to_file=model_png)

    #  We train
    model_history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, verbose=2,
                              validation_data=(X_validation, y_validation))

    # We evaluate the model
    score = model.evaluate(X_validation, y_validation, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # for i in range(0, len(array_x_train)):
    #     # We adapt the data
    #     X_train = np.stack(array_x_train[i], axis=0)
    #     y_train = np.stack(array_y_train[i], axis=0)
    #     X_test = np.stack(array_x_test[i], axis=0)
    #     y_test = np.stack(array_y_test[i], axis=0)
    #
    #     # We train
    #     model_history = model.fit(X_train, y_train, epochs=nb_epoch, validation_split=0.2, batch_size=batch_size)
    #
    #     # We evaluate the model
    #     score = model.evaluate(X_test, y_test, verbose=0)
    #     print('Test loss:', score[0])
    #     print('Test accuracy:', score[1])

    # We save the model
    model.save('models/model_nvidia.h5')

    # Plot the training and validation loss for each epoch
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('mse')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.ylim([0, 0.1])
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
