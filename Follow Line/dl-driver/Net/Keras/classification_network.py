#
#  Authors :
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>

import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model


class ClassificationNetwork():
    def __init__(self, net_model):
        # Load model
        self.model_file = 'Net/Keras/' + net_model['Model']
        self.model = load_model(self.model_file)

        # Obtain the graph
        self.graph = tf.get_default_graph()

        # The Keras network works on 160x120
        self.img_height = 120
        self.img_width = 160

        self.num_classes = 7

        self.prediction = ""


    def setCamera(self, camera):
        self.camera = camera


    def convertLabel(self, label):
        if self.num_classes == 2:
            if label == 0:
                string_label = "left"
            else:
                string_label = "right"
        elif self.num_classes == 7:
            if label == 0:
                string_label = "radically_left"
            elif label == 1:
                string_label = "moderately_left"
            elif label == 2:
                string_label = "slightly_left"
            elif label == 3:
                string_label = "slight"
            elif label == 4:
                string_label = "slightly_right"
            elif label == 5:
                string_label = "moderately_right"
            elif label == 6:
                string_label = "radically_right"
        return string_label


    def predict(self):
        input_image = self.camera.getImage()

        # Preprocessing
        img_resized = cv2.resize(input_image.data, (self.img_width, self.img_height))

        # We adapt the image
        input_img = np.stack([img_resized], axis=0)

        # While predicting, use the same graph
        with self.graph.as_default():
            # Make prediction
            predictions = self.model.predict(input_img)
        y_pred = [np.argmax(prediction) for prediction in predictions][0]

        # Convert int prediction to corresponded label
        y_pred = self.convertLabel(y_pred)

        self.prediction = y_pred
