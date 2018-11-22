#
#  Authors :
#       Vanessa Fernandez Martinez <vanessa_1895@msn.com>

import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model


class RegressionNetwork():
    def __init__(self, net_model):
        # Load models
        self.model_file_v = 'Net/Keras/' + net_model['Model_Regression_v']
        self.model_file_w = 'Net/Keras/' + net_model['Model_Regression_w']
        self.model_v = load_model(self.model_file_v)
        self.model_w = load_model(self.model_file_w)

        # Obtain the graph
        self.graph = tf.get_default_graph()

        # The Keras network works on 160x120
        self.img_height = 120
        self.img_width = 160

        self.prediction_v = ""
        self.prediction_w = ""


    def setCamera(self, camera):
        self.camera = camera


    def predict(self):
        input_image = self.camera.getImage()

        # Preprocessing
        img = cv2.cvtColor(input_image.data, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img, (self.img_width, self.img_height))

        # We adapt the image
        input_img = np.stack([img_resized], axis=0)

        # While predicting, use the same graph
        with self.graph.as_default():
            # Make prediction
            predictions_v = self.model_v.predict(input_img)
            predictions_w = self.model_w.predict(input_img)
        y_pred_v = [float(prediction[0]) for prediction in predictions_v][0]
        y_pred_w = [float(prediction[0]) for prediction in predictions_w][0]

        self.prediction_v = y_pred_v
        self.prediction_w = y_pred_w
