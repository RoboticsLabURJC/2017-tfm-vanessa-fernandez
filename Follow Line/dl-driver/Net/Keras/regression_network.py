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

        # Stack frames (stacked method)
        self.stacked_imgs = []
        self.margin = 2
        self.num_stacked_imgs = 3

    def setCamera(self, camera):
        self.camera = camera


    def predict(self):
        input_image = self.camera.getImage()

        # Preprocessing
        img = cv2.cvtColor(input_image.data, cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img, (self.img_width, self.img_height))

        # Stack frames
        # if len(self.stacked_imgs) == 0:
        #     for i in range(0, (self.num_stacked_imgs*2+1)):
        #         self.stacked_imgs.append(img_resized)
        # else:
        #     for i in range(0, len(self.stacked_imgs)-1):
        #        self.stacked_imgs[i] = self.stacked_imgs[i+1]
        #     self.stacked_imgs[len(self.stacked_imgs)-1] = img_resized
        # im1 = np.concatenate([self.stacked_imgs[0], self.stacked_imgs[self.margin+1]], axis=2)
        # img_resized = np.concatenate([im1, self.stacked_imgs[(self.margin+1)*2]], axis=2)

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
