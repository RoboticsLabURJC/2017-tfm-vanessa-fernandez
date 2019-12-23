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
        # If we have cropped images, the network works on 160x65
        #self.img_height = 65
        self.img_width = 160

        self.prediction_v = 0
        self.prediction_w = 0

        # Stack frames (stacked method)
        self.stacked_imgs = []
        self.margin = 10
        self.num_stacked_imgs = 2

        # Controlnet frames
        #self.num_frames = 5
        #self.frames = []
        self.num_frames_w = 5
        self.num_frames_v = 30
        self.frames_w = []
        self.frames_v = []

    def setCamera(self, camera):
        self.camera = camera

    def normalize_image(self, array):
        rng = np.amax(array) - np.amin(array)
        if rng == 0:
            rng = 1
        amin = np.amin(array)
        return (array - amin) * 255.0 / rng


    def predict(self):
        input_image = self.camera.getImage()

        # Preprocessing
        img = cv2.cvtColor(input_image.data, cv2.COLOR_RGB2BGR)
        #img= cv2.cvtColor(input_image.data[220:480, 0:640], cv2.COLOR_RGB2BGR)
        if img is not None:
            img_resized = cv2.resize(img, (self.img_width, self.img_height))

            # Controlnet
            # if len(self.frames) == 0:
            #     for i in range(0, self.num_frames):
            #         self.frames.append(img_resized)
            # else:
            #     for i in range(0, self.num_frames-1):
            #         self.frames[i] = self.frames[i+1]
            #     self.frames[self.num_frames-1] = img_resized

            #img_resized = self.frames

            # if len(self.frames_w) == 0:
            #     for i in range(0, self.num_frames_w):
            #         self.frames_w.append(img_resized)
            #
            #     for i in range(0, self.num_frames_v):
            #         self.frames_v.append(img_resized)
            # else:
            #     for i in range(0, self.num_frames_w-1):
            #         self.frames_w[i] = self.frames_w[i+1]
            #     self.frames_w[self.num_frames_w-1] = img_resized
            #
            #     for i in range(0, self.num_frames_v - 1):
            #         self.frames_v[i] = self.frames_v[i + 1]
            #     self.frames_v[self.num_frames_v - 1] = img_resized

            # Stack frames or temporal frames
            # if len(self.stacked_imgs) == 0:
            #     for i in range(0, (self.num_stacked_imgs + self.margin * (self.num_stacked_imgs-1))):
            #         self.stacked_imgs.append(img_resized)
            # else:
            #     for i in range(0, len(self.stacked_imgs)-1):
            #        self.stacked_imgs[i] = self.stacked_imgs[i+1]
            #        self.stacked_imgs[len(self.stacked_imgs)-1] = img_resized

            ##im1 = np.concatenate([self.stacked_imgs[0], self.stacked_imgs[self.margin+1]], axis=2)
            ##img_resized = np.concatenate([im1, self.stacked_imgs[(self.margin+1)*2]], axis=2)

            # img_resized = np.concatenate([self.stacked_imgs[0], self.stacked_imgs[self.margin + 1]], axis=2)
            # im = self.stacked_imgs[self.margin + 1] - self.stacked_imgs[0]
            # img_resized = np.concatenate([im, self.stacked_imgs[self.margin + 1]], axis=2)

            # i1 = cv2.cvtColor(self.stacked_imgs[self.margin + 1], cv2.COLOR_BGR2HSV)
            # i2 = cv2.cvtColor(self.stacked_imgs[0], cv2.COLOR_BGR2HSV)
            # dif = np.zeros((i1.shape[0], i1.shape[1], 2))
            # dif[:, :, 0] = cv2.absdiff(i1[:, :, 0], i2[:, :, 0])
            # dif[:, :, 1] = cv2.absdiff(i1[:, :, 1], i2[:, :, 1])
            #img_resized = self.normalize_image(dif)

            # f1 = np.power(dif[:, :, 0],2)
            # f2 = np.power(dif[:, :, 1],2)
            # f = np.sqrt(f1+f2)
            # cv2.imshow('h', f)

            # i1 = cv2.cvtColor(self.stacked_imgs[self.margin + 1], cv2.COLOR_BGR2GRAY)
            # i2 = cv2.cvtColor(self.stacked_imgs[0], cv2.COLOR_BGR2GRAY)
            # dif = np.zeros((i1.shape[0], i1.shape[1], 1))
            # dif[:, :, 0] = cv2.subtract(i1, i2)
            # img_resized = dif
            # cv2.imshow('im', dif)
            # img_resized = np.add(self.stacked_imgs[self.margin + 1], self.stacked_imgs[0])
            # cv2.imshow('mas channel', np.add(C, self.stacked_imgs[0]))


            # i1 = cv2.cvtColor(self.stacked_imgs[self.margin + 1], cv2.COLOR_BGR2GRAY)
            # i2 = cv2.cvtColor(self.stacked_imgs[0], cv2.COLOR_BGR2GRAY)
            # i1 = cv2.GaussianBlur(i1, (5, 5), 0)
            # i2 = cv2.GaussianBlur(i2, (5, 5), 0)
            # difference = np.zeros((i1.shape[0], i1.shape[1], 1))
            # difference[:, :, 0] = cv2.absdiff(i1, i2)
            # _, difference[:, :, 0] = cv2.threshold(difference[:, :, 0], 15, 255, cv2.THRESH_BINARY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # difference[:, :, 0] = cv2.morphologyEx(difference[:, :, 0], cv2.MORPH_CLOSE, kernel)
            # img_resized = difference
            # cv2.imshow('dif', difference)

            # i1 = cv2.cvtColor(self.stacked_imgs[self.margin + 1], cv2.COLOR_BGR2GRAY)
            # i2 = cv2.cvtColor(self.stacked_imgs[0], cv2.COLOR_BGR2GRAY)
            # i1 = cv2.GaussianBlur(i1, (5, 5), 0)
            # i2 = cv2.GaussianBlur(i2, (5, 5), 0)
            # difference = np.zeros((i1.shape[0], i1.shape[1], 1))
            # difference[:, :, 0] = cv2.subtract(np.float64(i1), np.float64(i2))
            # mask1 = cv2.inRange(difference[:, :, 0], 15, 255)
            # mask2 = cv2.inRange(difference[:, :, 0], -255, -15)
            # mask = mask1 + mask2
            # difference[:, :, 0][np.where(mask == 0)] = 0
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # difference[:, :, 0] = cv2.morphologyEx(difference[:, :, 0], cv2.MORPH_CLOSE, kernel)
            # im2 = difference
            # if np.ptp(im2) != 0:
            #     img_resized = 256 * (im2 - np.min(im2)) / np.ptp(im2) - 128
            # else:
            #     img_resized = 256 * (im2 - np.min(im2)) / 1 - 128
            #cv2.imshow('dif', np.uint8(img_resized))

            # We adapt the image
            input_img = np.stack([img_resized], axis=0)
            # input_img_w = np.stack([self.frames_w], axis=0)
            # input_img_v = np.stack([self.frames_v], axis=0)

            # While predicting, use the same graph
            with self.graph.as_default():
                # Make prediction
                predictions_v = self.model_v.predict(input_img)
                predictions_w = self.model_w.predict(input_img)
                # predictions_v = self.model_v.predict(input_img_v)
                # predictions_w = self.model_w.predict(input_img_w)

            y_pred_v = [float(prediction[0]) for prediction in predictions_v][0]
            y_pred_w = [float(prediction[0]) for prediction in predictions_w][0]

            if y_pred_v and y_pred_w:
                self.prediction_v = y_pred_v
                self.prediction_w = y_pred_w
