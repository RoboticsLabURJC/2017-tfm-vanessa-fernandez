#!/usr/bin/python
#-*- coding: utf-8 -*-
import threading
import time
from datetime import datetime

import math
import jderobot
import cv2
import numpy as np

time_cycle = 80

class MyAlgorithm(threading.Thread):

    def __init__(self, camera, motors, network):
        self.camera = camera
        self.motors = motors
        self.network = network
        self.threshold_image = np.zeros((640,360,3), np.uint8)
        self.color_image = np.zeros((640,360,3), np.uint8)
        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        self.threshold_image_lock = threading.Lock()

        self.color_image_lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)
    
    def getImage(self):
        self.lock.acquire()
        img = self.camera.getImage().data
        self.lock.release()
        return img

    def set_color_image (self, image):
        img  = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.color_image_lock.acquire()
        self.color_image = img
        self.color_image_lock.release()
        
    def get_color_image (self):
        self.color_image_lock.acquire()
        img = np.copy(self.color_image)
        self.color_image_lock.release()
        return img
        
    def set_threshold_image (self, image):
        img = np.copy(image)
        if len(img.shape) == 2:
          img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        self.threshold_image_lock.acquire()
        self.threshold_image = img
        self.threshold_image_lock.release()
        
    def get_threshold_image (self):
        self.threshold_image_lock.acquire()
        img  = np.copy(self.threshold_image)
        self.threshold_image_lock.release()
        return img

    def run (self):

        while (not self.kill_event.is_set()):
            start_time = datetime.now()
            if not self.stop_event.is_set():
                self.algorithm()
            finish_Time = datetime.now()
            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def algorithm(self):
        #GETTING THE IMAGES
        image = self.getImage()

        # Add your code here
        print("Runing")

        prediction_v = self.network.prediction_v
        prediction_w = self.network.prediction_w

        #EXAMPLE OF HOW TO SEND INFORMATION TO THE ROBOT ACTUATORS
        #self.motors.sendV(10)
        #self.motors.sendW(5)

        # REGRESSION NETWORK FOR W AND V
        self.motors.sendV(prediction_v)
        self.motors.sendW(prediction_w)

        # REGRESSION NETWORK FOR W AND CONSTANT V
        # self.motors.sendV(3)
        # self.motors.sendW(prediction_w)


        # CLASSIFICATION NETWORK 7 CLASSES FOR W AND CONSTANT V
        # self.motors.sendV(5)
        #
        # if prediction_w == "radically_left":
        #     self.motors.sendW(1.8)
        # elif prediction_w == "moderately_left":
        #     self.motors.sendW(0.75)
        # elif prediction_w == "slightly_left":
        #     self.motors.sendW(0.25)
        # elif prediction_w == "slight":
        #     self.motors.sendW(0)
        # elif prediction_w == "slightly_right":
        #     self.motors.sendW(-0.25)
        # elif prediction_w == "moderately_right":
        #     self.motors.sendW(-0.75)
        # elif prediction_w == "radically_right":
        #     self.motors.sendW(-1.8)



        # CLASSIFICATION NETWORK 7 CLASSES FOR W AND 4 FOR V
        # if prediction_v == "slow":
        #     self.motors.sendV(5)
        # elif prediction_v == "moderate":
        #     #self.motors.sendV(6)
        #     self.motors.sendV(8)
        # elif prediction_v == "fast":
        #     #self.motors.sendV(7)
        #     self.motors.sendV(10)
        # elif prediction_v == "very_fast":
        #     #self.motors.sendV(8)
        #     self.motors.sendV(13)
        #
        # if prediction_w == "radically_left":
        #     self.motors.sendW(1.9)
        # elif prediction_w == "moderately_left":
        #     self.motors.sendW(0.75)
        #     #self.motors.sendW(0.75)
        # elif prediction_w == "slightly_left":
        #     self.motors.sendW(0.25)
        #     #self.motors.sendW(0.5)
        # elif prediction_w == "slight":
        #     self.motors.sendW(0)
        # elif prediction_w == "slightly_right":
        #     self.motors.sendW(-0.25)
        #     #self.motors.sendW(-0.5)
        # elif prediction_w == "moderately_right":
        #     self.motors.sendW(-0.75)
        #     #self.motors.sendW(-0.75)
        # elif prediction_w == "radically_right":
        #     self.motors.sendW(-1.9)

        # CLASSIFICATION NETWORK 9 CLASSES FOR W AND 4 FOR V
        # if prediction_v == "slow":
        #     self.motors.sendV(5)
        # elif prediction_v == "moderate":
        #     self.motors.sendV(8)
        # elif prediction_v == "fast":
        #     self.motors.sendV(10)
        # elif prediction_v == "very_fast":
        #     self.motors.sendV(13)
        #
        # if prediction_w == "radically_left":
        #     self.motors.sendW(2.4)
        # elif prediction_w == "strongly_left":
        #     self.motors.sendW(1.7)
        # elif prediction_w == "moderately_left":
        #     self.motors.sendW(0.75)
        # elif prediction_w == "slightly_left":
        #     self.motors.sendW(0.25)
        # elif prediction_w == "slight":
        #     self.motors.sendW(0)
        # elif prediction_w == "slightly_right":
        #     self.motors.sendW(-0.25)
        # elif prediction_w == "moderately_right":
        #     self.motors.sendW(-0.75)
        # elif prediction_w == "strongly_right":
        #     self.motors.sendW(-1.7)
        # elif prediction_w == "radically_right":
        #     self.motors.sendW(-2.4)



        # CLASSIFICATION NETWORK 7 CLASSES FOR W AND 5 FOR V
        # if prediction_v == "slow":
        #     self.motors.sendV(5)
        # elif prediction_v == "moderate":
        #     self.motors.sendV(8)
        # elif prediction_v == "fast":
        #     self.motors.sendV(10)
        # elif prediction_v == "very_fast":
        #     self.motors.sendV(13)
        # elif prediction_v == 'negative':
        #     self.motors.sendV(-0.6)
        #
        # if prediction_w == "radically_left":
        #     self.motors.sendW(1.7)
        # elif prediction_w == "moderately_left":
        #     self.motors.sendW(0.75)
        # elif prediction_w == "slightly_left":
        #     self.motors.sendW(0.25)
        # elif prediction_w == "slight":
        #     self.motors.sendW(0)
        # elif prediction_w == "slightly_right":
        #     self.motors.sendW(-0.25)
        # elif prediction_w == "moderately_right":
        #     self.motors.sendW(-0.75)
        # elif prediction_w == "radically_right":
        #     self.motors.sendW(-1.7)


        #SHOW THE FILTERED IMAGE ON THE GUI
        self.set_threshold_image(image)