import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import numpy as np
import time
from scipy.misc import imread
import tensorflow as tf
import threading

from ssd import SSD300
from ssd_utils import BBoxUtility


class Detection_Network():
	''' Class to create a keras network, based on SSD detection trained on PASCAL VOC dataset.
    '''

	def __init__(self):
		self.voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                            'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                            'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                            'Sheep', 'Sofa', 'Train', 'Tvmonitor']
		NUM_CLASSES = len(self.voc_classes) + 1
		input_shape = (300, 300, 3)
		self.model = SSD300(input_shape, num_classes=NUM_CLASSES)
		self.model.load_weights('/home/vanejessi/Escritorio/Vanessa/2017-tfm-vanessa-fernandez/dl_objectdetector/Net/weights_SSD300.hdf5', by_name=True)
		self.model._make_predict_function()        
		self.graph = tf.get_default_graph()
		self.bbox_util = BBoxUtility(NUM_CLASSES)

		self.lock = threading.Lock()

		self.input_image = None
		self.output_image = None


	def detection(self, img):
		dst = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
		inputs = preprocess_input(np.array([image.img_to_array(dst).copy()]))

		with self.graph.as_default():
			preds = self.model.predict(inputs, batch_size=1, verbose=1)

		results = self.bbox_util.detection_out(preds)

		# Parse the outputs.
		det_label = results[0][:, 0]
		det_conf = results[0][:, 1]
		det_xmin = results[0][:, 2]
		det_ymin = results[0][:, 3]
		det_xmax = results[0][:, 4]
		det_ymax = results[0][:, 5]
		
		# Get detections with confidence higher than 0.6.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

		top_conf = det_conf[top_indices]
		top_label_indices = det_label[top_indices].tolist()
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]

		for i in range(top_conf.shape[0]):
		    xmin = int(round(top_xmin[i] * img.shape[1]))
		    ymin = int(round(top_ymin[i] * img.shape[0]))
		    xmax = int(round(top_xmax[i] * img.shape[1]))
		    ymax = int(round(top_ymax[i] * img.shape[0]))

		    score = top_conf[i]
		    label = int(top_label_indices[i])
		    label_name = self.voc_classes[label - 1]

		    if label_name == 'Person':
		        cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(0, 255, 0), 4)
		return img


	def predict(self):
		image_np = self.input_image
		if image_np is not None:
			image_np = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
			image_np = self.detection(image_np)
		else:
			image_np = np.zeros((360, 240), dtype=np.int32)
		return image_np


	def update(self):
		self.output_image = self.predict()

