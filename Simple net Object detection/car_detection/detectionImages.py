import cv2
import keras
import glob
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
import tensorflow as tf

from ssd import SSD300
from ssd_utils import BBoxUtility


# Function definition to apply vehicle detection on image
def img_detect_vehicle(img):
    dst = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
    inputs = preprocess_input(np.array([image.img_to_array(dst).copy()]))
    preds = model.predict(inputs, batch_size=1, verbose=1)
    results = bbox_util.detection_out(preds)

    # Parse the outputs.
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]
    
    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6 and det_label[i]==7]

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
        cv2.rectangle(img,(xmin, ymin),(xmax, ymax),(255, 0, 0), 4)

    return img


# Run pipeline on 4 test images
if __name__ == '__main__':
	voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
	NUM_CLASSES = len(voc_classes) + 1

	input_shape=(300, 300, 3)
	model = SSD300(input_shape, num_classes=NUM_CLASSES)
	model.load_weights('weights_SSD300.hdf5', by_name=True)
	bbox_util = BBoxUtility(NUM_CLASSES)

	listImg = glob.glob('test_images'+"/*")

	plt.figure(figsize=(30,10))

	for i in range(0,len(listImg)):
		img = cv2.imread(listImg[i])
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		res = img_detect_vehicle(img)
		plt.subplot(2,3,i+1)
		plt.imshow(res)

	plt.show()
