import os
import cv2
import math
import numpy as np

from random import randint

if __name__ == '__main__':
	# We check if dataset exists
	foldername = 'Dataset'
	if not os.path.exists(foldername): os.makedirs(foldername)
	folder_images = foldername + '/' + 'Images'
	if not os.path.exists(folder_images): os.makedirs(folder_images)
	filename = foldername + '/' + 'angles.txt'
	file = open(filename,"w")

	# Open background image
	background = cv2.imread('background.png')
	
	# Variables
	number_images = 200
	width1 = 4
	width2 = 80
	limit_p1 = [144, 121]
	limit_p2 = [180-width1, 121]
	limit_p3 = [0, 239]
	limit_p4 = [320-width2, 239]

	# We generate the dataset if it doesn't exist. If the dataset exists, we update values
	for i in range(0, number_images):
		# We update the img
		img = np.copy(background)

		# Random points
		pt1 = [randint(limit_p1[0], limit_p2[0]), limit_p1[1]]
		pt2 = [pt1[0]+width1, limit_p2[1]]
		pt3 = [randint(limit_p3[0], limit_p4[0]), limit_p3[1]]
		pt4 = [pt3[0]+width2, limit_p4[1]]

		# We draw polygone on image
		pts = np.array([pt1, pt2, pt4, pt3], np.int32)
		pts = pts.reshape((-1,1,2))
		img = cv2.fillPoly(img, [pts], (255,0,0))

		# We save the image
		number = i + 1
		name_image = folder_images + '/' + str(number) + '.png'

		cv2.line(img,((pt1[0] + pt2[0])/2, pt1[1]),((pt3[0] + pt4[0])/2, pt3[1]),(0,0,255),1)
		cv2.imwrite(name_image,img)

		# We calculate the line's angle
		pmiddle_up = [float((pt1[0] + pt2[0])/2), pt1[1]]
		pmiddle_down = [float((pt3[0] + pt4[0])/2), pt3[1]]
		angle = math.atan(float((pmiddle_up[1] - pmiddle_down[1]) / (pmiddle_up[0] - pmiddle_down[0])))

		degrees = -angle * 180 / math.pi	

		# We write on txt
		text = str(degrees) + '\n'
		file.write(text)

	# We close txt file
	file.close()
