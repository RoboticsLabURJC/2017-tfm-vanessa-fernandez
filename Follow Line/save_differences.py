import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_mse


def compare_images(list_images_dataset, img_driving):
	# We initialize variables
	init = True

	for name in list_images_dataset:
		img = cv2.imread(name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# We compare imagess
		ssim = compare_ssim(img, img_driving, multichannel=True, full=True)
		mse = compare_mse(img, img_driving)

		if init:
			init = False
			img_dataset = img
			min_ssim = ssim
			max_ssim = ssim
			min_mse = mse
			max_mse = mse
			name_max_mse = name
			name_min_ssim = name
			name_min_mse = name
			name_max_ssim = name
		else:
			print(name)
			print(ssim[0], min_ssim[0], max_ssim[0], mse, max_mse, min_mse)
			if mse > max_mse:
				max_mse = mse
				name_max_mse = name
			if mse < min_mse:
				min_mse = mse
				name_min_mse = name
			if ssim[0] > max_ssim[0]:
				max_ssim = ssim
				name_max_ssim = name
			if ssim[0] < min_ssim[0]:
				min_ssim = ssim
				name_min_ssim = name

	print(name_max_mse, name_min_mse)
	print(name_min_ssim, name_max_ssim)


if __name__ == "__main__":
	list_images_driving = glob.glob('Failed_driving/Images/' + '*')
	images_driving = sorted(list_images_driving, key=lambda x: int(x.split('/')[2].split('.png')[0]))
	list_images_dataset = glob.glob('Dataset/Train/Images/' + '*')
	images_dataset = sorted(list_images_dataset, key=lambda x: int(x.split('/')[3].split('.png')[0]))

	for name in images_driving:
		img_driving = cv2.imread(name)
		img_driving = cv2.cvtColor(img_driving, cv2.COLOR_BGR2RGB)
		compare_images(images_dataset, img_driving)
		i += 1
