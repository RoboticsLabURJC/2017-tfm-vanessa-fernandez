import cv2
import glob
import numpy as np
from skimage.measure import compare_ssim, compare_mse


def compare_images(list_images_dataset, img_driving):
	# We initialize variables
	init = True
	sum_ssim = 0
	sum_mse = 0

	for name in list_images_dataset:
		img = cv2.imread(name)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		# We compare imagess
		ssim = compare_ssim(img, img_driving, multichannel=True, full=True)
		mse = compare_mse(img, img_driving)

		sum_ssim += ssim[0]
		sum_mse += mse

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

	mean_mse = sum_mse / len(list_images_dataset)
	mean_ssim = sum_ssim / len(list_images_dataset)
	print(name_max_mse, name_min_mse, mean_mse)
	print(name_min_ssim, name_max_ssim, mean_ssim)
	return name_max_mse, name_min_mse, mean_mse, name_max_ssim, name_min_ssim, mean_ssim


if __name__ == "__main__":
	list_images_driving = glob.glob('Failed_driving/Images/' + '*')
	images_driving = sorted(list_images_driving, key=lambda x: int(x.split('/')[2].split('.png')[0]))
	list_images_dataset = glob.glob('Dataset/Train/Images/' + '*')
	images_dataset = sorted(list_images_dataset, key=lambda x: int(x.split('/')[3].split('.png')[0]))

	#file_txt = open('Failed_driving/measures.txt', 'a')
	#file_txt.write('Image_max_mse Image_min_mse Mean_mse Image_max_ssim Image_min_ssim Mean_ssim\n')
	#file_txt.close()

	for i in range(64, len(images_driving)):
		img_driving = cv2.imread(images_driving[i])
		img_driving = cv2.cvtColor(img_driving, cv2.COLOR_BGR2RGB)
		name_max_mse, name_min_mse, mean_mse, name_max_ssim, name_min_ssim, mean_ssim = compare_images(images_dataset, img_driving)

		file_txt = open('Failed_driving/measures.txt', 'a')
		file_txt.write(name_max_mse + ' ' + name_min_mse + ' ' + str(mean_mse) + ' ' + name_max_ssim + ' ' + name_min_ssim + ' ' + str(mean_ssim) + '\n')
		file_txt.close()

