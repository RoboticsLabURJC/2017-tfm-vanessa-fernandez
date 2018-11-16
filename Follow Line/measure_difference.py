import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

from skimage.measure import compare_ssim, compare_mse
from scipy.spatial import distance

#https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c
#https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
#https://ece.uwaterloo.ca/~z70wang/research/ssim/
#https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py
#https://samvankooten.net/2018/09/25/earth-movers-distance-in-python/
#http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
#https://yiyibooks.cn/sorakunnn/scipy-1.0.0/scipy-1.0.0/generated/scipy.spatial.distance.cdist.html
# https://stackoverrun.com/es/q/4524578
# https://es.wikipedia.org/wiki/Entrop%C3%ADa_(informaci%C3%B3n)
# https://relopezbriega.github.io/blog/2018/03/30/introduccion-a-la-teoria-de-la-informacion-con-python/
# https://gist.github.com/iamaziz/02491e36490eb05a30f8
# https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy



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
	return name_max_mse, name_min_mse, mean_mse, max_mse, min_mse, name_max_ssim, name_min_ssim, mean_ssim, max_ssim, min_ssim


def show_comparison (img_dataset, img_driving, ssim, mse, mean_mse, mean_ssim):
	fig = plt.figure('Comparison between image')
	plt.suptitle("MSE: %.2f, SSIM: %.2f, Mean MSE: %.2f, Mean SSIM: %.2f" % (mse, ssim[0], mean_mse, mean_ssim))

	ax = fig.add_subplot(1, 3, 1)
	ax.set_title('Dataset image')
	plt.imshow(img_dataset)
	plt.axis("off")
 
	ax = fig.add_subplot(1, 3, 2)
	ax.set_title('Failed driving image')
	plt.imshow(img_driving)
	plt.axis("off")

	ax = fig.add_subplot(1, 3, 3)
	ax.set_title('SSIM image')
	plt.imshow(ssim[1])
	plt.axis("off")

	plt.show()


if __name__ == "__main__":
	img_driving = cv2.imread('Failed_driving/Images/49.png')

	list_images_dataset = glob.glob('Dataset1/Train/Images/' + '*')
	images_dataset = sorted(list_images_dataset, key=lambda x: int(x.split('/')[3].split('.png')[0]))

	img_driving = cv2.cvtColor(img_driving, cv2.COLOR_BGR2RGB)

	name_max_mse, name_min_mse, mean_mse, max_mse, min_mse, name_max_ssim, name_min_ssim, mean_ssim, max_ssim, min_ssim = compare_images(images_dataset, img_driving)

	img_max_ssim = cv2.imread(name_max_ssim)
	img_max_ssim = cv2.cvtColor(img_max_ssim, cv2.COLOR_BGR2RGB)

	img_min_ssim = cv2.imread(name_min_ssim)
	img_min_ssim = cv2.cvtColor(img_min_ssim, cv2.COLOR_BGR2RGB)

	show_comparison (img_max_ssim, img_driving, max_ssim, min_mse, mean_mse, mean_ssim)
	show_comparison (img_min_ssim, img_driving, min_ssim, max_mse, mean_mse, mean_ssim)

	print(name_max_mse, name_min_mse, mean_mse, max_mse, min_mse, name_max_ssim, name_min_ssim, mean_ssim, max_ssim, min_ssim)

	#ssim = compare_ssim(img_dataset, img_driving, multichannel=True, full=True)
	#mse = compare_mse(img_dataset, img_driving)

	#show_comparison (img_dataset, img_driving, ssim)
