import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_mse

#https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c
#https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
#https://ece.uwaterloo.ca/~z70wang/research/ssim/
#https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py


def show_comparison (img_dataset, img_driving, ssim):
	fig = plt.figure('Comparison between image')
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (mse, ssim[0]))

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
	#img_driving = cv2.imread('Failed_driving/Images/1.png')
	#img_dataset = cv2.imread('Dataset/Train/Images/1.png')

	#img_dataset = cv2.cvtColor(img_dataset, cv2.COLOR_BGR2RGB)

	#ssim = compare_ssim(img_dataset, img_driving, multichannel=True, full=True)
	#mse = compare_mse(img_dataset, img_driving)

	#show_comparison (img_dataset, img_driving, ssim)
