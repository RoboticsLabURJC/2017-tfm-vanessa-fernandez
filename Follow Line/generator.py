import os
import glob
import json
import cv2

def create_dataset():
	# We check if dataset exists
	foldername = 'Dataset'
	if not os.path.exists(foldername): os.makedirs(foldername)
	folder_images = foldername + '/' + 'Images'
	if not os.path.exists(folder_images): os.makedirs(folder_images)


def save_json(v, w):
	# We save the speed data

	file_name = 'Dataset/data.json'

	data = {
    	'v': v,
    	'w': w
	}

	with open(file_name, 'a') as file:
		json.dump(data, file)


def check_empty_directory(directory):
	# We check if the directory is empty
	empty = False
	for dirName, subdirList, fileList in os.walk(directory):
		print(len(fileList))
		if len(fileList) == 0:
			empty = True

	return empty


def get_number_image(path):
	list_images = glob.glob(path + '*')
	sort_images = sorted(list_images, key=lambda x: int(x.split('/')[2].split('.png')[0]))
	last_number = sort_images[len(sort_images)-1].split('/')[2].split('.png')[0]
	number = int(last_number) + 1
	return number


def save_image(img):
	# We save images
	folder_images = 'Dataset/Images/'
	empty = check_empty_directory(folder_images)
	print(empty)
	if empty:
		number = 1
	else:
		number = get_number_image(folder_images)
	name_image = folder_images + str(number) + '.png'
	cv2.imwrite(name_image, img)

