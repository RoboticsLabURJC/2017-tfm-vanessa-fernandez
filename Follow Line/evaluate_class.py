import matplotlib.pyplot as plt
import numpy as np


def parse_dataset_json(data):
	array_class = []
    # We process json
	data_parse = data.split('"class2": ')[1:]
	for d in data_parse:
		classification = d.split(', "classification":')[0]
		array_class.append(classification)

	return array_class


def parse_driving_json(data):
	array_class = []
    # We process json
	data_parse = data.split('"w": ')[1:]
	for d in data_parse:
		w = float(d.split(', "v":')[0])
		if w == 1.7:
			classification = 'radically_left'
		elif w == 0.75:
			classification = 'moderately_left'
		elif w == 0.25:
			classification = 'slightly_left'
		elif w == 0:
			classification = 'slight'
		elif w == -0.25:
			classification = 'slightly_right'
		elif w == -0.75:
			classification = 'moderately_right'
		elif w == -1.7:
			classification = 'radically_right'
		array_class.append(classification)

	return array_class


def count_classes(array_data):
	# Array with number of data for each class
	# [num_radically_left, num_moderately_left, num_slightly_left, num_slight, 
	# num_slightly_right, num_moderately_rigth, num_radically_right]
	array_num_classes = np.zeros(7)
	for class_w in array_data:
		if class_w == 'radically_left':
			array_num_classes[0] = array_num_classes[0] + 1
		elif class_w == 'moderately_left':
			array_num_classes[1] = array_num_classes[1] + 1
		elif class_w == 'slightly_left':
			array_num_classes[2] = array_num_classes[2] + 1
		elif class_w == 'slight':
			array_num_classes[3] = array_num_classes[3] + 1
		elif class_w == 'slightly_right':
			array_num_classes[4] = array_num_classes[4] + 1
		elif class_w == 'moderately_right':
			array_num_classes[5] = array_num_classes[5] + 1
		elif class_w == 'radically_right':
			array_num_classes[6] = array_num_classes[6] + 1
	return array_num_classes


def plot_histogram(array_num):
	# The histogram of the data
	y = ['radically_left', 'moderately_left', 'slightly_left', 'slight', 'slightly_right', 'moderately_right', 'radically_right']
	fig, ax = plt.subplots() 
	# Width of the bars    
	width = 0.75
	ind = np.arange(len(array_num))
	ax.barh(ind, array_num, width, color="blue")
	ax.set_yticks(ind+width/2)
	ax.set_yticklabels(y, minor=False)
	for i, v in enumerate(array_num):
		ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')
	plt.title('Histogram of classes')
	plt.xlabel('Number of data')
	plt.ylabel('Class name')  
	plt.show()


if __name__ == "__main__":
	# Load data
	file = open('Dataset1/Train/train.json', 'r')
	data_dataset = file.read()
	file.close()

	file = open('Failed_driving/data.json', 'r')
	data_driving = file.read()
	file.close()

	# Parse data
	array_class_dataset = parse_dataset_json(data_dataset)
	array_class_driving = parse_driving_json(data_driving)

	# Count number of data for each class
	num_class_dataset = count_classes(array_class_dataset)
	num_class_driving = count_classes(array_class_driving)

	# Plot histogram
	plot_histogram(num_class_dataset)
	plot_histogram(num_class_driving)
