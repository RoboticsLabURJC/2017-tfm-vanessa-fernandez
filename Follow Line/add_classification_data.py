# This file modify json data if it hasn't the classification data


def get_w(data):
	v = data.split('"v":')[1]
	d_parse = data.split(', "v":')[0]
	w = d_parse.split((', "w": '))[1]
	return w


def get_classification(w):
	if w[0] == '-' and abs(float(w)) >= 1.0:
		classification = 'radically_right'
	elif w[0] == '-' and abs(float(w)) >= 0.5:
		classification = 'moderately_right'
	elif w[0] == '-' and abs(float(w)) >= 0.1:
		classification = 'slightly_right'
	elif abs(float(w)) >= 1.0:
		classification = 'radically_left'
	elif abs(float(w)) >= 0.5:
		classification = 'moderately_left'
	elif abs(float(w)) >= 0.1:
		classification = 'slightly_left'
	else:
		classification = 'slight'
	return classification


if __name__ == "__main__":
	filename = 'Dataset/data.json'
	output = ''

	with open(filename, "r+") as f:
		data = f.read()
		data_parse = data.split('{')[1:]

		for d in data_parse:
			w = get_w(d)
			classification = get_classification(w)
			output = output + '{"class2": ' + classification + ', ' + d

		f.seek(0)
		f.write(output)
		f.truncate()

