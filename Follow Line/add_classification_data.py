# This file modify json data if it hasn't the classification data


def get_classification(data):
	v = data.split('"v":')[1]
	d_parse = data.split(', "v":')[0]
	w = d_parse.split((': '))[1]
	if w[0] == '-':
		classification = 'right'
	else:
		classification = 'left'
	return classification


if __name__ == "__main__":
	filename = 'Dataset/data.json'
	output = ''

	with open(filename, "r+") as f:
		data = f.read()
		data_parse = data.split('{')[1:]

		for d in data_parse:
			classification = get_classification(d)
			output = output + '{"classification": ' + classification + ', ' + d

		f.seek(0)
		f.write(output)
		f.truncate()

