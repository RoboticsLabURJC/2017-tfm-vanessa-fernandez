import glob

from models.model_nvidia import nvidia_model


def parse_json(data):
    array_v = []
    array_w = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v":')[1]
        array_v.append(int(v))
        d_parse = d.split(', "v":')[0]
        w = d_parse.split((': '))[1]
        array_w.append(float(w))

    return array_v, array_w


if __name__ == "__main__":

    # Load data
    list_images = glob.glob('../Dataset/Images/' + '*')
    list_images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))

    file = open('../Dataset/data.json', 'r')
    data = file.read()
    file.close()

    # We process json
    array_v, array_w = parse_json(data)

    # Variables
    batch_size = 128
    nb_epoch = 2000
    img_shape = (640, 480, 3)

    model = nvidia_model(img_shape)