import glob
import numpy as np

from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from models.model_nvidia import nvidia_model


def parse_json(data):
    array = []
    # We process json
    data_parse = data.split('}')[:-1]
    for d in data_parse:
        v = d.split('"v":')[1]
        d_parse = d.split(', "v":')[0]
        w = d_parse.split((': '))[1]
        array.append([int(v), float(w)])

    return array


def get_images(list_images):
    array_imgs = []

    for name in list_images:
        # This is a PIL image
        print(name)
        img = image.load_img(name)
        print('load')
        # This is a Numpy array
        img = image.img_to_array(img)
        print('array')
        array_imgs.append(img)

    return array_imgs


if __name__ == "__main__":
    seed = 7
    np.random.seed(seed)

    # Load data
    list_images = glob.glob('../Dataset/Images/' + '*')
    images = sorted(list_images, key=lambda x: int(x.split('/')[3].split('.png')[0]))

    file = open('../Dataset/data.json', 'r')
    data = file.read()
    file.close()

    # We preprocess images
    x = get_images(images)
    # We preprocess json
    y = parse_json(data)

    print(len(x), len(y))

    # Split data into 67% for train and 33% for test
    #X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=seed)

    # Variables
    batch_size = 128
    nb_epoch = 2000
    img_shape = (640, 480, 3)

    model = nvidia_model(img_shape)

    #print(len(X_train), len(X_test), len(y_train), len(y_test))