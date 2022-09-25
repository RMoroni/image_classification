import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_label(filename):
    category = filename.split('/')[1]
    if category == 'children':
        return 0
    elif category == 'adults':
        return 1
    else:
        print('label fails')


def open_image(image_data):
    return cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)


def read_dataset():
    zf = zipfile.ZipFile('dataset.zip', 'r')
    # test_image(zf.read(name=zf.namelist()[0]))
    x_train, y_train, x_test, y_test = [], [], [], []
    for file in tqdm(zf.namelist()):
        image_data = zf.read(name=file)
        if 'train' in file:
            x_train.append(open_image(image_data))
            y_train.append(create_label(file))
        elif 'test' in file:
            x_test.append(open_image(image_data))
            y_test.append(create_label(file))
        else:
            print(f'not possible to read from {file}')

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)


def test_image(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("Image Sample", img)
    cv2.waitKey(0)
    print(np.array(img).shape)


def check_label_from_sample(train, test):
    train_sample = train[0]
    test_sample = test[0]
    print('should show the same category:')
    print(test_sample)
    plt.figure(figsize=(10, 5))
    plt.imshow(train_sample)
    plt.show()


def load_dataset():
    x_train, y_train, x_test, y_test = read_dataset()
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # check_label_from_sample(x_test, y_test)
    # x_train, x_test = x_train.reshape(-1, INPUT_LENGTH), x_test.reshape(-1, INPUT_LENGTH)  # reshape into a list
    return x_train, y_train, x_test, y_test
