import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader

DATASET_FILE = 'dataset.zip'
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
OPENCV_SCALE = cv2.IMREAD_GRAYSCALE if IMAGE_CHANNELS == 1 else cv2.IMREAD_COLOR


def create_label(filename):
    category = filename.split('/')[1]
    if category == 'children':
        return 0
    elif category == 'adults':
        return 1
    else:
        print('label fails')


def open_image_from_raw_data(image_data):
    cv2_img = cv2.imdecode(np.frombuffer(image_data, np.uint8), OPENCV_SCALE)
    return cv2.resize(cv2_img, (IMAGE_SIZE, IMAGE_SIZE))


def show_image_from_raw_data(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), OPENCV_SCALE)
    cv2.imshow("Image Sample", img)
    cv2.waitKey(0)
    print(np.array(img).shape)


def read_dataset_from_zipfile():
    zf = zipfile.ZipFile(DATASET_FILE, 'r')
    # show_image_from_raw_data(zf.read(name=zf.namelist()[0]))
    x_train, y_train, x_test, y_test = [], [], [], []
    for file in tqdm(zf.namelist()):
        image_data = zf.read(name=file)
        if 'train' in file:
            x_train.append(open_image_from_raw_data(image_data))
            y_train.append(create_label(file))
        elif 'test' in file:
            x_test.append(open_image_from_raw_data(image_data))
            y_test.append(create_label(file))
        else:
            print(f'not possible to read from {file}')

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)


def check_label_from_sample(image, label):
    print('should show image with label:')
    print(label)
    plt.figure(figsize=(10, 5))
    plt.imshow(image)
    plt.show()


def load_dataset():
    x_train, y_train, x_test, y_test = read_dataset_from_zipfile()
    # print(f'Train Shape: \n\tX: {x_train.shape}\n\tY: {y_train.shape}')
    # print(f'Test Shape: \n\tX: {x_test.shape}\n\tY: {y_test.shape}')
    # check_label_from_sample(x_test, y_test)
    # x_train, x_test = x_train.reshape(-1, INPUT_LENGTH), x_test.reshape(-1, INPUT_LENGTH)  # reshape into a list
    return x_train, y_train, x_test, y_test


def dataset_to_dataloader(data, labels, batch_size=1):
    data = data.reshape((len(data), IMAGE_SIZE, IMAGE_SIZE, 1)) if IMAGE_CHANNELS == 1 else data
    tensor_x = from_numpy(data.transpose(0, 3, 1, 2))
    tensor_y = from_numpy(labels)
    tensor_dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)
