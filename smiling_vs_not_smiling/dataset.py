import cv2
import zipfile
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

DATASET_FILENAME = 'dataset.zip'


def label_from_filename(filename) -> int:
    category = filename.split('/')[0]
    if category == 'non_smile':
        return 0
    elif category == 'smile':
        return 1
    else:
        print(f'label fails for file: {filename}')
        return -1


def cv_image_from_file_data(image):
    return cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)


def load_dataset() -> tuple:
    zip_file = zipfile.ZipFile(DATASET_FILENAME, 'r')
    x_train, y_train, test = [], [], []
    for file in tqdm(zip_file.namelist()):
        file_data = zip_file.read(name=file)
        if 'test' in file:
            test.append(cv_image_from_file_data(file_data))
        elif 'smile' in file:
            x_train.append(cv_image_from_file_data(file_data))
            y_train.append(label_from_filename(file))
        else:
            print(f'not possible append {file} in dataset')
    x_train, y_train, test = np.asarray(x_train), np.asarray(y_train), np.asarray(test)
    print(f'x_train: {x_train.shape}, y_train: {y_train.shape}, test: {test.shape}')
    return x_train, y_train, test
