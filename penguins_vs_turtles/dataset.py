import json
import zipfile
from tqdm import tqdm

DATASET_FILE = 'dataset.zip'


def load_dataset():
    read_dataset_from_zipfile()


def read_dataset_from_zipfile():
    zf = zipfile.ZipFile(DATASET_FILE, 'r')
    # show_image_from_raw_data(zf.read(name=zf.namelist()[0]))
    x_train, y_train, x_test, y_test = [], [], [], []
    for file in tqdm(zf.namelist()):
        # image_data = zf.read(name=file)
        if 'train/train' in file:
            # x_train.append(open_image_from_raw_data(image_data))
            # y_train.append(create_label(file))
            print(file)
        elif 'valid/valid' in file:
            # x_test.append(open_image_from_raw_data(image_data))
            # y_test.append(create_label(file))
            print(file)
        else:
            annotation = zf.open(name=file)
            annotation_json = json.load(annotation)
            print(f'not possible to read from {annotation_json}')

    # return np.asarray(x_train), np.asarray(y_train), np.asarray(x_test), np.asarray(y_test)