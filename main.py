import os
import cv2
import zipfile
import numpy as np
from tqdm import tqdm


def read_dataset():
    zf = zipfile.ZipFile('dataset.zip', 'r')
    # print(zf.namelist())
    img_file = zf.read(name='test/adults/1.jpg')
    test_image(img_file)


def test_image(img_data):
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow("Image Sample", img)
    cv2.waitKey(0)
    print(np.array(img).shape)


if __name__ == '__main__':
    read_dataset()
