import keras
import numpy as np
from matplotlib import pyplot as plt

PREDICT_MAP = {0: 'non_smile', 1: 'smile'}


def predict_and_show(data: np.array, model: keras.Sequential):
    predict = model.predict(data[:1])
    predict = np.argmax(predict, axis=1)
    print(PREDICT_MAP[predict[0]])

    plt.figure(figsize=(10, 5))
    plt.imshow(data[0])
    plt.show()
