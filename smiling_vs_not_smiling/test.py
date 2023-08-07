import keras
import numpy as np
from matplotlib import pyplot as plt


def predict_and_show(data: np.array, model: keras.Sequential, n=20):
    predict = model.predict(data[:n])
    predict = np.argmax(predict, axis=1)

    for index, image in enumerate(data[:n]):
        plt.figure(figsize=(10, 5))
        plt.imshow(image)
        plt.title(f'Predict: {predict[index]}')
        plt.show()
