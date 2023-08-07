import numpy as np
from keras import Sequential


def train_model(model: Sequential, x: np.array, y: np.array, batch_size=32, epochs=40):
    model.fit(
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
    )
