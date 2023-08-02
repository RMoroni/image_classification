from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


def set_model_lenet5(model: Sequential):
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(2, activation='softmax'))


def get_model_by_name(model_name='lenet5'):
    model = Sequential()
    if model_name == 'lenet5':
        set_model_lenet5(model)
    else:
        pass
    model.build()
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
