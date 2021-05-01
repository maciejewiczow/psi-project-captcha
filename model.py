from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

def make_model(input_shape):
    model=Sequential()

    model.add(Conv2D(filters=4, kernel_size=(3,3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=4, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=36, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
