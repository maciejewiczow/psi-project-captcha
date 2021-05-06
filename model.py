from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from typing import Tuple, Dict
from ModelArch import ModelArch

def get_checkpoint_dir(arch: ModelArch) -> str:
    return {
        ModelArch.ARCH_1: "arch1",
        ModelArch.ARCH_2: "arch2",
        ModelArch.ARCH_3: "arch3"
    }[arch]

def make_model(input_shape: Tuple[int, int, int], arch: ModelArch) -> Sequential:
    return {
        ModelArch.ARCH_1: make_model_arch1(input_shape),
        ModelArch.ARCH_2: make_model_arch2(input_shape),
        ModelArch.ARCH_3: make_model_arch3(input_shape)
    }[arch]

def make_model_arch1(input_shape: Tuple[int, int, int]):
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

def make_model_arch2(input_shape: Tuple[int, int, int]) -> Sequential:
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

    model.compile(optimizer="SGD", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

def make_model_arch3(input_shape: Tuple[int, int, int]) -> Sequential:
    model=Sequential()

    model.add(Conv2D(filters=4, kernel_size=(3,3), activation="relu", input_shape=input_shape))
    model.add(Conv2D(filters=4, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=8, kernel_size=(3,3), activation="relu"))
    model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(units=256, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=36, activation="softmax"))

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model
