# uncompyle6 version 3.5.1
# Python bytecode 3.5 (3351)
# Decompiled from: Python 3.7.4 (tags/v3.7.4:e09359112e, Jul  8 2019, 20:34:20) [MSC v.1916 64 bit (AMD64)]
# Embedded file name: C:\Users\PranayDev\Documents\ComputerVision\Project\ImageEmotion\prepare_network.py
# Compiled at: 2019-11-29 19:29:37
# Size of source mod 2**32: 1797 bytes
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
input_shape = (112, 112, 3)

def get_network():
    model = Sequential([
     Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
     Conv2D(64, (3, 3), activation='relu', padding='same'),
     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     Conv2D(128, (3, 3), activation='relu', padding='same'),
     Conv2D(128, (3, 3), activation='relu', padding='same'),
     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     Conv2D(256, (3, 3), activation='relu', padding='same'),
     Conv2D(256, (3, 3), activation='relu', padding='same'),
     Conv2D(256, (3, 3), activation='relu', padding='same'),
     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     Conv2D(512, (3, 3), activation='relu', padding='same'),
     Conv2D(512, (3, 3), activation='relu', padding='same'),
     Conv2D(512, (3, 3), activation='relu', padding='same'),
     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     Conv2D(512, (3, 3), activation='relu', padding='same'),
     Conv2D(512, (3, 3), activation='relu', padding='same'),
     Conv2D(512, (3, 3), activation='relu', padding='same'),
     MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
     Flatten(),
     Dense(1024, activation='relu'),
     Dense(512, activation='relu'),
     Dense(256, activation='relu'),
     Dense(8, activation='softmax')])
    print(model.summary())
    sgd = keras.optimizers.SGD(lr=0.01, clipnorm=1.0)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])
    return model


get_network()