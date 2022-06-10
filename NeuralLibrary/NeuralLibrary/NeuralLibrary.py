from Network import Network
from Layers.Layers import *

import keras
from keras.datasets import mnist
from keras.utils import np_utils

import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

network = Network()
network.addLayer(Input(784))
network.addLayer(RNN(50))
#network.addLayer(Dense(80, bias=0))
#network.addLayer(Dense(80, bias=0))
network.addLayer(Dense(10, bias=0))
network.compile()

num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')

X_train = X_train / 255

X = X_train[0: 1000]

y_train = np_utils.to_categorical(y_train)
Y = y_train[0: 1000]

o = network.getOutput(X[0])

network.fit(X, Y, 10, 0.15)

good = 0

X = X_train[1000: 2000]
Y = y_train[1000: 2000]

for i in range(len(X)):
    o = network.getOutput(X[i])
    o = o.index(max(o))
    o1 = Y[i].argmax()
    if o == o1:
        good += 1

print(str(good))

input()