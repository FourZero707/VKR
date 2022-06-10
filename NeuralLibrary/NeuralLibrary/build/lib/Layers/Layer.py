from Neurons.Neuron import Neuron
import numpy as np
from abc import abstractmethod
import random

class Layer(object):
    def __init__(self, neuronsCount, bias, activationFunction = None):
        self.neuronsList = list()
        self.activationFunction = activationFunction
        self.bias = bias

        for i in range(neuronsCount):
            self.neuronsList.append(Neuron(activationFunction))

        if not activationFunction == None:
            self.weights = list()

    def printLayer(self):
        if not self.activationFunction == None:
            print(self.weights)
        else:
            print('input')

    def initWeights(self, prevLayer, index):
        self.weights = list()
        for i in range(len(self.neuronsList)):
            self.weights.append(list())
            for j in range(len(prevLayer.neuronsList)):
                self.weights[i].append(random.uniform(-1.0, 1.0))

    @abstractmethod
    def getOutput(self, input):
        pass

    @abstractmethod
    def refresh(self):
        pass

    def updateParams(self, y, layerIndex, layers):
        sigmaList = list()

        for i in range(len(self.weights)):
            sigma = self.getSigma(i, y, layerIndex, layers)
            sigmaList.append(sigma)
        return sigmaList

    def setNewWeights(self, sigmaList, scale, inputs):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= scale * sigmaList[i] * inputs[j]
