from Layers.Layer import Layer
from Activations.Sigmoid import Sigmoid
import numpy as np
import random
from copy import copy, deepcopy


class Dense(Layer):
    def __init__(self, neuronsCount, bias = 0, activationFunction = Sigmoid()):
        super().__init__(neuronsCount, bias,  activationFunction)

    def getSigma(self, i, y, layerIndex, layers):
        currentLayer = layers[layerIndex]
        isLast = layerIndex == len(layers) - 1
        if isLast == True:
            sigma = -self.activationFunction.deriviate(currentLayer.lastOutput[i]) * (y[i] - currentLayer.lastOutput[i])
        else:
            sumSigma = 0.0
            for k in range(len(layers[layerIndex + 1].neuronsList)):
                sumSigma += self.getSigma(k, y, layerIndex + 1, layers) * layers[layerIndex + 1].weights[k][i]
            sigma = self.activationFunction.deriviate(currentLayer.lastOutput[i]) * sumSigma
        return sigma

    def getOutput(self, input):
        result = list()
        for i in range(len(self.neuronsList)):
            total = np.dot(self.weights[i], input) + self.bias
            result.append(self.neuronsList[i].calculate(total))

        self.lastOutput = result

        return result

class Input(Layer):
    def __init__(self, neuronsCount):
        super().__init__(neuronsCount, 0)

    def getOutput(self, input):
        self.lastOutput = input
        return input

class RNN(Layer):
    def __init__(self, neuronsCount, bias = 0, activationFunction = Sigmoid()):
        super().__init__(neuronsCount, bias, activationFunction)
        self.refresh()

    def refresh(self):
        self.previousWeight = list()
        self.previousInputs = list()

    def initWeights(self, prevLayer, index):
        super().initWeights(prevLayer, index)

    def getOutput(self, input):
        result = list()
        if not self.activationFunction == None:
            for i in range(len(self.neuronsList)):
                if len(self.previousInputs) == 2:
                    total = np.dot(self.weights[i], input) + np.dot(self.previousWeight[0][i], self.previousInputs[0]) + self.bias
                else:
                    total = np.dot(self.weights[i], input) + self.bias
                result.append(self.neuronsList[i].calculate(total))

        self.lastOutput = result

        self.previousInputs.append(deepcopy(input))
        self.previousWeight.append(deepcopy(self.weights))

        if len(self.previousInputs) == 3:
            self.previousInputs.pop()
            self.previousWeight.pop()

        return result

    def getPrevOutput(self):
        result = list()
        for i in range(len(self.neuronsList)):
            if len(self.previousInputs) == 2:
                total = np.dot(self.previousWeight[0][i], self.previousInputs[0]) + self.bias
                result.append(self.neuronsList[i].calculate(total))
            else:
                result.append(0.0)

        return result

    def getSigma(self, i, y, layerIndex, layers):
        currentLayer = layers[layerIndex]
        isLast = layerIndex == len(layers) - 1
        if isLast == True:
            sigma = -self.activationFunction.deriviate(currentLayer.lastOutput[i]) * (y[i] - currentLayer.lastOutput[i])
        else:
            sumSigma = 0.0
            for k in range(len(layers[layerIndex + 1].neuronsList)):
                sumSigma += self.getSigma(k, y, layerIndex + 1, layers) * layers[layerIndex + 1].weights[k][i]
            sigma = (self.activationFunction.deriviate(currentLayer.lastOutput[i]) + self.activationFunction.deriviate(self.getPrevOutput()[i])) * sumSigma
        return sigma