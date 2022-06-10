import numpy as np

from Activations.ActivationFunction import ActivationFunction

class Sigmoid(ActivationFunction):
    def calculate(self, input):
        return 1 / (1 + np.exp(-input))

    def deriviate(self, input):
        return input * (1 - input)