class Neuron(object):
    def __init__(self, activationFunction):
        self.activationFunction = activationFunction

    def calculate(self, input):
        return self.activationFunction.calculate(input)