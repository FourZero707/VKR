from Loss.MSE import MSE

class Network(object):
    def __init__(self):
        self.layers = list()
        self.compiled = False

    def addLayer(self, layer):
        self.layers.append(layer)

    def print(self):
        for i in range(len(self.layers)):
            print(i)
            self.layers[i].printLayer()

    def compile(self, lossFunction = MSE()):
        for i in range(len(self.layers) - 1):
            self.layers[i+1].initWeights(self.layers[i], i)

        self.lossFunction = lossFunction

    def getOutput(self, input):
        result = input
        for i in range(len(self.layers)):
            result = self.layers[i].getOutput(result)
        return result

    def refreshRNN(self):
        for i in range(len(self.layers)):
            self.layers[i].refresh()

    def fit(self, X, Y, epochs, learningRate):
        self.refreshRNN()
        for epoch in range(epochs):
            print('epoch: ' + str(epoch))
            outputList = list()
            if epoch == epochs - 2:
                stop = True
            for i in range(len(X)):
                output = self.getOutput(X[i])
                real = Y[i]

                outputList.append(output)

                sigmaList = list()

                for k in range(len(self.layers) - 1, 0, -1):
                    sigmaList.append(self.layers[k].updateParams(real, k, self.layers))

                for k in range(len(sigmaList)):
                    self.layers[len(self.layers) - k - 1].setNewWeights(sigmaList[k], learningRate, self.layers[len(self.layers) - k - 2].lastOutput)

            print('error: ' + str(self.lossFunction.calculate(Y, outputList)))