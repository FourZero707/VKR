from Loss.LossFunction import LossFunction
import numpy as np

class MSE(LossFunction):
    def calculate(self, y_true, y_pred):
        sum = 0.0
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                sum += (y_true[i][j] - y_pred[i][j]) ** 2

        return (1.0 / len(y_true) * sum) ** 0.5