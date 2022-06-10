from abc import abstractmethod

class LossFunction(object):
    @abstractmethod
    def calculate(self, y_true, y_pred):
        pass