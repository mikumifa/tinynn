from core.layer.layer import Layer
import numpy as np


class Sigmoid(Layer):
    def __init__(self):
        self.dz = None
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, x):
        def sigmoid_d(y):
            return y(1 - y)

        self.dz = sigmoid_d(self.output)*x
        return self.dz
