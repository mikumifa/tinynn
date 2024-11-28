from core.layer.layer import Layer
import numpy as np


class Sigmoid(Layer):

    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, x):
        def sigmoid_d(y):
            return y * (1 - y)

        return sigmoid_d(self.output) * x
