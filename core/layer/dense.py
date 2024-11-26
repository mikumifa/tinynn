from core.layer.layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.dz = None
        self.outputs = None
        self.weights = np.random.randn(n_inputs, n_outputs)
        self.bias = np.random.randn(1, n_outputs)

    def forward(self, x):
        self.outputs = np.dot(x, self.weights) + self.bias
        return self.outputs

    def backward(self, x):
        self.dz = x @ self.weights.T
        return self.dz
