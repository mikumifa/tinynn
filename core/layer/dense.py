from core.layer.layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, n_inputs, n_outputs):
        self.inputs = None
        self.params = {
            'w': np.random.randn(n_inputs, n_outputs),
            "b": np.random.randn(1, n_outputs)
        }
        self.grads = {
            'w': np.zeros((n_inputs, n_outputs)),
            "b": np.zeros((1, n_outputs))
        }

    def params_weights(self):
        return self.params["w"]

    def params_bias(self):
        return self.params["b"]

    def grad_weights(self):
        return self.grads["w"]

    def grad_bias(self):
        return self.grads["b"]

    def forward(self, x):
        self.inputs = x
        return np.dot(x, self.params_weights()) + self.params_bias()

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = grad
        return grad @ self.params_weights().T
