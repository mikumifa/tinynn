from abc import ABCMeta, abstractmethod


class Layer:
    def __init__(self):
        self.params = {
            'w': 0,
            "b": 0
        }
        self.grads = {
            'w': 0,
            "b": 0
        }

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass
