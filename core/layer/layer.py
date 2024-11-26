from abc import ABCMeta, abstractmethod


class Layer():
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, x):
        pass
