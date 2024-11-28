import numpy as np

from core.loss.loss import Loss


class MSE(Loss):

    def loss(self, outputs, y):
        return 1 / 2 * np.sum(np.square(outputs - y))

    def grad(self, outputs, y):
        return outputs - y
