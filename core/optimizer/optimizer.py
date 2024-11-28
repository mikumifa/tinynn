from abc import abstractmethod


class Optimizer:
    @abstractmethod
    def update_params(self, grads, params):
        pass
