from abc import abstractmethod

from core.optimizer.optimizer import Optimizer


class ClassicOptimizer(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update_params(self, grads, params):
        steps = self.calculate_steps(grads)
        for step, param in zip(steps, params):
            for k in step:
                param[k] += step[k]

    def calculate_steps(self, grads):
        steps = []
        for grad in grads:
            step = {}
            for k in grad:
                step[k] = - self.learning_rate * grad[k]
            steps.append(step)
        return steps
