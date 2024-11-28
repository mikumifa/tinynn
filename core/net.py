class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def params(self):
        grads = []
        for layer in self.layers:
            grads.append(layer.params)
        return grads

    def grads(self):
        grads = []
        for layer in self.layers:
            grads.append(layer.grads)
        return grads

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
