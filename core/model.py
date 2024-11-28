class Model:

    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = loss
        self.optimizer = optimizer

    def train(self, epoch_num, x, y):
        for epoch in range(epoch_num):
            print(f"Epoch: {epoch + 1}/{epoch_num}")
            len = x.shape[0]
            for i in range(len):
                self.train_epoch(x[i], y[i])

    def train_epoch(self, x, y):
        outputs = self.net.forward(x)
        loss = self.loss.loss(outputs, y)
        grad = self.loss.grad(outputs, y)
        self.net.backward(grad)
        self.optimizer.update_params(self.net.grads(), self.net.params())
        print(f"    Loss: {loss}")

    def predict(self, x):
        return self.net.forward(x)
