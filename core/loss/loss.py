from abc import abstractmethod


class Loss:

    @abstractmethod
    def loss(self, outputs, y):
        """
        损失函数计算损失
        :param outputs: 预测值
        :param y: 真实值
        :return: 误差
        """
        pass

    def grad(self, outputs, y):
        """
        计算出初始grad
        :param outputs: 预测值
        :param y: 真实值
        :return: 梯度
        """
        pass
