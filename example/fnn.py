import numpy as np

from core.layer.dense import Dense
from core.layer.sigmoid import Sigmoid
from core.loss.MSE import MSE
from core.model import Model
from core.net import Net
from core.optimizer.ClassicOptimizer import ClassicOptimizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model = Model(net=Net(layers=[Dense(1, 25), Sigmoid(), Dense(25, 1)]),
                  loss=MSE(),
                  optimizer=ClassicOptimizer(1e-3))
    np.random.seed(42)
    x = np.random.uniform(-np.pi, np.pi, 100).reshape(-1, 1)
    y = 1 / 2 * (np.sin(x) + np.random.normal(0, 0.1, x.shape) + 1)
    model.train(epoch_num=400, x=x, y=y)

    # 预测结果
    x_test = np.linspace(-np.pi, np.pi, 50).reshape(-1, 1)  # 测试集：连续的点
    y_pred = model.predict(x_test)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Random Data", color="blue", alpha=0.6)
    plt.scatter(x_test, y_pred, label="Model Prediction", color="red")
    plt.legend()
    plt.title("Fitting sin(x) with Neural Network on Random Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
