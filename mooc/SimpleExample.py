"""
简单的实现，根据视频写的
"""
import numpy as np


# 不会加载数组集，此功能先略过
def load_sets():
    pass


class SimpleNN:

    def __init__(self, W, B, alpha, n):
        """
        :param W: 权值，这是一个二维数组包括坐标和值
        :param B: 阙值，这也是一个二维数组包括坐标和值
        :param alpha: 学习率
        :param n: 是个list包含每个隐藏层的神经元个数
        """
        self.W = W
        self.B = B
        self.alpha = alpha
        self.n = n

    def sigmoid(self, Z):
        """
        使用sigmoid函数做激活函数
        :param Z:
        :return: a 前一个神经元的输出，下一个神经元的输入
        """
        return 1 / (1 + np.exp(Z))

    # 计算数列的值
    def gfunction(self, sig, Z):
        return list(map(sig, Z))

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def train(self, x, y):
        """
        训练方法,并不知道能不能用
        :param x: 是一个一维数组
        :param y: 是一个数
        :return: 返回w二维数组b二维数组
        """
        # 这是输入参数的维度
        n0 = x.shape[1]
        a = x
        # n表示有n层神经
        Z, A, dZ = [], [], []
        # 正向传播
        for l in range(self.n):
            # 每层有多少个神经元
            w, b = self.W[l], self.B[l]
            # 这一层的神经元数
            m = w.shape[1]
            z = np.dot(a, w) + b
            a = self.gfunction(self.sigmoid, Z)
            Z.append(z)
            A.append(a)
        # 取出输出层的数据
        dZL = A[self.n - 1] - y
        AL = a[self.n - 1]
        dWL = np.dot(dZL, AL) / self.W[self.n - 1][1]
        dBL = np.sum(dZL, axis=1, keepdims=True)
        self.W[self.n - 1] -= self.alpha * dWL
        self.B[self.n - 1] -= self.alpha * dBL
        # 反向
        for l in range(self.n - 1, 0, - 1):
            dz = dWL.T * dZL * (Z[l - 1] * (1 - Z[l - 1]))
            dw = (dz * A[l - 1].T) / m
            db = np.sum(dz, axis=1, keepdims=True)
            self.W[l - 1] -= self.alpha * dw
            self.B[l - 1] -= self.alpha * db
            dWL, dZL = dw, dz
        return self.W, self.B, A

    def predict(self, x_test):
        """
        把正向传播走一遍 激活函数选择不同的函数
        :param x_test:
        :return:
        """
        for i in range(self.n):
            w = self.W[i]
            b = self.B[i]
            z = np.dot(w, x_test) + b
            a = self.gfunction(self.sigmoid, z)
            x_test = a
        return a

    # 损失函数
    def lossFunction(self, y, y_predict):
        return (-y) * np.log(y_predict) - (1 - y) * np.log(1 - y_predict)


if __name__ == '__main__':
    # 权值
    w = np.random.randn(4, 5)
    # 阙值
    b = np.zeros(4, 5)
    # 步长因子
    alpha = 0.01
    # 层数
    n = w.shape[0]
