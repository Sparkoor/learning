"""
    实现单层神经网络
"""
import numpy as np


# 加载数据集
def load_dataset():
    pass


# 激活函数 ReLU函数 f(x)=MAX(0,X)
def ReLU(z):
    mz, nz = z.shape
    zero = np.zeros(mz, nz)
    return np.maximum(zero, z)


# sigmoid函数
def sigmoid(z):
    return 1.0 / (1 + np.exp(z))


# 将四位数组转换成2维数组,在这不知道，元素数据的结构是什么，没办法理解
def tansformArray(inX):
    shape = inX.shape
    shape1 = shape[1] * shape[2] * shape[3]
    tranformedX = np.mat(np.zeros((shape1, shape[0])))
    for i, item in enumerate(inX):
        tranformedItem = item.flatten().reshape(-1, 1)
        tranformedX[:, i] = tranformedItem
    return tranformedX


# 损失函数
def lossFunc(A, y_train):
    pass


# 单层神经网络
class OneLNN:
    def __init__(self, w1, b1, alpha, af, iterNum):
        self.w1 = w1
        self.b1 = b1
        self.alpha = alpha
        self.af = af
        self.iterNum = iterNum

    # 还没转化成向量的计算，不知道数据集是什么样子的
    def trainFunc(self, X_train, Y_train):
        m = X_train.shape[1]
        x = X_train
        Y = Y_train
        Z = self.w1 * x + self.b1
        for i in range(self.iterNum):
            Z = self.w1 * x + self.b1
            A = self.af(Z)
            dZ = A - Y
            dw = dZ * x.T / m
            db = np.sum(dZ) / m
            self.w1 -= self.alpha * dw
            self.b1 -= self.alpha * db
        return self.w1, self.b1

    def predictFunc(self, X_test):
        Z_pred = self.w1 * X_test + self
        # af传的是一个函数，sigmoid函数
        A_pred = self.af(Z_pred)

        return y_pred

    def testError(self, y_test, y_pred):
        m = y_test.shape[1]
        errors = len(np.nonzero(y_test != y_pred)[0])
        errorRate = errors * 1.0 / m
        return errorRate


# 加载数据
X_train, y_train, X_test, y_test, classse = load_dataset()
# 将采集的数组转换成二维数组，这是作者手动转的
x_train_transformed = tansformArray(X_train)
x_test_tranformed = tansformArray(X_test)
# 训练样本维数
n0 = x_train_transformed[0]
# 训练样本个数
m = x_train_transformed[1]
# 输出单元个数
n1 = 1
# 初始权值
w1 = np.random.randn(n1, n0) * 0.01
b1 = 0
iterNum = 500
# 交叉检验
errList = []
for i in range(10):
    oneLNN = OneLNN(w1, b1, alpha=0.001, af=sigmoid, iterNum=iterNum)
    w, b = oneLNN.trainFunc(x_train_transformed, y_train)
    y_pred = oneLNN.predictFunc(x_test_tranformed)
    errorRate = oneLNN.testError(y_test, y_pred)
    errList.append(errorRate)
