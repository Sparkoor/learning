"""
画出图像
"""
import matplotlib.pyplot as plt
import numpy as np

from logicRegression import *


def plotBestFit(weights):
    """
    画出数据集和logistic回归拟合直线的函数
    :param weights:
    :return:
    """
    dataMat, classLabels = loadDataSet()
    dataArr = np.array(dataMat)
    n = dataArr.shape[0]
    # 存放点的，因为需要表示两个类别所以分成两块
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(n):
        if int(classLabels[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    # 令函数为0 变量为x1和x2 使用x1表示x2 这是最佳拟合曲线
    y = (-weights[0, 0] - weights[1, 0] * x) / weights[2, 0]
    ax.plot(x, y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__ == "__main__":
    dataMat, classLabels = loadDataSet()
    weights = gradAscent(dataMat, classLabels)
    plotBestFit(weights)
