"""
基于logistic回归和sigmoid函数的分类
sigmoid函数类似于一种单位阶跃函数，该函数在跳越点上从0瞬间跳越到1
"""
from math import *
import numpy as np
import random


def loadDataSet():
    dataMat = [];
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(z):
    """
    sigmoid函数
    :param z:
    :return:
    """
    # 这是numpy中的
    return 1.0 / (1 + np.exp(-z))


def gradAscent(dataMat, classLabels):
    """
    使用梯度上升算法求梯度
    :param dataMat:
    :param classLabels:
    :return:
    """
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(classLabels).transpose()
    m, n = dataMatrix.shape
    alpha = 0.001
    theta = np.ones((n, 1))
    for i in range(500):
        h = sigmoid(dataMatrix * theta)
        error = labelMatrix - h
        # 同一次进行几百个运算
        theta = theta + alpha * dataMatrix.transpose() * error
    return theta


def stocGradAscent0(dataMatrix, classLabels):
    """
    随机梯度上升算法，
    :param dataMatrix:
    :param classLabels:
    :return:
    """
    m, n = dataMatrix.shape
    alpha = 0.01
    # 横着的
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * dataMatrix[i] * error


def classifyLogRege(dataMat, classLabels):
    """
    训练样本,随机10作为测试样本
    :param dataMat:
    :param classLabels:
    :return:
    """
    numData = len(dataMat)
    trainingSet = list(range(numData))
    trainingTest = []
    for i in range(10):
        randomIndex = int(random.uniform(0, len(trainingSet)))
        trainingTest.append(randomIndex)
        del (trainingSet[randomIndex])
    # 准备训练数据
    trainMatrix = []
    trainLabels = []
    for i in trainingSet:
        trainMatrix.append(dataMat[i])
        trainLabels.append(classLabels[i])
    theta = gradAscent(trainMatrix, trainLabels)
    # 测试
    testMatrix = []
    testLabels = []
    error = 0
    for i in trainingTest:
        testMatrix.append(dataMat[i])
        testLabels.append(classLabels[i])
    for i in range(len(testMatrix)):
        z = np.mat(testMatrix[i]) * theta
        x = sigmoid(z)
        print(x, '--', testLabels[i])
        if x > 0.5:
            label = 1
        else:
            label = 0
        if label != testLabels[i]:
            error += 1
    print("错误率为", (error / len(testLabels)))


if __name__ == "__main__":
    dataMat, classLabels = loadDataSet()
    theta = gradAscent(dataMat, classLabels)
    # print(theta)
    classifyLogRege(dataMat, classLabels)
