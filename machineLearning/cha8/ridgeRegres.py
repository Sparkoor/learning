"""
岭回归,通过引入惩罚项，减少不重要的参数
"""
import numpy as np
from commonUtils.Loggings import *

logger = Logger().getLogger()


def loadData(fileName):
    """
    加载数据
    :param fileName:
    :return:
    """
    #  note: 这是使用异常的形式
    try:
        fr = open(fileName, 'r')
        numFeat = len(fr.readlines().split('\t')) - 1
    except FileNotFoundError as e:
        logger.error(e)
    dataMat = []
    classLabels = []
    # note：with上下文管理语句，解决多线程问题
    with open(fileName) as fr:
        lines = fr.readlines()
        for line in lines:
            dataline = line.strip().split('\t')
            # note:转换成int
            dataArr = list(map(float, dataline[:-2]))
            dataMat.append(dataArr)
            classLabels.append(float(dataline[-1]))
    return dataMat, classLabels


def standRegress(dataArr, classLabel):
    """
    标准回归函数
    :param dataArr:
    :param classLabel:
    :return:
    """
    xMat = np.mat(dataArr)
    yMat = np.mat(classLabel).T
    xTx = xMat.T * xMat
    # note:求矩阵行列式
    if np.linalg.det(xTx) == 0:
        logger.error("逆矩阵为0")
        return
    # note:求逆
    ws = xTx.I * (xMat.T * yMat)
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    局部加权线性回归
    :param testPoint: 测试的一个样本
    :param xArr: 训练的点
    :param yArr: 结果
    :param k: 参数k
    :return:
    """
    testMat = np.mat(testPoint)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    m = xMat.shape[0]
    # 生成对角阵
    weight = np.mat(np.eye(m))
    for i in range(m):
        diffMat = testMat - xMat[i, :]
        weight[i, i] = np.exp(diffMat * diffMat.T / (-2 * k ** 2))
    xTx = xMat.T * (weight * xMat)
    # 零矩阵不存在颗逆矩阵
    if np.linalg.det(xTx) == 0.0:
        logger.error("行列式为零")
        return
    ws = xTx.I * (xMat.T * (weight * yMat))
    return testMat * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    局部加权回归测试函数
    :param testArr:
    :param xArr:
    :param yArr:
    :param k:
    :return:
    """
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def regressErrorr(yHat, yArr):
    """
    计算误差
    :param yHat:
    :param yArr:
    :return:
    """
    return ((yHat - yArr) ** 2).sum()


def ridgeRegress(xMat, yMat, lam=0.2):
    """
    岭回归算法
    :param xMat:
    :param yMat:
    :param lam:
    :return: X.T*X+lam*I
    """
    xTx = xMat.T * xMat
    denom = xTx + lam * np.eye(xTx.shape[1])
    if np.linalg.det(denom) == 0.0:
        logger.error("行列式为0")
        return
    # todo:验证不同的相乘顺序
    return denom.I * (xMat.T * yMat)


def ridgeTest(xArr, yArr):
    """
    岭回归的测试
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # 均值 --------------------标准化处理start----------------------
    yMean = np.mean(yArr, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    # 求方差
    xVar = np.var(xMean, 0)
    # note:特征值的标准化：所有特征都减去各自的均值并除以方差
    xMat = (xMat - xMean) / xVar
    # ------------------------标准化处理end---------------------------
    # 调用的次数
    numTestPts = 30
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        # lam 选取30个数
        ws = ridgeRegress(xMat, yMean, np.exp(i - 10))
        wMat[i, :] = ws
    return wMat
