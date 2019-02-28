"""
线性回归算法
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
    #  note:
    try:
        fr = open(fileName, 'r')
        numFeat = len(fr.readlines().split('\t')) - 1
    except FileNotFoundError as e:
        logger.error(e)
    dataMat = []
    classLabels = []
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
