"""
k均值聚类算法
"""
import numpy as np
from commonUtils.Loggings import *

logger = Logger().getLogger()


def loadData(fileName):
    dataMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineList = line.strip().split()
            dataMat.append(list(map(float, lineList)))
    logger.info("加载数据%d条" % len(dataMat))
    return dataMat


def distEclud(vecA, vecB):
    """
    欧式距离
    :param vecA:
    :param vecB:
    :return:
    """
    logger.info("计算向量{}和向量{}的欧式距离".format(vecA, vecB))
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    初始化质心
    :param dataSet:
    :param k:
    :return:
    """
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k, n)))
    for i in range(n):
        minI = np.min(dataSet[:, i])
        rangI = float(np.max(dataSet[:, i]) - minI)
        centroids[:, i] = minI + rangI * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    K-均值聚类算法
    :param dataSet:
    :param k:
    :param distMeas:
    :param createCent:
    :return:
    """
    dataMat = np.mat(dataSet)
    m = dataMat.shape[0]
    # 存放分类结果和距离，序列可以存样本的的坐标
    clusterAssment = np.mat(np.zeros((m, 2)))
    centriods = createCent(dataMat, k)
    logger.warning("更新前的质心为{}".format(centriods))
    # 迭代次数
    times = 0
    # 为了迭代更新质心
    cluaterChanged = True
    while cluaterChanged:
        times += 1
        logger.debug("质心更新次数{}".format(times))
        cluaterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                # 计算距离
                minDiste = distMeas(dataMat[i, :], centriods[j, :])
                # 这个地方范错了
                if minDist > minDiste:
                    minDist = minDiste
                    minIndex = j
            # 判断是否继续循环更新
            if clusterAssment[i, 0] != minIndex:
                logger.critical('继续循环')
                cluaterChanged = True
            # logger.critical("----{}---{}".format(minIndex, minDist))
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 更新
        for cent in range(k):
            # logger.critical("cluster{}".format(clusterAssment))
            # 取出相同质心的数据集
            ptsInclust = dataMat[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            # logger.error("ptsInclust{}".format(ptsInclust))
            # 求均值 axis=0是竖着 1是横着
            centriods[cent, :] = np.mean(ptsInclust, axis=0)
        # logger.warning("更新后的质心为{}".format(centriods))
    return centriods, clusterAssment


if __name__ == '__main__':
    dataSet = loadData('testSet.txt')
    centriods, cluster = kMeans(dataSet, 5, distEclud, randCent)
