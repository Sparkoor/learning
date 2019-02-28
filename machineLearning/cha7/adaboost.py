"""
元算法（集成算法）
"""
import numpy as np
import matplotlib.pyplot as plt
from commonUtils.Loggings import *

logger = Logger().getLogger()


def loadSimpleData():
    dataMat = np.matrix([[1, 2.1],
                         [2, 1.1],
                         [1.3, 1],
                         [1, 1],
                         [2, 1]])
    classLabels = [1, 1, -1, -1, 1]
    return dataMat, classLabels


def plotSimpleData():
    dataMat, classLabels = loadSimpleData()
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(len(classLabels)):
        if classLabels[i] == 1:
            xcord1.append(dataMat[i, 0])
            ycord1.append(dataMat[i, 1])
        else:
            xcord2.append(dataMat[i, 0])
            ycord2.append(dataMat[i, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.show()


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树，树墩，使用循环变化步长，改变分解的值，使正确率最高
    :param dataMatrix:
    :param dimen: 任一元素
    :param threshVal: 分界的数值
    :param threshIneq: 大于还是小于好 lt 小于等于 gt大于
    :return:
    """
    retArray = np.ones((dataMatrix.shape[0], 1))
    if threshIneq == 'lt':
        # 这是某个特征的值大于某个固定的量，用于分类
        retArray[dataMatrix[:, dimen] <= threshVal] = -1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    找到数据集上的最佳单层决策树
    :param dataArr:
    :param classLabels:
    :param D: 权重向量
    :return:
    """
    dataMat = np.mat(dataArr)
    # 按列比对
    labelMat = np.mat(classLabels).T
    m, n = dataMat.shape
    # 设置成无穷大
    errorMin = np.inf
    numStep = 10.0
    bestStump = {}
    # 最好的分类结果
    bestClassEst = np.mat(np.zeros((m, 1)))
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        # 确定步长，用于寻找最佳的分类点
        stepSize = (rangeMax - rangeMin) / numStep
        for j in range(-1, int(numStep) + 1):
            for inequal in ['lt', 'gt']:
                thresVal = (rangeMin + float(j) * stepSize)
                predictVals = stumpClassify(dataMat, i, thresVal, inequal)
                errorMat = np.mat(np.ones((m, 1)))
                # 对的就为0，为了计算错误率
                errorMat[predictVals == labelMat] = 0
                # 这步不是很明白 D是权重向量，计算错误的比重
                # logger.critical(errorMat.T)
                weightError = D.T * errorMat
                # logger.info(weightError)
                if weightError < errorMin:
                    errorMin = weightError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = thresVal
                    bestStump['ineq'] = inequal
    return bestStump, errorMin, bestClassEst


def adaBoostTrainDs(dataArr, classLabels, numIt=40):
    """
    基于单层决策树的adsBoost训练过程
    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    """
    weakClassArr = []
    m = dataArr.shape[0]
    # 这个地方一定要初始为一
    D = np.mat(np.ones((m, 1)) / m)
    # 记录每个数据点的类别估计累计值
    aggClassEst = np.mat(np.ones((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        logger.info("D %s", D.T)
        # 1e-16防止除零溢出
        logger.critical(max(error, 1e-16))
        alpha = float(0.5 * np.log((1 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        logger.info(classEst.T)
        # 这把正确的和不正确的同时计算了，使用了负负得正的原理
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        logger.info("aaaaa %s", classLabels)
        logger.info('classEst%s', classEst.T)
        logger.info('expon%s', expon.T)
        D = np.multiply(D, np.exp(expon))
        # 更新D的值 都是同时更新的
        D = D / D.sum()
        # 错误累加率 这个公式可用来预测
        aggClassEst += alpha * classEst
        logger.error('aggClassEst%s', aggClassEst.T)
        # np.sign(aggClassEst) != np.mat(classLabels).T 得到预测错误的
        aggError = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        logger.critical((np.sign(aggClassEst) != np.mat(classLabels).T).T)
        # 计算错误率
        errorRate = aggError.sum() / m
        logger.warning('total error%s', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classiferArr):
    """
    分类函数
    :param datToClass:
    :param classiferArr: 训练出来的分类器
    :return:
    """
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classiferArr)):
        # 使用简单分类器里的数据
        classExt = stumpClassify(dataMatrix, classiferArr[i]['dim'], classiferArr[i]['thresh'],
                                 classiferArr[i]['ineq'])
        # 对分类器的结果数量进行累加，权值是局部的权值，每个树里面的
        aggClassEst += classiferArr[i]['alpha'] * classExt
        logger.warning('分类结果%s', aggClassEst)
    return np.sign(aggClassEst)


def plotRoc(predStrongths, classLabels):
    """
    ROC曲线的绘制及AUC计算函数,啥也没画出来
    :param predStrongths:
    :param classLabels:
    :return:
    """
    # 绘制光标的位置
    cur = (1.0, 1.0)
    ysum = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    # 正例的步长
    yStep = 1 / float(numPosClas)
    # 反正例的步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrongths.flatten().argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    logger.critical(sortedIndicies)
    logger.error(classLabels)
    for index in sortedIndicies.tolist()[0]:
        logger.critical(cur)
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
            logger.info("为1的Ystep%s", yStep)
        else:
            delX = xStep
            delY = 0
            logger.info("为-1的%s", yStep)
        ax.plot([cur[0], (cur[0] - delX)], [cur[1], (cur[1] - delY)], c='b')
        cur = ((cur[0] - delX), (cur[1] - delY))
        ax.plot([0, 0.3], [0.2, 0.8],c='r')
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Position Rate')
    plt.ylabel('True Position Rate')
    plt.title('title')
    ax.axis([0, 1, 0, 1])
    plt.show()


if __name__ == "__main__":
    dataArr, classLabels = loadSimpleData()
    weak, aggErrorEst = adaBoostTrainDs(dataArr, classLabels, 3)
    # logger.info(type(aggErrorEst))
    # logger.info("aggErrorEst%s", aggErrorEst)
    # logger.critical("aggErrorEst%s", aggErrorEst.argsort())
    plotRoc(aggErrorEst, classLabels)
    # sortedList = []
    # for row in aggErrorEst:
    #     for i in row:
    #         sortedList.append(i[0, 0])
    # logger.info(np.array(sortedList).argsort())
    # print(sortedList)
    print(weak)
