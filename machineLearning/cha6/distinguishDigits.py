"""
手写识别数字
"""
from svmWithKernel import *


def loadImages(dirName):
    """
    加载图片
    :param dirName:
    :return:
    """
    from os import *
    hwLabel = []
    trainFileList = listdir(dirName)
    m = len(trainFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabel.append(-1)
        else:
            hwLabel.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabel


def testDigits(kTup=('rbf', 10)):
    """
    测试函数
    :param kTup:
    :return:
    """
    dataArr, labelArr = loadImages("trainingDigits")
    logger.warning(type(dataArr))
    b, alpha = smop2(dataArr, labelArr, 200, 0.00001, 1000, kTup)
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).A
    # 找出非零的alpha
    svInd = np.nonzero(alpha > 0)[0]
    logger.info(svInd)
    # 选取支持向量
    svs = dataMat[svInd]
    labelSV = labelMat[svInd]
    logger.info("支持向量的数量%d" % (svs.shape[0]))
    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs, dataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(labelSV, alpha[svInd]) + b
        if np.sign(predict) != np.sign(labelSV[i]):
            errorCount += 1


def img2vector(imgPath):
    """
    加载图像矩阵
    :param imgPath:
    :return:
    """
    returnVect = np.zeros((1, 1024))
    with open(imgPath) as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def img2vector2(imgPath):
    """
    构成向量的不同想法
    :param imgPath:
    :return:
    """
    returnList = []
    with open(imgPath) as fr:
        imgStr = fr.readlines()
        for str in imgStr:
            returnList.extend(str)
    return np.mat(returnList)
