"""
使用径向基核函数
"""
from SvmOp import *


def kernelTrans(X, A, kTup):
    """
    核转换函数
    :param X: 矩阵
    :param A:列
    :param kTup: 包含核信息的元组
    :return:
    """
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == "rbf":
        for j in range(m):
            dataRow = X[j, :] - A
            K[j] = dataRow * dataRow.T
        K = np.exp(K / -1 * kTup[1] ** 2)
    else:
        logger.error("输入错误")
        raise NameError('数据名称错误')
    return K


class optStruct2:
    def __init__(self, dataMatIn, classLabels, c, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.c = c
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 第一列给出的是有效标志位，第二个是给出的实际E值
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.k = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.k[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


def calcEk2(os, k):
    """
    计算过的是代价函数， y=w.T*x+bb w=
    :param os:
    :param k:
    :return:
    """
    fxk = float(np.multiply(os.alphas, os.labelMat).T * os.K) + os.b
    # 这里只是为了求出该alpha的差值
    Ek = fxk - float(os.labelMat[k])
    return Ek


def innerL2(i, os):
    """
    完整的smo算法内循环
    :param i:
    :param os:
    :return:
    """
    Ei = calcEk(os, i)
    # tol正负间隔 Ei误差 alpha满足c>α>0
    if ((os.labelMat[i] * Ei < -os.tol) and (os.alphas[i] < os.c)) or (
            (os.labelMat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
        # 选出最优步长
        j, Ej = selectJ(i, os, Ei)
        # 保存更新前的数据
        alphaIold = os.alphas[i].copy()
        alphaJold = os.alphas[j].copy()
        # 计算alpha[j]的最大最小值 根据的公式是 alpha[i]*y[i]+alpha[j]*y[j]=k
        if os.labelMat[i] != os.labelMat[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.c, os.c + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.c)
            H = min(os.c, os.alphas[j] + os.alphas[i])
        if L == H:
            # print('L==H')
            return 0
        # 用来计算新的alpha2
        eta = 2.0 * os.k[i, j] - os.k[i, i] - os.k[j, j]
        if eta >= 0:
            # print("eta>=0")
            return 0
        os.alphas[j] -= os.labelMat[j] * (Ei - Ej) / eta
        # 保证alpha在一定范围内
        os.alphas[j] = clipAlpha(os.alphas[j], H, L)
        updateEk(os, j)
        if abs(os.alphas[j] - alphaJold) < 0.00001:
            logger.info("j not moving")
            return 0
        # 有公式
        os.alphas[i] += os.labelMat[j] * os.labelMat[i] * (alphaJold - os.alphas[j])
        b1 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.k[i, i] - os.labelMat[j] * os.k[i, j] * (
                os.alphas[j] - alphaJold)
        b2 = os.b - Ej - os.labelMat[i] * os.k[i, j] * (os.alphas[i] - alphaIold) - os.labelMat[j] \
             * os.k[j, j] * (os.alphas[j] - alphaJold)
        if 0 < os.alphas[i] and os.c > os.alphas[i]:
            os.b = b1
        elif 0 < os.alphas[j] and os.c > os.alphas[j]:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smop2(dataMatIn, classLabels, c, toler, maxIter, kTup=('lin', 0)):
    """
    smo算法的外循环
    :param dataMatIn:
    :param classLabels:
    :param c:
    :param toler:
    :param maxIter:
    :param kTup: 数据不能被改变
    :return:
    """
    os = optStruct2(np.mat(dataMatIn), np.mat(classLabels).transpose(), c, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter and alphaPairsChanged > 0) or entireSet:
        alphaPairsChanged = 0
        if entireSet:
            # 所有的点遍历一遍
            for i in range(os.m):
                alphaPairsChanged += innerL2(i, os)
                iter += 1
                # print("内循环结束")
        else:
            # 没整明白 当nonzero(x) x是行向量时要使用[1]表示列x是列时使用[0]
            # note:进行过滤
            nonBoundIs = np.nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            # logging.warning((os.alphas.A > 0) * (os.alphas.A < c))
            for i in nonBoundIs:
                alphaPairsChanged += innerL2(i, os)
                # print("nonboundIn")
                iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
            # print("iter num %d" % iter)
    return os.b, os.alphas


def testRbf(k1=1.3):
    """
    测试核函数
    :param k1:
    :return:
    """
    dataArr, labelArr = loadDataSet("testSetRBF.txt")
    b, alphas = smop2(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    # 这是获取什么值啊:取大于0的alpha做什么 答：取支持向量
    svInd = np.nonzero(alphas.A > 0)[0]
    svs = dataMat[svInd]
    labelSV = labelMat[svInd]
    logger.info("there are %d support vector" % svs.shape[0])
    m, n = dataMat.shape
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(svs, dataMat[i, :], ('rbf', k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1
    logger.warning("errorRate: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet("testSetRBF2.txt")
    errorCount = 0
    dataMat = np.mat(dataMat)
    labelMat = np.mat(labelArr).T
    # 选择支持向量
    labelSV = labelMat[svInd]
    m, n = dataMat.shape
    for i in range(m):
        # 计算核函数值
        kernelEval = kernelTrans(svs, dataMat[i, :], ('rbf', k1))
        # 预测值
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        # 使用sign函数分类
        if np.sign(predict) != np.sign(labelArr[i]):
            errorCount += 1

    #
    logger.warning("错误率为%f" % (float(errorCount) / m))


def plotSmo(dataArr, labelArr, alpha, svs, b):
    """
    画出图像
    :param dataArr:
    :param labelArr:
    :param alpha:
    :param svs: 这是支持向量
    :param b:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(dataArr[1]):
        if labelArr[i] == -1:
            xcord1.append(dataArr[i][0])
            ycord1.append(dataArr[i][1])
        else:
            xcord2.append(dataArr[i][0])
            ycord2.append(dataArr[i][1])
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    plt.show()


if __name__ == "__main__":
    testRbf()
