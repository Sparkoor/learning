from svmMLiA import *
import matplotlib.pyplot as plt
from commonUtils.Loggings import *

# Logger需要带括号
logger = Logger().getLogger()


class optStruct:
    """
    用于存储变量
    """

    def __init__(self, dataMatIn, classLabels, c, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.c = c
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))


# 计算误差，使用了最优化的某个定理
def calcEk(os, k):
    """
    计算过的是代价函数， y=w.T*x+bb w=
    :param os:
    :param k:
    :return:
    """
    # print(np.multiply(os.alphas.T, os.labelMat.T))
    fxk = float(np.multiply(os.alphas, os.labelMat).T * (os.X * os.X[k, :].T)) + os.b
    # 这里只是为了求出该alpha的差值
    Ek = fxk - float(os.labelMat[k])
    return Ek


# 查找最大步长
def selectJ(i, os, Ei):
    """
    用于选择第二个alpha保证每次优化中采用最大步长
    :param i:
    :param os:
    :param Ei:
    :return:
    """
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    os.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(os.eCache[:, 0].A)[0]
    if (len(validEcacheList) > 1):
        # 对应非零alpha的索引
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(os, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                # 这里错了
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, os.m)
        Ek = calcEk(os, j)
    return j, Ek


def updateEk(os, k):
    """
    更新误差的缓存
    :param os:
    :param k:
    :return:
    """
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]


def innerL(i, os):
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
        eta = 2.0 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T - os.X[j, :] * os.X[j, :].T
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
        b1 = os.b - Ei - os.labelMat[i] * (os.alphas[i] - alphaIold) * os.X[i, :] * os.X[i, :].T - os.labelMat[j] * (
                os.X[j, :] * os.X[j, :].T) * (os.alphas[j] - alphaJold)
        b2 = os.b - Ej - os.labelMat[i] * os.X[i, :] * os.X[j, :].T * (os.alphas[i] - alphaIold) - os.labelMat[j] \
             * os.X[j, :] * os.X[j, :].T * (os.alphas[j] - alphaJold)
        if 0 < os.alphas[i] and os.c > os.alphas[i]:
            os.b = b1
        elif 0 < os.alphas[j] and os.c > os.alphas[j]:
            os.b = b2
        else:
            os.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smop(dataMatIn, classLabels, c, toler, maxIter, kTup=('lin', 0)):
    """

    :param dataMatIn:
    :param classLabels:
    :param c:
    :param toler:
    :param maxIter:
    :param kTup: 数据不能被改变
    :return:
    """
    os = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), c, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter and alphaPairsChanged > 0) or entireSet:
        alphaPairsChanged = 0
        if entireSet:
            # 所有的点遍历一遍
            for i in range(os.m):
                alphaPairsChanged += innerL(i, os)
                iter += 1
                # print("内循环结束")
        else:
            # 没整明白
            nonBoundIs = np.nonzero((os.alphas.A > 0) * (os.alphas.A < c))[0]
            print(nonBoundIs)
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, os)
                # print("nonboundIn")
                iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
            # print("iter num %d" % iter)
    return os.b, os.alphas


def calcWs(alphas, dataArr, classLabels):
    # 计算w的值
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = X.shape
    w = np.zeros((n, 1))
    for i in range(m):
        # 对应位置相加
        w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


def classifySvmOp():
    dataSet, classLabels = loadDataSet('testSet.txt')
    b, alpha = smop(dataSet, classLabels, 1, 0.00001, 1000)
    w = calcWs(alpha, dataSet, classLabels)
    xcord1 = []
    xcord2 = []
    ycord1 = []
    ycord2 = []
    for i in range(len(classLabels)):
        if int(classLabels[i]) == -1:
            xcord1.append(dataSet[i][0])
            ycord1.append(dataSet[i][1])
        else:
            xcord2.append(dataSet[i][0])
            ycord2.append(dataSet[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3, 10, 0.1)
    y = (-w[1, 0] * x + b[0, 0]) / w[0, 0]
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    classifySvmOp()
