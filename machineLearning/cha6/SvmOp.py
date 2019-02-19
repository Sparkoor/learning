import numpy as np


class optStruct:
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
    fxk = float(np.multiply(os.alphas, os.labelMat).T * (os.X * os.X[k, :].T)) + os.b
    Ek = fxk - float(os.labelMat)
    return Ek


# 查找最大步长
def selectJ(i, os, Ei):
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
                maxK = deltaE
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJ(i, os.m)
        Ek = calcEk(os, j)
    return j, Ek


def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]


def innerL(i, os):
    Ei = calcEk(i, os)
    if ((os.labelMat[i] * Ei < -os.tol) and (os.alphas[i] < os.c)) or (
            (os.labelMat[i] * Ei > os.tol) and (os.alphas[i] > 0)):
        j,Ej=selectJ(i,os,Ei)

