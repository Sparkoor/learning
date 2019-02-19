import random
import numpy as np


def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        dataLines = fr.readlines()
        for line in dataLines:
            lineArr = line.strip().split('\t')
            dataMat.append(lineArr[:-2])
            labelMat.append(lineArr[-1])
    return dataMat, labelMat


def selectJrand(i, m):
    j = i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
    min(w,b) 1/2||w||^2 ①
    s.t w.T*x+b>=1  ②
    
"""


# 数据集 类别标签 常数c 容错率 退出前的最大循环次数
def smoSimple(dataMatIn, classLabels, c, toler, maxIter):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels)
    b = 0
    m, n = np.shape(dataMatrix)
    # 结果就是为了求出alpha 初始值设为0
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 这是超平面
            fxi = float(np.multiply(alphas, labelMat).T) * (dataMatrix * dataMatrix[i, :]) + b
            # 这是满足的条件KT条件中的一个条件
            EI = fxi - float(labelMat[i])
            # 判断alpha是否满足二次规划
            if ((labelMat[i] * EI - 1) < -toler and (alphas[i] < c)) or (
                    (labelMat[i] * EI) > toler and (alphas[i] > 0)):
                # 选择一个和i不同的随机数
                j = selectJrand(i, m)
                fxj = float(np.multiply(alphas, labelMat).T) * (dataMatrix * dataMatrix[j, :]) + b
                EJ = fxj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                # 保存更新前的值
                alphaJold = alphas[j].copy()
                # 保证alpha在0-c之间 如果是就进行下一次循环
                # 满足条件 alpha[i]*lable[i]+aplha[j]*label[j]=c  label只能取1 -1
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(c, c + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - c)
                    H = min(c, alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T \
                      - dataMatrix[j, :] * dataMatrix[j, :]
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * [EI - EJ] / eta
                alphas[i] += clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.000001):
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaIold - alphas[j])
                b1 = b - EI - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * \
                     dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - EJ - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2
                alphaPairsChanged += 1
                print('更新一次')

            if (alphaPairsChanged == 0):
                iter += 1
            else:
                iter = 0
        return b, alphas
