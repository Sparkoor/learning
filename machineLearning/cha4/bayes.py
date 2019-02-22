"""
伯努利模型实现：不考虑词在文档中出现的次数，只考虑出不出现
"""
import numpy as np
from math import *


def loadDataSet(fileName):
    """
    读取数据
    :param fileName:
    :return: 样本 样本标签
    """
    datalist = []
    classVec = []
    return datalist, classVec


def createVocabList(dataSet):
    """
    创建词集，在所有文章里出现，但不重复
    :param dataSet:
    :return:
    """
    vocabSet = ([])
    # dataSet 是矩阵
    for document in dataSet:
        # 将每篇文档返回的新词集添加到该集合中 |是并集的意思
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    把文章转换成词向量，词在词集中是1 不在是0
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(word, "该词不在词集中")
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯训练函数
    :param trainMatrix: 词集向量
    :param trainCategory:
    :return:
    """
    numTrainDoc = len(trainMatrix)
    numTrainWord = len(trainMatrix[0])
    # p(c) 计算目标文章所占的比例
    pAbusive = sum(trainCategory) / float(numTrainDoc)
    p0Num = np.ones(numTrainWord)
    p1Num = np.ones(numTrainWord)
    p1Denmo = 2.0
    p0Denmo = 2.0
    for i in range(numTrainDoc):
        if trainCategory[i] == 1:
            # 出现关键词的个数，对应位置相加，为了计算该词在
            p1Num += trainMatrix[i]
            # 该类型文章的总字数
            p1Denmo += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denmo += sum(trainMatrix[i])
    # p(w|c) 使用log防止数太小，造成下溢
    p1Vect = log(p1Num / p1Denmo)
    p0Vec = log(p0Num / p0Denmo)
    return p1Vect, p0Vec, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    p(x1,y1)>p(x2,y2) 使用的是这个算法 p(w,c)=p(w|c)p(c)
    :param vec2Classify: 需要分类的样本
    :param p0Vec: p(w|C0)
    :param p1Vec: p(w|C1)
    :param pClass1: p(C1)
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    listOPosts, listClass = loadDataSet("dataset.txt")
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    # 向量化样本
    for postInDoc in listOPosts:
        # 基于词集导出这个样本的词向量
        trainMat.append(setOfWords2Vec(myVocabList, postInDoc))
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClass))
    # 分完词的文本
    testEntry = ['love', 'my', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    p = classifyNB(thisDoc, p0V, p1V, pAb)
    if p == 1:
        print("是")
    else:
        print("不是")
