"""
实现垃圾邮件的过滤
"""
from bayes import *
import random


def bagOfWords2VecMN(vocabList, inputSet):
    """
    朴素贝叶斯词袋模型
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def loadData():
    return 'This book is the best book on python or M.L. I have ever laid eyes upon'


import re


def splitDataSet():
    """
    字符串分割
    :return:
    """
    dataStr = loadData()
    dataStr.split()
    # 使用正则公式切分
    regRex = re.compile('\\W*')
    listofword = regRex.split(dataStr)
    listofword = [word.lower() for word in listofword if len(word) > 0]
    return listofword


def apamTest():
    docList = []
    classList = []
    fullText = []
    # 加载文件的个数
    for i in range(5):
        wordList = splitDataSet()
        docList.append(wordList)
        classList.append(1)
        fullText.extend(wordList)
        wordList = splitDataSet()


import feedparser
import operator


def calcMostFreq(vocabList, fullText):
    """
    获取出现在词集里面的，文章中的高频词
    :param vocabList:
    :param fullText:
    :return:
    """
    freqDict = {}
    for token in vocabList:
        if token in freqDict.keys():
            freqDict[token] = fullText.count(token)
    sortedWordList = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedWordList[:30]


def localWords(feed1, feed0):
    """
    使用朴素贝叶斯进行分类
    :param feed1: 类别一
    :param feed0: 类别二
    :return:
    """
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed0), len(feed1))
    for i in range(minLen):
        """
        加载测试数据
        """
    vocabList = createVocabList(docList)
    # 返回的是字典
    top30Words = calcMostFreq(vocabList, fullText)
    for pariw in top30Words:
        if pariw in vocabList:
            vocabList.remove(pariw[0])
    trainingSet = range(2 * minLen)
    testSet = []
    # 随机产生
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = []
    trainLabel = []
    for docIndex in trainingSet:
        # 放入的是向量
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainLabel.append(classList[docIndex])
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(trainLabel))
    errorCount = 0
    for docIndex in testSet:
        wordVec = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(wordVec, p0V, p1V, pAb) != 1:
            errorCount += 1
    print("错误率", errorCount / len(testSet))
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    """
    最具表征性的词汇显示函数
    :param ny:
    :param sf:
    :return:
    """
    vocabList, p0V, p1V = localWords(ny, sf)
    topNy = []
    topSf = []
    for i in range(len(p0V)):
        if p0V[i] > -0.6: topSf.append((vocabList[i], p0V[i]))
        if p1V[i] > -0.6: topNy.append((vocabList[i], p1V[i]))
    # 排序
    sortedF = sorted(topNy, key=lambda pair: pair[1], reverse=True)
