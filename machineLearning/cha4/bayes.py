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
        # 将每篇文档返回的新词集添加到该集合中
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


