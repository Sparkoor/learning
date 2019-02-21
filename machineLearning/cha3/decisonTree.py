from math import log
# 运算符
import operator



def calcShannonEnt(dataset):
    """
    计算香农熵:数据的整齐程度
    数学公式H=-sum(p(x)*log(p(x)))  p(x) 出现x类别的概率
    :param dataset:
    :return:
    """
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannoEnt -= log(prob, 2) * prob
    return shannoEnt


# 创建数据集
def createDataSet():
    """
    创建数据集
    :return: 数据集 分类
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axio, value):
    """
    根据某列的数据进行拆分,根据值进行划分
    :param dataSet: 待划分的数据集
    :param axio: 划分数据集的特征
    :param value: 需要返回的特征的值
    :return: 该特征值下相同值的一个样本，但不带当前特征值
    """
    returnData = []
    for featVec in dataSet:
        if featVec[axio] == value:
            # 去掉该列
            reduceFeatVec = featVec[:axio]
            reduceFeatVec.extend(featVec[axio:])
            returnData.append(reduceFeatVec)
    return returnData


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式,返回特征的位置
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataset)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 取i列的值
        featList = [example[i] for example in dataset]
        uniqueVal = set(featList)
        newEntropy = 0.0
        for value in uniqueVal:
            subDataSet = splitDataSet(dataset, i, value)
            prob = len(subDataSet) / float(len(dataset))
            newEntropy += prob * calcShannonEnt(subDataSet)
        inforGain = baseEntropy - newEntropy
        if inforGain > bestInfoGain:
            bestInfoGain = inforGain
            bestFeature = i
    return bestFeature


def majoritCnt(classList):
    """
    投票方法，谁的票数多，投谁
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 排序的方法
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    # 创建决策树
    :param dataSet:
    :param labels:
    :return:
    """
    # 获取类别
    classList = [example[-1] for example in dataset]
    # 判断是否还剩最后一个类别,类别完全相同，停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的类别
    if len(dataset[0]) == 1:
        return majoritCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataset)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataset]
    uniqueVals = set(featValues)
    # 遍历子节点
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createDataSet(splitDataSet(dataset, bestFeat, value), subLabels)
    return myTree


if __name__ == "__main__":
    dataset, label = createDataSet()
    shannoEnt = calcShannonEnt(dataset)
    print(shannoEnt)
