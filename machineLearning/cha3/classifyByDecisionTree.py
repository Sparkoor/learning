from decisonTree import *
from plotTreeNode import *
import pickle


def classify(inputTree, featLabel, testVec):
    """
    进行分类
    :param inputTree:
    :param featLabel:
    :param testVec:
    :return:
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabel.index(firstStr)
    for key in secondDict.keys():
        # 进行比较
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabel, testVec)
            else:
                classLabel = featLabel[featIndex]
    return classLabel


def storeTree(inputTree, fileName):
    """
    保存决策树
    :param inputTree:
    :param fileName:
    :return:
    """
    with open(fileName, 'w') as fr:
        pickle.dump(inputTree, fr)


def grabTree(fileName):
    """
    读取决策树
    :param fileName:
    :return:
    """
    with open(fileName, 'r') as fr:
        return pickle.load()
