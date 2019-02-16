"""
根据信息增益实现决策树
"""
import math
# 不会用这个处理数据
import pandas as pd
import numpy as np
import re


# 加载数据
def loadData():
    with open('./dataset2.0.txt', 'r+', encoding='utf-8') as f:
        # l = f.readline()
        # # 数据分割
        # label = re.split(r'\s+', l.strip())
        lines = f.readlines()
        data = []
        for line in lines:
            data.append(re.split(r'\s+', line.strip()))
    return data


# 得到一堆数据，进行拆分
def prepareData(dataset):
    col, row = dataset.shape
    if row == 0:
        return
        # label = dataset[0]
    # key-value形式的数组，存放节点加叶子或子节点
    datadicts = {}
    # 去掉一列的矩阵，该矩阵用来迭代
    removeOneRow = []
    # 用来计算好瓜坏瓜的比例,这应该是个错误数据用不到
    removeTitle = []
    for line in dataset.T:
        key = line[0]
        removeTitle.append(line[1:])
        removeOneRow.append(line[:])
        # 这是用来防止出现重复的叶子或节点
        values = set(line[1:])
        datadicts[key] = values
        # print(line)
    # 访问二维数组的列还可以这样写,感觉有点多此一举
    # for i in range(row):
    #     labelList = [example[i] for example in dataset]
    #     print(labelList)
    # 返回一个map以及去掉一列的矩阵
    return datadicts, np.array(removeOneRow[1:]).T, np.array(removeTitle).T


# 去掉一列的函数
def removeOneRow(dataset):
    return dataset[:, 1:]



# 获取子集的所有样本
def getChildSet(k, dataset):
    childset = []
    for line in dataset:
        if np.unique(line, k):
            childset.append(line)
    return childset


# 计算某个属性可得子集的正比例数
def maxNumLeaf(childdict):
    childdict = {}
    maxnum = 0
    for k, v in childdict:
        childdict[k] = v
        maxnum = len(v)
    return


def treeGenerate(node, attr, data):
    """
    :param node: 树
    :param attr: 一个分支
    :param data: 矩阵
    :return:
    """
    if attr.values() in ['是', '否']:
        node[attr.keys()[0]] = attr.values()
    # 生成节点node
    for k, v in attr:
        node[k] = v


if __name__ == '__main__':
    b = loadData()
    a, c, d = prepareData(np.array(b))
    print(a)
    print(c)
    print(d)
    node = {}
    treeGenerate(node, a, c)
