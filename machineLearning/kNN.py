# 科学计算包
from numpy import *
# 运算符模块
import operator


def createDataSet():
    # 数据集，特征
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 特征所对应的类型
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 分类的主函数,k-近邻分类器
def classify0(inX, dataSet, labels, k):
    # 数据集的列
    dataSetSize = dataSet.shape[0]
    # 矩阵运算，计算欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    # 平方
    sqDiffMat = diffMat ** 2
    # 矩阵项求和
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 取索引值
    sortedDistIndicies = distances.argsort()
    # 分类总计
    classCount = {}
    for i in range(k):
        # 获取类型名称
        voteIlabel = labels[sortedDistIndicies[i]]
        # 放入元组中
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序
    print("classCount", classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print("sortedClassCount", sortedClassCount)
    return sortedClassCount[0][0]


# 将文本记录转化成可使用的数据格式 numpy格式
def file2matrix(filename):
    fr = open(filename, 'r')
    arrayOLine = fr.readlines()
    numberOfLines = len(arrayOLine)
    # 具有三个特征
    returnMat = zeros((numberOfLines, 3))
    # 初始化类型
    classLabelVector = []
    index = 0
    for line in arrayOLine:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 数据的最后一列为对应的类别
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


if __name__ == '__main__':
    group, labels = createDataSet()
    aa = classify0([1, 1], group, labels, 3)
    print(aa)
