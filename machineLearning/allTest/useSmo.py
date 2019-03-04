from cha6.SvmOp import *


def loadDataSet():
    dataArr = []
    labelArr = []

    with open(r'D:\workspace\pproject\machineLearning\allTest\data\Iris.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split(',')
            dataArr.append(list(map(float, lineArr[:-1])))
            classStr = lineArr[-1]
            if classStr == 'Iris-setosa':
                labelArr.append(1.0)
            elif classStr == 'Iris-versicolor':
                labelArr.append(2.0)
            else:
                labelArr.append(3.0)
    return dataArr, labelArr


if __name__ == "__main__":
    dataArr, lebalArr = loadDataSet()
    b, alpha = smop(dataArr, lebalArr, 1, 0.00001, 1000)
    w = calcWs(alpha, dataArr, lebalArr)
