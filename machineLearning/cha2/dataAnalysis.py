"""
自动化生成，书中的一些训练数据
需要有四列数据，最后一列为类型，
1，2 ，3分别为对应的类别
"""
import random
import numpy as np


def createDatingSet():
    with open('datingTestSet.txt') as fr:
        dataSet = fr.readlines()
        dataSize = len(dataSet)
        # newDataSet = np.zeros((dataSize, 4))
        index = 0
        newDataSet = []
        for line in dataSet:
            line = line.strip()
            listLine = line.split("\t")
            # print(listLine[-1])
            className = listLine[-1]
            if className == 'didntLike':
                listLine[-1] = '3'
            elif className == 'smallDoses':
                listLine[-1] = '2'
            else:
                listLine[-1] = '1'

            # newDataSet.append(str(listLine))
            s = "\t".join(listLine)
            newDataSet.append(s+"\n")
            index += 1
    # 权限
    with open('kNNdataset.txt', 'w') as f:
        f.writelines(newDataSet)
        print(newDataSet)


if __name__ == "__main__":
    createDatingSet()
