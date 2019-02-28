"""
分析
"""
import numpy as np
from commonUtils.Loggings import *
import matplotlib.pyplot as plt

logger = Logger().getLogger()


def loadDataSet():
    # 学术硕士
    masterofArts = []
    # 专业硕士
    professionalMaster = []
    with open('grade.txt') as fr:
        gradeLines = fr.readlines()
        logger.info("总体人数%s", len(gradeLines))
        for line in gradeLines:
            stud = line.strip().split()
            if stud[2] == '085212':
                professionalMaster.append(stud)
            else:
                masterofArts.append(stud)
        logger.info("学术人数%s", len(masterofArts))
        logger.info("专硕人数%s", len(professionalMaster))
    return masterofArts, professionalMaster


def plotGrade():
    xue, zhuan = loadDataSet()
    xueMat = np.mat(xue)
    zhuanMat = np.mat(zhuan)
    # note:转化成list
    xueGradeArr = [int(i[0, 0]) for i in xueMat[:, -1]]
    zhuanGradeArr = [int(i[0, 0]) for i in zhuanMat[:, -1]]
    sorted(xueGradeArr)
    sorted(zhuanGradeArr)
    numx = {}
    numz = {}
    for i in range(10):
        numx[(i * 50)] = 0
        numz[(i * 50)] = 0
    for i in xueGradeArr:
        index = int(i / 50) * 50
        numx[index] += 1
    for i in zhuanGradeArr:
        index = int(i / 50) * 50
        numz[index] += 1
    x1 = numx.keys()
    x2 = numz.keys()
    y1 = numx.values()
    y2 = numz.values()
    width = 25
    plt.bar(x1, y1, width=width, label='boy', fc='y')
    x2 = [i + width for i in x2]
    plt.bar(x2, y2, width=width, label='boy', fc='r')
    plt.show()


# for i in xueGradeArr:

# plt.show()


if __name__ == "__main__":
    xue, zhuan = loadDataSet()
    xueMat = np.mat(xue)
    zhuanMat = np.mat(zhuan)
    xueGradeArr = [int(i[0, 0]) for i in xueMat[:, -1]]
    logger.error(sorted(xueGradeArr, reverse=True))
    logger.error(len(xueGradeArr))
    numx = {}
    numz = {}
    for i in range(10):
        numx[(i + 1) * 50] = 0
        numz[(i + 1) * 50] = 0
    print(numx)
    # # logger.info(xueMat)
    # index = 0
    # for i in zhuanMat[:, -1]:
    #     if int(i[0, 0]) > 351:
    #         index += 1
    # logger.info(index)
    plotGrade()
