# import operator
#
# classCount = {'a': 1, 'b': 3}
# print(type(classCount))
# 排序
# aa = sorted(classCount.items(), key=operator.itemgetter(0), reverse=True)
# print(aa)
# print(aa[0][0])
# print(aa[1][0])
# print(aa[0][1])

import numpy as np
from math import log


def fun1():
    """
    全局变量
    :return:
    """
    fun1.a = 1
    print(fun1.a)


def fun2():
    fun1.a = 2
    print(fun1.a)


def test1():
    """
    取矩阵的列
    """
    arrayMat = np.array([[1, 2, 4],
                         [3, 4, 5],
                         [1, 2, 4]])
    row = [r[1] for r in arrayMat]
    print(row)


def Ent(datas):
    entSum = 0
    for d in datas:
        entSum -= log(d, 2) * d
    return entSum


if __name__ == "__main__":
    p = [7 / 9, 2 / 9]
    dd = Ent(p)
    a1 = [3 / 4, 1 / 4]
    aa1 = Ent(a1)
    a2 = [3 / 4, 1 / 4]
    aa2 = Ent(a2)
    a3 = [1]
    aa3 = Ent(a3)
    p = dd - (4 / 9) * aa1 - (4 / 9) * aa2 - (1 / 9) * aa3
    print(p)
