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


if __name__ == "__main__":
    test1()
