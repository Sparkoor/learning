"""
处理数组的几种类别
修改数组形状
翻转数组
修改数组维度
连接数组
分割数组
数组元素的添加与删除
"""
import numpy as np


def test1():
    a = np.arange(9).reshape(3, 3)
    for row in a:
        print(row)
    # flat数组元素迭代器
    for element in a.flat:
        print(element)
    # 展开数组,返回一份拷贝数据,不会对原数据造成影响 C:按行 F:按列 A:原顺序 K:元素内存中出现的顺序
    b = a.flatten(order='F')
    # 展开数组,修改会影响原始数组
    c = a.ravel(order='F')


def test2():
    a = np.arange(12).reshape(3, 4)
    # transpose 对换数组维数 类似于 a.T
    b = np.transpose(a)
    # rollaxis函数向后滚动特定的轴到一个特定的位置,这是一个三维数组
    c = np.arange(8).reshape(2, 2, 2)
    # 将轴2滚动到轴0
    print(np.rollaxis(a, 2))
    # 将轴0滚动到轴1
    print(np.rollaxis(a, 2, 1))
    # 对换轴
    print(np.swapaxes(a, 2, 0))


def test3():
    pass
