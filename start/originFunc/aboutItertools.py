"""
python的内建模块itertools
"""
import itertools


def test1():
    """
    无限的迭代
    :return:
    """
    natuals = itertools.count(1)
    for i in natuals:
        print(i)


def test2():
    """
     重复迭代
    """
    cs = itertools.cycle("abc")


def test3():
    natuals = itertools.count(1)
    # takewhile 可以截取有限序列
    ns = itertools.takewhile(lambda x: x < 10, natuals)
    print(list(ns))


def test4():
    # chain把两个迭代对象串联起来
    for c in itertools.chain('abc', 'xyz'):
        print(c)


def test4():
    for key, group in itertools.groupby("aaabbmmmddeew"):
        print(list(group))


def test5():
    a = [[1, 2, 3],
         [1, 3, 5]]
    # a前的*表示竖着，[]*3表示循环几遍
    for i in itertools.zip_longest(*[iter(a)]*4, fillvalue=None):
        print(i)


if __name__ == '__main__':
    test5()
