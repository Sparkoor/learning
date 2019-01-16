"""
 numpy广播
"""
import numpy as np


def displa():
    a = np.arange(6).reshape(3, 2)
    for x in np.nditer(a):
        print(x, end=',')
    # 输出转置 a和a.T的输出顺序相同,他们在内存长存储的顺序是一样的
    for x in np.nditer(a.T):
        print(x)
    # a.T.copy(order = 'C') 的遍历结果是不同的，那是因为它和前两种的存储方式是不一样的，默认是按行访问。 F 是安列
    for x in np.nditer(a.T.copy(order='C')):
        print(x)


def modify():
    # 从0开始,步长为5,个数 60/5
    a = np.arange(0, 60, 5)
    print(a)
    a = a.reshape(3, 4)
    print('修改前', a)
    for x in np.nditer(a, op_flags=['readwrite']):
        print('循环里面的', x)
        x[...] = x * x

    print('修改后的', a)
    #     使用外部循环
    for x in np.nditer(a, flags=['external_loop'], order='F'):
        print(x, end=',')


# 广播迭代
def brastIter():
    a = np.arange(0, 60, 5).reshape(4, 3)
    print('原始数据\n', a)
    b = np.array([1, 2, 3], dtype=int)
    print('b\n', b)
    # a的列要和b的列个数相同
    for x, y in np.nditer([b, a]):
        print('x=', x, 'y=', y)
        print("%d:%d" % (x, y), end=',')
    print([a, b])


def test3():
    # 广播的性质，可以自动补全
    a = np.array([1, 2, 3, 4])
    b = a - 1
    # 初始化二维到三维数组时一定要注意中括号啊
    c = np.array([
        [2, 3, 4, 5],
        [3, 4, 5, 6]
    ]
    )
    d = c - a
    print(d)
    print(b)


if __name__ == "__main__":
    test3()
