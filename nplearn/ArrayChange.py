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


# 修改数组维度,用于模仿广播对象，返回一个对象，
# 该对象封装了将一个数组广播到另一个数组的结果
def broad_cast():
    x = np.array([[1], [2], [3]])
    y = np.array([4, 5, 6])
    # 对y广播x 这个对象拥有iterator属性，基于自身组件的迭代器
    b = np.broadcast(x, y)
    print('对y广播x：')
    # b存在的形似应该是key，value的形式
    r, c = b.iters
    print(b)
    print(next(r), next(c))
    print(next(r), next(c))
    print(next(r), next(c))
    print(next(r), next(c))
    print(b.shape)

    b = np.broadcast(x, y)
    # 手动使用广播将x与y相加
    c = np.empty(b.shape)

    print(c.shape)
    print(c)
    c.flat = [u + v for (u, v) in b]
    print(c)

    print(x + y)


# numpy.broadcast_to 函数将数组广播到新形状。
# 它在原始数组上返回只读视图。 它通常不连续。
# 如果新形状不符合 NumPy 的广播规则，该函数可能会抛出ValueError。
def broast_cast_to():
    a = np.arange(4).reshape(1, 4)
    print(a)
    print(np.broadcast_to(a, (4, 4)))
    x = np.array(([1, 2], [3, 4]))
    print(x)
    y = np.expand_dims(x, axis=0)
    print(y)
    print(x.shape, y.shape)
    # 维数 TODO:这个略难
    print(x.ndim, y.ndim)
    y = np.expand_dims(x, axis=1)
    print(y)


# numpy.squeeze 函数从给定数组的形状中删除一维的条目
def squeeze_test():
    x = np.arange(9).reshape(1, 3, 3)
    print(x)
    y = np.squeeze(x)
    print(y)
    m = np.arange(9).reshape(3, 3)
    z = np.squeeze(m)
    print(z)


if __name__ == '__main__':
    squeeze_test()
