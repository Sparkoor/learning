"""
NumPy 通常与 SciPy（Scientific Python）和 Matplotlib（绘图库）一起使用，
这种组合广泛用于替代 MatLab，是一个强大的科学计算环境，有助于我们通过 Python 学习数据科学或者机器学习。
SciPy 是一个开源的 Python 算法库和数学工具包。
SciPy 包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、
信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。
Matplotlib 是 Python 编程语言及其数值数学扩展包 NumPy 的可视化操作界面。
它为利用通用的图形用户界面工具包，如 Tkinter, wxPython, Qt 或 GTK+ 向应用程序嵌入式绘图提供了应用程序接口（API）。
线性代数、傅里叶变换、随机数生成等功能
"""
import numpy as np


def func():
    # y_pred = np.zeros(A_pred.shape)
    # # 这句话的意思
    # y_pred[0, np.nonzero(A_pred > 0.5)[1]] = 1.0
    a = np.random.randn(4, 3)
    print(a)
    a_l = a.shape
    print(a_l)
    a_l2 = a.shape[0]
    print(a_l2)
    ar = np.array([
        [1, 2, 3],
        [2, 3, 4]
    ])
    # 调整数组大小
    al = ar.reshape(3, 2)
    print(ar)
    # 标记为二维数组
    a = np.array([1, 2, 3, 4], ndmin=2)
    # 数据的类型
    b = np.array([1, 2, 3], dtype=complex)
    # 创建一个指定形状的类型的且未初始化的数组，数组有值
    a = np.empty([2, 3], dtype=float, order='C')
    print(a)
    # 创建一个以0为填充的数组
    a = np.zeros([2, 3])
    # 创建一个指定形状以1为填充
    a = np.ones([2, 3])
    # 把某种数据结构的数据转化成数组
    l = [1, 2, 3]
    d = np.asarray(l)
    # 以输入流的形式读入
    a = 'hello world'
    s = np.frombuffer(a)
    # 从迭代器中获取数据
    s = [1, 2, 3, 4]
    i = iter(s)
    d = np.fromiter(i)
    # 创建有规律的数组,start stop step
    x = np.arange(1, 5, 1)
    # 创建一个一维数组，等差数列
    x = np.linspace(1, 2, num=10)
    # 创建一个等差数列,log的底数默认是10
    x = np.logspace(1, 2, num=5, base=10)
    # numpy切片和索引,从2到7，间隔为2
    a = np.arange(10)
    s = slice(2, 7, 2)
    print(a[s])
    # 从索引2开始到索引7停止,间隔为2
    a[2:7:2]
    # 索引2后的所有项
    a[2:]
    # 多维数组
    a = np.array([
        [1, 2, 3, 4],
        [3, 4, 5, 6],
        [2, 3, 4, 5]
    ])
    # 1表示的是行索引
    a[1:]
    # 使用省略号...
    # 第二行
    a[..., 1]
    # 还是第二行
    a[1, ...]
    # 第二行及以后
    a[..., 1:]


# numpy高级索引 NumPy 比一般的 Python 序列提供更多的索引方式。
# 除了之前看到的用整数和切片的索引外，数组可以由整数数组索引、布尔索引及花式索引
def gaoji():
    x = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11]
    ])
    # 前面的是行的索引,后面是列的索引,去对应就好了
    rows = np.array([[0, 0], [3, 3]])
    cols = np.array([[0, 2], [0, 2]])
    y = x[rows, cols]
    print(y)
    # [[这里对应的是第一个坐标 行],[对应着第二个坐标 列]]
    y = x[[0, 1, 2], [0, 1, 0]]
    print('y', y)


def qiepian():
    x = np.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11]
    ])
    #     切片加...
    # [第一个切片表示从索引为1的到索引为3的,第二个切片也是]
    a = x[1:3, 1:3]
    # 和上一个功能一样,
    a = x[1:3, [1, 2]]
    # 所有的行,列从索引为1的开始
    a = x[..., 1:]
    #     bool索引,通过条件筛选
    print(x[x > 5])
    #     ~取补运算 np.nan 不能和复数在同一个列表里
    a = np.array([1, np.nan, 5])
    print(a[~np.isnan()])
    print(a[np.iscomplex()])


def huasi():
    x = np.arange(32).reshape(8, 4)
    print(x)
    print(x[[4, 2, 1, 7]])
    #     传入倒叙索引
    print(x[[-4, -2, -1, -7]])
    #     传入多个索引数组,这个是两个数组进行交叉相乘
    print(x[np.ix_([1, 5, 7, 2], [0, 3, 1, 2])])


if __name__ == "__main__":
    huasi()
