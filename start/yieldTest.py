"""
关于yield的测试
yield生成器，它提供了工具在需要的时候才产生结果，
在每个结构之间挂起和继续它们的状态
它返回按需产生结果的一个对象，而不是构建一个结果列表
"""


def test(i):
    for n in range(1, 5):
        # print(i)
        i = i + 2
        print("迭代器开始")
        yield i
        print("第一个迭代器结束")


# yield 生成迭代器，相当于随用随取
for i in test(1):
    print(i)

print("*" * 20)


def fab1(max):
    n, a, b = 0, 0, 1
    while n < max:
        a, b = b, a + b
        n = n + 1


import os


def traversalDir(rootDir):
    for i, lists in enumerate(os.listdir(rootDir)):
        path = os.path.join(rootDir, lists)
        if os.path.isfile(path):
            pass
        if os.path.isdir(path):
            traversalDir(path)


class loadDir():
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for p in os.listdir(self.path):
            pa = os.path.join(self.path, pa)
            if os.path.isdir(pa):
                yield pa


class loadFile():
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        folders = os.listdir(self.path)
        for folder in folders:
            cag = folder.split(os.sep)[-1]
            for file in os.listdir(folder):
                yield cag, file
