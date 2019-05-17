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
