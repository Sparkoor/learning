"""
python的单例模式
例一、直接在本文件中进行实例化，也是代码最简单的一种
"""


# 例二
def Singleton1(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    print("装饰器")
    return _singleton


# 使用装饰器模式
@Singleton1
class A(object):
    a = 1

    def __init__(self, x=0):
        self.x = x


a1 = A(2)
print(a1.x)
a2 = A(3)
print(id(a1))
print(id(a2))
