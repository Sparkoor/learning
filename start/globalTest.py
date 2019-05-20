"""
全局变量的用法
"""
__number = 1


def a():
    print(__number)


def b():
    global __number
    __number = 2
    a()


if __name__ == '__main__':
    b()
    a
