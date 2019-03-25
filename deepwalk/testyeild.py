def test1(va):
    return va * 4


g = 2


def test2():
    for i in range(6):
        yield i * i


def fun1(list):
    return zip(iter(list))


def fun2(a):
    return a ** 2


if __name__ == '__main__':
    a = test1(3)
    b = test2()
    print(a)
    print(next(b))
    print(next(b))
    print(next(b))
    print(next(b))
    print(next(b))

    c = map(fun2, fun1, [2, 3])
    print(c)
