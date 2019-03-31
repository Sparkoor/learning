def test0():
    return [1, 5]


def test1(a, b):
    v = []
    for i in range(a, b):
        v.append(i)
    return v


def test2(a):
    return a ** 2


if __name__ == '__main__':
    print(list(map(test2, test1(1, 4))))
    a = []
    a.append([1])
    a.extend([2])
    print(a)
