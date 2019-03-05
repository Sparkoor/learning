from numpy.random import RandomState

if __name__ == '__main__':
    #
    rdm = RandomState(1)
    dataSet_size = 128
    X = rdm.rand(dataSet_size, 2)
    # note:int强转bool类型时True为1，False为0
    Y = [[int(x1 + x2)] for (x1, x2) in X]
    Y2 = [[int(x1 + x2 < 1)] for (x1, x2) in X]
    Y3 = [[x1 + x2 < 1] for (x1, x2) in X]
    print(Y)
    print(Y2)
    print(Y3)
    print(int(True))
    print(int(False))
