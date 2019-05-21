from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from collections import defaultdict

__global = 1


def a(x, g):
    # g = __global
    print("x:{}".format(x))
    print("g:{}".format(g))
    return x


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)


def b():
    g = Graph()
    s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('yellow', 3), ('blue', 4), ('red', 1)]
    for i, j in s:
        g[i].append(j)
    print(['1', g.items()])
    lis = Manager().list([{'a': g.items()}])
    with ProcessPoolExecutor(max_workers=1) as executor:
        for s in [executor.submit(a, x, lis) for x in [1, 2]]:
            print(s.result())


if __name__ == '__main__':
    [(10207, [8324, 9798]), (9056, [8775, 8859, 8868, 9027]), (8913, [8859]), (9386, [8859]), (9477, [8859, 9133]), (
    9757, [8859]), (9852, [8859, 9091]), (10080, [8859]), (9767, [9188]), (9993, [9188, 9823]), (9805, [9355]), (
    10008, [9997])]
    print(aa)
