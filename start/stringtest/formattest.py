"""
字符窜的相关测试
u 表示有中文字符串，
r 表示转义符号
"""

# l = 'programming'
# s = u'{}'.format(u" ".join(v for v in l))
# print(s)
#
# num_paths = 10
# num_workers = 4
#
# from itertools import zip_longest
#
#
# # http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
# def grouper(n, iterable, padvalue=None):
#     "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
#     print(*[iter(iterable)])
#     return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)
#
#
# a = [len(list(filter(lambda z: z != None, [y for y in x])))
#      for x in grouper(int(num_paths / num_workers) + 1, range(1, num_paths + 1))]
#
# print(a)
import time


def test(x, name):
    x += 10
    print('{}进程开始睡'.format(name))
    time.sleep(3)
    print(x)
    return x


import concurrent.futures
from concurrent.futures import ProcessPoolExecutor


def exc():
    with ProcessPoolExecutor(max_workers=2) as executor:
        size = executor.map(test, [1, 2, 3], [1, 2, 3])
        # for i in size:
        #     print(i)
        # print("size{}".format(type(size)))


def test2():
    """

    :return:
    """
    with ProcessPoolExecutor(max_workers=1) as executor:
        print('aa')


from collections import defaultdict
# from mpi4py import MPI
#
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
__global_a = None


class Graph(defaultdict):
    def __init__(self):
        super(Graph, self).__init__(list)


def a(x):
    # print('ss')
    g = __global_a
    print("x:{}".format(x))
    print("g:{}".format(g))


def b():
    g = Graph()
    s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('yellow', 3), ('blue', 4), ('red', 1)]
    for i, j in s:
        g[i].append(j)
    print(g)
    __global_a = g
    # __global_a = comm.bcast(__global_a, root=0)
    with ProcessPoolExecutor(max_workers=1) as executor:
        features = [executor.submit(a, x) for x in [1, 2]]
        for feature in concurrent.futures.as_completed(features):
            print(feature.result())


if __name__ == '__main__':
    b()
