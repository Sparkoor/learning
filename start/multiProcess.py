"""
多线程测试,不能传数据结构啊
"""
from multiprocessing import Pipe, Manager, Queue, pool, Process

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import concurrent
from multiprocessing.managers import BaseManager


# class father(object):
#     def __init__(self):


class Graph(object):
    def __init__(self, graph, dicts):
        # super(Graph, self).__init__(list)
        self.graph = graph
        self.dicts = dicts

    def getGraph(self):
        return self.graph + ':' + self.dicts


class GraphManager(BaseManager):
    pass


def start_manager():
    m = GraphManager()
    m.start()
    return m


def a(a, v):
    print(a)
    print(v.getGraph())


GraphManager.register('Graph', Graph)


def b():
    # g = Graph()
    #
    # # print()
    # s = [('yellow', 1), ('blue', 2), ('yellow', 3), ('yellow', 3), ('blue', 4), ('red', 1)]
    # for i, j in s:
    #     g[i].append(j)
    m = start_manager().Graph("aaa", "aaa")
    with ProcessPoolExecutor(max_workers=2) as executor:
        features = [executor.submit(a, x, m) for x in [1, 2]]
        for fe in concurrent.futures.as_completed(features):
            print(fe.result())


if __name__ == '__main__':
    b()
