from collections import defaultdict
from time import time

from six import iterkeys


class Node(object):
    """
    时序网络节点
    """

    def __init__(self, name, time):
        """
        节点名称和时间
        :param name:
        :param time:
        """
        self.name = name
        self.time = time


class TemporalGraph(defaultdict):
    """
    时序网络
    """

    def __init__(self):
        super(TemporalGraph, self).__init__(list)

    def convert_node_to_list(self):
        ls = set()
        for i, k in self.items():
            ls.add(i)
            for l in k:
                ls.add(l.name)
        return ls

    def make_undirected(self):

        t0 = time()
        keys = list(self.keys())
        for v in keys:
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        t1 = time()
        print(t1 - t0)
        # self.make_consistent()
        return self

    def make_consistent(self):
        t0 = time()
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))

        t1 = time()

        self.remove_self_loops()

        return self

    def remove_self_loops(self):

        removed = 0
        t0 = time()

        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1

        t1 = time()

        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True

        return False
