"""
实现FSCNMF算法
"""
from collections import defaultdict


def load_cites(filepath):
    """
    加载文件后缀为cites的文件
    :param filepath:
    :return:
    """
    graph = Graph()
    node = Node()
    with open(filepath) as f:
        for i in f:
            cite1, cite2 = i.strip().split(" ")
            node[cite1].append(cite2)
    index = 0
    for n in node:
        graph[index].append(n)


if __name__ == '__main__':
    load_cites(r'D:\work\learning\NMF\datasets\citeseer.cites')


class Node(defaultdict):
    def __init__(self):
        super(Node, self).__init__(list)


class NodeAndVec():
    def __init__(self, node, vec=None):
        self.node = node
        self.vec = vec


class Graph(dict):
    def __init__(self):
        super(Graph, self).__init__(NodeAndVec)

    def getNode(self, index):
        pass


class FSCNMF():
    def __init__(self, graph, dimen):
        self.graph = graph
        self.dimen = dimen

    def calculateMatrixA(self):
        for node in self.graph:
            pass
