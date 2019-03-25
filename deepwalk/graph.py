"""
图的数据结构
"""

from commonUtils.Loggings import *
from collections import defaultdict, Iterable
from time import time
import random
from scipy.io import loadmat
from scipy.sparse import issparse
import numpy as np
from itertools import product, permutations, zip_longest

logger = Logger().getLogger()


class Graph(defaultdict):
    """
    实现图的功能
    """

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacencyIter(self):
        return self.items()

    def subgraph(self):
        """
        返回子图
        :return:
        """
        subgraph = Graph()
        for n in self.nodes():
            if n in self:
                subgraph[n] = [x for x in self[n] if x in self.nodes()]
        return subgraph

    def makeUndirected(self):
        """
        把有序图变成无序
        :return:
        """
        # 用于计时
        t0 = time()
        # 就是把单向的弄成双向的
        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)
        t1 = time()
        logger.info('转化成无向图用时{}'.format(t1 - t0))
        self.makeConsistent()
        return self

    def makeConsistent(self):
        """
        去除重复的
        :return:
        """
        t0 = time()
        for k in iter(self):
            self[k] = list(sorted(set[k]))
        t1 = time()
        logger.info("去除重复顶点{}".format(t1 - t0))
        self.removeSelfLoop()
        return self

    def removeSelfLoop(self):
        """
        去除自循环节点
        :return:
        """
        removed = 0
        t0 = time()

        for x in self:
            for x in self[x]:
                self[x].remove(x)
                removed += 1
        t1 = time()
        logger.info("去除循环用时{}".format(t1 - t0))
        return self

    def checkSelfLoop(self):
        """
        检查是否存在圈
        :return:
        """
        for x in self:
            for y in self[x]:
                if x == y:
                    return True
        return False

    def hasEdge(self, v1, v2):
        """
        检查是否存在边
        :param v1:
        :param v2:
        :return:
        """
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        """
        计算顶点的度
        :param nodes:
        :return:
        """
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def order(self):
        """图的节点数量"""
        return len(self)

    def numberOfEdge(self):
        """图的边数"""
        return sum([self.degree(x) for x in self.keys()]) / 2

    def numberOfNode(self):
        return self.order()

    def randWalk(self, pathLength, alpha=0, rand=random.Random(), start=None):
        """
        返回一个短的随机游走
        :param pathLength: 随机游走的长度
        :param alpha: 从新出发的概率
        :param rand:
        :param start: 开始的节点
        :return:
        """
        G = self
        if start:
            path = [start]
        else:
            logger.warning("图的keys{}".format(G.keys()))
            path = [rand.choice(list(G.keys()))]
        while len(path) < pathLength:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


def buildDeepWalkCorpus(G, numPaths, pathLength, alpha=0, rand=random.Random(0)):
    """
    构建深度游走语料库
    :param G:
    :param numPaths:
    :param pathLength:
    :param alpha:
    :param rand:
    :return:
    """
    logger.error('random.Random(0)输出的是什么{}'.format(rand))
    walks = []
    nodes = list(G.nodes())
    for cnt in range(numPaths):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.randomWalk(pathLength, rand=rand, alpha=alpha, start=node))

    return walks


def buildDeepWalkCorpusIter(G, numLength, pathLength, alpha=0, rand=random.Random(0)):
    """
    单个随机游走
    :param G:
    :param numLength:
    :param pathLength:
    :param alpha:
    :param rand:
    :return:
    """
    nodes = list(G.nodes)
    for cnt in range(numLength):
        rand.shuffle(nodes)
        for node in nodes:
            # note:每次通过next()才能执行
            yield G.randomWalk(pathLength, rand=rand, alpha=alpha, start=node)


def clique(size):
    """
    通过邻接矩阵返回图
    :param size:
    :return:
    """
    # todo：分割函数
    # note：permutations的作用
    return fromAdjList(permutations(range(1, size + 1)))


def grouper(n, iterable, padvalue=None):
    """
    todo:不明白其作用
    :param n:
    :param iterable:
    :param padvalue:
    :return:
    """
    logger.warning("zip_lonest的参数{}".format(*[iter(iterable)] * n))
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def parseAdjacencyList(f):
    """
    将矩阵转化成列表,这个需要更具数据格式进行调整
    :param f:
    :return:
    """
    adjlist = []
    for l in f:
        if l and l[0] != '#':
            introw = [int(x) for x in l.strip().split()]
            row = [introw[0]]
            row.extend(set(sorted(introw[1:])))
            adjlist.extend(row)
    return adjlist


def parseAdjacentListUncheck(f):
    """

    :param f:
    :return:
    """
    adjList = []

    for l in f:
        if l and l[0] != '#':
            adjList.extend([[int(x) for x in l.strip().split()]])

    return adjList


def fromAdjList(adjlist):
    """
    在无检查的列表中构造图
    :param adjlist:
    :return:
    """
    G = Graph()
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))
    return G


def fromAdjListUnchecked(adjlist):
    """
    不带重复检查
    :param adjlist:
    :return:
    """
    G = Graph()
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors


def loadAdjacencyList(file_, undirected=False, chunksize=10000, unchecked=True):
    """
    从adjlist文件中获取数据
    :param file_:
    :param undirected:
    :param chunksize:
    :param unchecked:
    :return:
    """
    # 设置调用的功能
    if unchecked:
        parseFunc = parseAdjacentListUncheck
        convertFunc = fromAdjListUnchecked
    else:
        parseFunc = parseAdjacencyList
        convertFunc = fromAdjList

    adjlist = []

    t0 = time()

    total = 0
    with open(file_, mode='rb') as f:
        # todo:不懂下面这句的意思
        logger.error("从文件中读取信息")
        for idx, adjChunk in enumerate(map(parseFunc, grouper(int(chunksize), f))):
            adjlist.extend(adjChunk)
            total += len(adjChunk)
    t1 = time()

    logger.info("从邻接表读取数据完成！！！")
    G = convertFunc(adjlist)
    if undirected:
        logger.info("转化成无向图")
        G = G.makeUndirected()

    return G


def loadEdgeList(file_, undiercted=True):
    """

    :param file_:
    :param undiercted:
    :return:
    """
    G = Graph()
    with open(file_) as f:
        for l in f:
            x, y = l.strip().split()[:2]
            x = int(x)
            y = int(y)
            G[x].append(y)
    G.makeConsistent()
    return G


def loadMatfile(file_, variableName='network', undirected=True):
    """

    :param file_:
    :param variableName:
    :param undirected:
    :return:
    """
    matVarables = loadmat(file_)
    matMatrix = matVarables(variableName)
    return fromNumpy(matMatrix, undirected)


def fromNumpy(mat, undirected=True):
    """

    :param mat:
    :param undirected:
    :return:
    """
    G = Graph()
    # note:这是验证什么的
    if issparse(mat):
        cx = mat.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            G[i].append(j)
    else:
        raise Exception("Dense matrices not yet supported")
    if undirected:
        G.makeUndirected()
    G.makeConsistent()
    return G


def fromNetworks(G_input, undirected=True):
    """

    :param G_input:
    :param undirected:
    :return:
    """
    G = Graph()
    for idx, x in enumerate(G_input.nodesIter()):
        for y in iter(G_input[x]):
            G[x].append(y)
    if undirected:
        G.makeUndirected()
    return G


if __name__ == '__main__':
    G = loadAdjacencyList(r'D:\work\learning\deepwalk\example\blogcatalog.mat')
    print(G)
