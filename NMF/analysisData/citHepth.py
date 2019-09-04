"""
高能物理引用网络
"""
import numpy
import networkx as nx
from TemporalNetwork import TemporalGraph, Node
import datetime as dt
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import show


class LoadGraph(object):
    def __init__(self, graph):
        self.graph = graph

    def loadfile(self, filename1, filename2):
        """
        加载文件
        :return:
        """
        # graph = TemporalGraph()
        # 先读取节点时间
        node = dict()
        num = 0
        with open(filename1) as f:
            for s in f:
                num = num + 1
                if s.startswith('#'):
                    continue
                n = s.strip().split('\t')
                # print(n)
                # 有重复的
                node[n[0]] = dt.datetime.strptime(n[1], "%Y-%m-%d")

        # print(num)
        # print(len(node.keys()))
        with open(filename2) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                n = line.strip().split('\t')
                self.graph[n[0]].append(Node(n[1], node.get(n[1], dt.datetime.strptime('1992-2-1', "%Y-%m-%d"))))
        # print(len(graph.keys()))
        # 是有向的
        # print(len(graph.make_undirected().keys()))
        return self.graph

    def write_graph_to_file(self, filename):
        """
        把图以便于理解的形式写入文件
        :param filename:
        :return:
        """
        with open(filename, 'w') as f:
            for k, v in self.graph.items():
                for i in v:
                    s = k + ' ' + i.name + ' ' + str(i.time.date())
                    f.write(s + '\n')


def loadfile_by_limit_date(filename, date):
    """
    根据时间选取图
    :param filename:
    :param date:
    :return:
    """
    limit_date = dt.datetime.strptime(date, "%Y-%m-%d")
    g = TemporalGraph()
    total_size = os.path.getsize(filename)
    print(total_size)
    read_size = 0
    total = 0
    ls = set()
    with open(filename) as f:
        for line in f:
            read_size = read_size + len(line)
            total += 1
            if total % 10000 == 0:
                # 加载文件的大小
                print(read_size / total_size)
            s = line.strip().split(" ")
            time = dt.datetime.strptime(s[2], "%Y-%m-%d")
            if time < limit_date:
                g[s[0]].append(s[1])
                ls.add(s[0])
                ls.add(s[1])
    return g, list(ls)


def draw_graph(graph, nodes):
    """

    :param graph:
    :return:
    """
    G = nx.from_dict_of_lists(graph)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, nodelist=nodes, node_size=200, node_color='r')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    show()


if __name__ == '__main__':
    # graph = TemporalGraph()
    # load = LoadGraph(graph)
    # load.loadfile(r"D:\work\learning\NMF\datasets\citHepTH\Cit-HepTh-dates.txt",
    #               r'D:\work\learning\NMF\datasets\citHepTH\Cit-HepTh.txt')
    # load.write_graph_to_file(r'D:\work\learning\NMF\datasets\citHepTH\Cit-HepTh-graph.txt')
    path = r'D:\work\learning\NMF\datasets\citHepTH\Cit-HepTh-graph.txt'
    g, nodes = loadfile_by_limit_date(path, '1992-2-2')
    draw_graph(g, nodes)
    # print(g)
