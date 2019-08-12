"""
马尔可夫聚类
"""
import sys
import numpy as np
import time
from optparse import OptionParser
import logging


def normalize(A):
    """
    标准化矩阵
    :param A:
    :return:
    """
    # 按列相加
    column_sums = A.sum(axis=0)
    # np.newaxis是把列表扩张了吧？？
    new_matrix = A / column_sums[np.newaxis, :]
    return new_matrix


def inflate(A, inflate_factor):
    """
    防止
    :param A:
    :param inflate_factor:
    :return:
    """
    return normalize(np.power(A, inflate_factor))


def expand(A, expand_factor):
    """

    :param A:
    :param expand_factor:
    :return:
    """
    return np.linalg.matrix_power(A, expand_factor)


def add_diag(A, mult_factor):
    """
    添加矩阵对角元素啊
    :param A:
    :param mult_factor:
    :return:
    """
    return A + mult_factor * np.identity(A.shape[0])


def get_clusters(A):
    """

    :param A:
    :return:
    """
    cluster = []
    for i, r in enumerate((A > 0).tolist()):
        if r[i]:
            cluster.append(A[i, :] > 0)
    clust_map = {}

    for cn, c in enumerate(cluster):
        for x in [i for i, x in enumerate(c) if x]:
            clust_map[cn] = clust_map.get(cn, []) + [x]
    return clust_map


def draw(G, A, cluster_map):
    """
    这样画网络图的啊
    :param G:
    :param A:
    :param cluster_map:
    :return:
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    clust_map = {}
    # 边
    for k, vals in cluster_map.items():
        for v in vals:
            clust_map[v] = k
    colors = []
    for i in range(len(G.nodes())):
        # todo:get后面的数字是做什么的
        colors.append(clust_map.get(i, 100))
    pos = nx.spring_layout(G)

    from matplotlib.pylab import matshow, show, cm
    plt.figure(2)
    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=colors, cmap=plt.cm.Blues)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    show()


def stop(M, i):
    """
    判断矩阵迭代没有变化了。。。
    :param M:
    :param i:
    :return:
    """
    if i % 5 == 4:
        m = np.max(M ** 2 - M) - np.min(M ** 2 - M)
        if m == 0:
            logging.info("stop at iteration %s" % i)
            return True

    return False


def mcl(M, expand_factor=2, inflate_factor=2, max_loop=10, mult_factor=1):
    """

    :param M:
    :param expand_factor:
    :param inflate_factor:
    :param max_loop:
    :param mult_factor:
    :return:
    """
    M = add_diag(M, mult_factor)
    M = normalize(M)
    for i in range(max_loop):
        logging.info("loop %s" % i)
        M = inflate(M, inflate_factor)
        M = expand(M, expand_factor)
        if stop(M, i):
            break
    cluster = get_clusters(M)
    return M, cluster


def networkx_mcl(G, expand_factor=2, inflate_factor=2, max_loop=10, mult_factor=1):
    """

    :param G:
    :param expand_factor:
    :param inflate_factor:
    :param max_loop:
    :param mult_factor:
    :return:
    """
    import networkx as nx
    A = nx.adj_matrix(G)
    return mcl(np.array(A.todense()), expand_factor, inflate_factor, max_loop, mult_factor)


def get_graph_csv(filename):
    import networkx as nx
    M = []
    with open(filename) as f:
        for l in f:
            r = l.strip().split(",")
            M.append((list(map(lambda x: float(x.strip()), r))))
    G = nx.from_numpy_matrix(np.mat(M))
    return np.array(M), G


def get_graph_adjlist(filename):
    """
    从邻接矩阵文件中构造矩阵
    :param filename:
    :return:
    """
    import networkx as nx
    mat = np.zeros((34, 34))
    with open(filename) as f:
        for l in f:
            r = l.strip().split(" ")
            col = 0;
            for i, v in enumerate(r):
                if i == 0:
                    col = int(v.strip()) - 1
                    continue
                row = int(v.strip()) - 1
                mat[col, row] = 1
    G = nx.from_numpy_matrix(np.mat(mat))
    return mat, G


def clusters_to_output(cluster, option):
    """

    :param cluster:
    :param option:
    :return:
    """
    if option.output and len(option.output) > 0:
        f = open(option.output, 'a')
        for k, v in cluster.items():
            f.write("%s|%s\n" % (k, ",".join(map(str, v))))
        f.write("----------------------------------------------------")
        f.close()
    else:
        print("cluster:")
        for k, v in cluster.items():
            print('{},{}'.format(k, v))


def get_options():
    # todo:作用是啥
    usage = "usage: %prog[options]<input_matrix>"
    parser = OptionParser(usage)
    parser.add_option("-e", "--expand_factor",
                      dest="expand_factor",
                      default=2,
                      type=int,
                      help="he")
    parser.add_option("-i", "--inflate_factor",
                      dest="inflate_factor",
                      default=2,
                      type=float,
                      help="help")
    parser.add_option("-m", "--mult_factor",
                      dest="mult_factor",
                      default=2,
                      type=float,
                      help="help t")
    parser.add_option("-l", "--max_loops",
                      dest="max_loop",
                      default=60,
                      type=int,
                      help="hee")
    parser.add_option("-o", "--output",
                      metavar="FILE",
                      help="file")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=True,
                      help="verbose (default: %default)")
    parser.add_option("-d", "--draw-graph",
                      action="store_true", dest="draw", default=True,
                      help="show graph with networkx (default: %default)")
    # dest才是真正的名字
    parser.add_option("-f", "--file_path",
                      dest="file_path",
                      # default=r"D:\work\learning\NMF\datasets\example.csv",
                      default=r"D:\work\learning\deepwalk\example\karate.adjlist",
                      type=str,
                      help="file")
    (options, args) = parser.parse_args()

    # try:
    #     # 会把文件名放在第一个吗
    #     filename = args[0]
    # except:
    #     raise Exception('input', 'ssss')

    return options


if __name__ == '__main__':
    options = get_options()
    print(options)
    M, G = get_graph_adjlist(options.file_path)
    print(" number of nodes: %s\n" % M.shape[0])

    print("{}: {}".format(time.time(), "evaluating clusters..."))
    M, clusters = networkx_mcl(G, expand_factor=options.expand_factor,
                               inflate_factor=options.inflate_factor,
                               max_loop=options.max_loop,
                               mult_factor=options.mult_factor)
    print("{}: {}".format(time.time(), "done\n"))

    clusters_to_output(clusters, options)

    if options.draw:
        print("{}: {}".format(time.time(), "drawing..."))
        draw(G, M, clusters)
        print("{}: {}".format(time.time(), "done"))
