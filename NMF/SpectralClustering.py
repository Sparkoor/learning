"""
图聚类算法之谱聚类
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import k_means
from sklearn.cluster import SpectralClustering
from itertools import cycle, islice
from machineLearning.cha10 import kMeans
import networkx as nx

from commonUtils.Loggings import Logger

logger = Logger.getLogger()


def load_data_from_adj(filename):
    """
    加载数据，生成邻接矩阵
    :param filename:
    :return:
    """
    data = np.zeros((34, 34))
    with open(filename) as f:
        for line in f:
            row = 0
            l = line.strip().split(' ')
            for i, li in enumerate(l):
                li = int(li)
                if i == 0:
                    row = li - 1
                    continue
                col = li - 1
                data[row, col] = 1
    return data


def load_data_from_sklearn(n_sample=1000):
    """
    在sklearn数据集中加载数据
    :param n_sample:
    :return:
    """
    # 带噪声的圆形数据
    nosiy_circle = datasets.make_circles(n_samples=n_sample, factor=5, noise=.05)
    # 带噪声的月牙数据
    nosiy_moons = datasets.make_moons(n_samples=n_sample, noise=.05)
    # 随机分布数据 ？？？？？
    no_structure = np.random.rand(n_sample, 2), np.ones((1, n_sample), dtype=np.int32).tolist()[0]
    # 各向异性分布数据？？？？？
    random_state = 170
    #
    x, y = datasets.make_blobs(n_samples=n_sample, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(x, transformation)
    aniso = (X_aniso, y)
    # 不同方差的气泡数据
    varied = datasets.make_blobs(n_samples=n_sample, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
    # 相同方差的数据
    blobs = datasets.make_blobs(n_samples=n_sample, random_state=8)
    # 合并数据
    data_set = [nosiy_circle, nosiy_moons, no_structure, aniso, varied, blobs]
    cluster_nums = [2, 2, 3, 3, 3, 3]
    data_mats = []
    for i in range(len(data_set)):
        X, y = data_set[i]
        # 标准化处理
        X = StandardScaler().fit_transform(X)
        X_mat = np.mat(X)
        Y_mat = np.mat(y)
        data_mats.append((X_mat, Y_mat))

    # 展示数据
    plt.figure(figsize=(2.5, 1.4))
    plt.subplots_adjust(left=.02, right=.98, bottom=0.001, top=.96, wspace=.05, hspace=.01)
    for i in range(len(data_set)):
        X, Y = data_set[i]
        X = StandardScaler().fit_transform(X)
        # todo:islice的用法
        colors = np.array(
            list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999',
                               '#e41a1c', '#dede00']), int(max(Y) + 1))))
        plt.subplot(len(data_set), 1, i + 1)
        if i == 0:
            plt.title('self-built Data Set', size=18)
        plt.scatter(X[:, 0], X[:, 1], c=colors[Y], s=10)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
    plt.show()
    return data_mats, cluster_nums


"""
求拉普拉斯矩阵
"""


def degree_matrix(A):
    """
    求度矩阵，对角线为顶点的度
    :param A:
    :return:
    """
    n = A.shape[0]
    degreeMat = np.zeros((n, n))
    for i, row in enumerate(A):
        degreeMat[i, i] = np.sum(row)
    return degreeMat


def laplacian_mat(A, D):
    """
    拉普拉斯矩阵
    :param A: 邻接矩阵
    :param D: 度矩阵
    :return:
    """
    return D - A


def eigen_matrix(L):
    """
    计算特征值特征向量
    :param L:
    :return:
    """
    return np.linalg.eig(L)


def get_eigenVec(k=10):
    A = load_data_from_adj(r'D:\work\learning\deepwalk\example\karate.adjlist')
    D = degree_matrix(A)
    L = laplacian_mat(A, D)
    w, v = eigen_matrix(L)
    l = w.tolist()
    dic = {}
    for i, m in enumerate(l):
        dic[m] = i
    d = sorted(dic, reverse=True)
    temp = np.zeros(v.shape)
    for i, key in enumerate(d):
        index = dic.get(key)
        if k == 0:
            break
        temp[:, i] = v[:, index]
        k = k - 1
    # 用这个进行k-means聚类
    return A, temp


def k_means(V):
    """
    对向量集合进行聚类，按行计算
    :param V:
    :return:
    """
    return kMeans.kMeans(V, 3, kMeans.distEclud, kMeans.randCent)


def get_cluster(M):
    """
    得到聚类结果
    :param M:
    :return:
    """
    cluster = {}
    for i, k in enumerate(M):
        key = k[0, 0]
        cluster[key] = cluster.get(key, []) + [i]
    logger.warning("cluster的个数{}".format(len(cluster)))
    return cluster


def draw_graph(A, cluster_map):
    """
    画出
    :return:
    """
    G = nx.from_numpy_matrix(A)
    clust_map = {}
    for k, val in cluster_map.items():
        for v in val:
            # 转换成节点对类型
            clust_map[v] = int(k)
    colors = []
    labels = {}
    for i in range(len(G.nodes())):
        # 为不同的节点分配不同的颜色
        labels[i] = i
        colors.append(clust_map.get(i, 100))
    pos = nx.spring_layout(G)
    from matplotlib.pylab import show, cm
    plt.figure(2)
    nx.draw_networkx_nodes(G, pos, nodelist=list(cluster_map.get(0)), node_size=200, node_color='r')
    nx.draw_networkx_nodes(G, pos, nodelist=list(cluster_map.get(1)), node_size=200, node_color='b')
    nx.draw_networkx_nodes(G, pos, nodelist=list(cluster_map.get(2)), node_size=200, node_color='y')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    plt.show()
    show()


def load_deepwalk_emb(filename):
    """
    对deepwalk嵌入的向量进行k-means聚类
    :param filename:
    :return: 返回一个矩阵
    """
    mat = []
    with open(filename) as f:
        msg = f.readline()
        for emb in f:
            vec = emb.strip().split(" ")
            mat.append([x for x in map(float, vec[1:])])
    return np.mat(mat)


if __name__ == '__main__':
    A, V = get_eigenVec(20)
    E = load_deepwalk_emb(r"D:\work\learning\deepwalk\example\karate")
    c, m = k_means(E)
    cluster = get_cluster(m)
    draw_graph(A, cluster)
