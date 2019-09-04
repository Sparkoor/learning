"""
from Incorporating Network Embedding into Markov Random Field for Better Community Detection
"""
import numpy as np
import math
from commonUtils.Loggings import Logger

logger = Logger.getLogger()

"""
需要对参数进行修改
"""


class UnaryPotential(object):

    def __init__(self, v, k, vectors, beta):
        """

        :param v: 节点向量长度
        :param k: 社区数
        :param vectors: 节点向量
        """
        self.vectors = vectors
        self.U = self.init_niu(v, k)
        self.Sigma = self.init_sigma(k)
        self.Alpha = self.init_alpha(k)
        self.beta = beta

    @staticmethod
    def init_niu(v, k):
        """

        :param v:
        :param k:
        :return:
        """
        return np.random.rand((v, k))

    @staticmethod
    def init_sigma(k):
        return np.random.rand((1, k))

    @staticmethod
    def init_alpha(k):
        a = np.random.rand(k)
        sum = np.sum(a)
        return a / sum

    @staticmethod
    def gmm_component(u, sigma, y):
        """
        高斯混合分布的一个组件
        :param u:
        :param sigma:
        :param y:
        :return:
        """
        di = np.sqrt(2 * math.pi) * sigma
        if di == 0:
            logger.error("di", str(di))
            raise Exception("除数为0了")
        # 两个向量的差
        yu = y - u
        z = np.matmul(yu, yu)
        x = np.power(sigma, 2) * 2
        m = np.exp(-np.divide(z, x))
        result = np.divide(m, di)
        return result

    def q_func(self, r, u, y, sigma, alpha, k):
        """
        EM算法中的Q函数
        :param r:
        :param u:
        :param y:
        :param sigma:
        :param alpha:
        :return:
        """
        Q_result = 0
        for i in range(k):
            nk = np.sum(r[i])
            dy = nk * np.log(self.Alpha[i])
            yu = y[i] - u[i]
            Ejk = np.log(1 / np.sqrt(2 * np.pi)) - np.log(sigma[i]) - np.matmul(yu, yu) / 2 * np.power(sigma[i], 2)
            dm = np.sum(np.multiply(r[:, i], Ejk))
            Q_result = Q_result + (dy + dm)
        return Q_result

    def get_param_by_em(self, y, u, sigma, alpha, k, max_error):
        """
        EM算法的实现
        :param y: 节点向量
        :param u: 高斯分布参数,这也是个向量吗
        :param sigma: 高斯分布参数
        :param alpha: 每个高斯分布所占比例
        :param k 社区数
        :param max_error 迭代误差容忍值
        :return:
        """
        # 节点向量的长度
        vlen = y.shape[0]
        # 隐变量
        r = np.zeros((vlen, k), dtype=np.float)
        while True:
            var = self.Q_func(r, u, y, sigma, alpha, k)
            for i in range(vlen):
                for j in range(k):
                    q = self.gmm_component(u[j], sigma[j], y[i])
                    r[i, j] = (alpha[j] * q) / np.sum(np.multiply(alpha, q))
                    # 按行加,然后分别除
                    sumr = np.sum(r[:, k])
                    u[j] = np.sum(np.multiply(r[:, k], y[i]), axis=0) / sumr
                    yu = y[i] - u[j]
                    sigma_2 = np.sum(np.multiply(r[:, k], np.multiply(yu, yu))) / sumr
                    sigma[j] = np.sqrt(sigma_2)
            N = np.sum(np.sum(r))
            alpha = np.sum(r, axis=1) / N
            # 怎么证明其收敛
            tempVar = self.Q_func(r, u, y, sigma, alpha, k)
            if var - tempVar < max_error:
                break
            else:
                var = tempVar
        return u, sigma, alpha

    def compute_gamma(self, vectors, u, simga, alpha):
        """
        计算一元函数中的gamma
        :param vectors: 节点向量集合
        :param u: 高斯分布的参数，是一个矩阵
        :param simga: 高斯分布的参数
        :param alpha: 高斯分布的参数
        :return: 是个矩阵吧
        """
        M = len(u)
        assert len(u) == len(simga)
        # 存放γ的，有几个节点该数组的长度就为几
        gamma = []
        # 求节点向量在该分布的概率，然后求和在任何分布的概率得到γ，todo：我是这样理解的。
        for y in vectors:
            community = []
            for i in range(M):
                p = self.gmm_component(u[i], simga[i], y)
                sum_gamma = alpha[i] * p
                community.append(sum_gamma)
            gamma.append(community)
        # 这是个矩阵表示，节点属于该社区的概率
        return np.array(gamma)

    def unary_function(self, i, k, gamma):
        """
        一元势函数
        :param i: 节点
        :param k: 社区
        :param gamma:节点属于该社区的概率
        :param beta: 超参数
        :return:
        """
        return -np.log(gamma[i, k]) / self.beta


def pairwise_potentials(A):
    """
    计算成对势函数
    :param A:
    :return:
    """
    # 计算邻接矩阵的特征值特征向量
    eig_vector, eig_value = np.linalg.eigh(A)
