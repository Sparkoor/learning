"""
from Incorporating Network Embedding into Markov Random Field for Better Community Detection
"""
import numpy as np
import math
from commonUtils.Loggings import Logger

logger = Logger.getLogger()


def init_niu():
    pass


def init_sigma():
    pass


def init_alpha():
    pass


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


def Q_func(r, u, y, sigma, alpha, k):
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
        dy = nk * np.log(alpha[i])
        yu = y[i] - u[i]
        Ejk = np.log(1 / np.sqrt(2 * np.pi)) - np.log(sigma[i]) - np.matmul(yu, yu) / 2 * np.power(sigma[i], 2)
        dm = np.sum(np.multiply(r[:, i], Ejk))
        Q_result = Q_result + (dy + dm)
    return Q_result


def get_param_by_em(y, u, sigma, alpha, k, max_error):
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
        var = Q_func(r, u, y, sigma, alpha, k)
        for i in range(vlen):
            for j in range(k):
                q = gmm_component(u[j], sigma[j], y[i])
                r[i, j] = (alpha[j] * q) / np.sum(np.multiply(alpha, q))
                # 按行加,然后分别除
                sumr = np.sum(r[:, k])
                u[j] = np.sum(np.multiply(r[:, k], y[i]), axis=0) / sumr
                yu = y[i] - u[j]
                sigma_2 = np.sum(np.multiply(r[:, k], np.multiply(yu, yu))) / sumr
                sigma[j] = np.sqrt(sigma_2)
                alpha[j] = sumr / vlen
        # 怎么证明其收敛
        tempVar = Q_func(r, u, y, sigma, alpha, k)
        if var - tempVar < max_error:
            break
        else:
            var = tempVar
    return u, sigma, alpha


def unary_function():
    pass
