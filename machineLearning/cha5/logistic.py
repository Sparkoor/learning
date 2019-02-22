"""
逻辑回归，对代价函数使用梯度下降法
"""
import numpy as np


def errorFunc(theta, x, y):
    """
    代价函数，使用的是距离
    :param theta:
    :param x:
    :param y:
    :return:
    """
    m = len(theta)
    diff = np.dot(theta, x.T) - y
    return (1 / 2) * (1 / m) * diff ** 2


def derivativesFunc(theta, x, y):
    """
    计算代价函数的导函数
    :param theta:
    :param x:
    :param y:
    :return:
    """
    m = len(theta)
    diff = np.dot(theta, x.T) - y
    return (1 / m) * np.dot(x.T, diff)


def gradientDescent(theta, alpha, x, y):
    """
    梯度下降算法实现
    :param alpha:
    :param x:
    :param y:
    :return:
    """
    theta1 = theta.copy()
    # 计算差值
    gradient = errorFunc(theta, x, y)
    while not np.all(np.absolute(gradient) < 1e-5):
        theta1 = theta1 - alpha * derivativesFunc(theta1, x, y)
        gradient = gradientDescent(theta1, alpha, x, y)
    return theta1
