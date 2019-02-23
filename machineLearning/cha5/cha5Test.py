"""
练习矩阵的乘法
"""
import numpy as np


def test1():
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    b = np.array([[3, 2, 3],
                  [10, 10, 10]])
    d = np.array([1, 1, 1]).reshape(1, 3)
    # print(a.T)
    # print(b.T)
    # 第一个矩阵是一行的计算
    c = np.dot(a.T, d.T)
    print(c.shape[0])
    print(a[1, 1])
    cc = np.ones(6)
    print(cc)


"""
------------试一下画图
"""
import matplotlib.pyplot as plt


def testPlot():
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-4.12414349 - 0.48007329 * x) / -0.6168482
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    plt.show()


if __name__ == "__main__":
    test1()

"""
---------书上的实现-------------------
"""
# import numpy as np
#
# # Size of the points dataset.
# m = 20
#
# # Points x-coordinate and dummy value (x0, x1).
# X0 = np.ones((m, 1))
# X1 = np.arange(1, m+1).reshape(m, 1)
# X = np.hstack((X0, X1))
# print(X)
# # Points y-coordinate
# y = np.array([
#     3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
#     11, 13, 13, 16, 17, 18, 17, 19, 21
# ]).reshape(m, 1)
#
# # The Learning Rate alpha.
# alpha = 0.01
#
# def error_function(theta, X, y):
#     '''Error function J definition.'''
#     diff = np.dot(X, theta) - y
#     return (1./2*m) * np.dot(np.transpose(diff), diff)
#
# def gradient_function(theta, X, y):
#     '''Gradient of the function J definition.'''
#     diff = np.dot(X, theta) - y
#     return (1./m) * np.dot(np.transpose(X), diff)
#
# def gradient_descent(X, y, alpha):
#     '''Perform gradient descent.'''
#     theta = np.array([1, 1]).reshape(2, 1)
#     gradient = gradient_function(theta, X, y)
#     while not np.all(np.absolute(gradient) <= 1e-5):
#         theta = theta - alpha * gradient
#         gradient = gradient_function(theta, X, y)
#     return theta
#
# optimal = gradient_descent(X, y, alpha)
# print('optimal:', optimal)
# print('error function:', error_function(optimal, X, y)[0,0])
