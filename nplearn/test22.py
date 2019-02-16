import numpy as np
from numpy import genfromtxt
# 注意是.pyplot中
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = genfromtxt("Delivery.csv", delimiter=',')
print(data)
x_data = data[:, :-1]
y_data = data[:, -1]
# 学习率
lr = 0.0001
# 参数
theta0 = 0
theta1 = 0
theta2 = 0
# 最大迭代次数
epochs = 1000


# 最小二乘法
def computer_error(theta0, theta1, theta2, x_data, y_data):
    totalerror = 0
    for i in range(0, len(x_data)):
        totalerror += (theta0 + theta1 * x_data[i, 0] + theta2[i, 1] - y_data[i]) ** 2
    return totalerror / float(len(x_data))


def gradient_descent_runner(x_data, y_data, theta0, theta1, theta2, lr, epochs):
    # 计算梯度
    # 计算总数据量
    m = float(len(x_data))
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        for j in range(0, len(x_data)):
            theta0_grad += (1 / m) * (theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1] - y_data[j])
            theta1_grad += (1 / m) * (theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1] - y_data[j]) * x_data[j, 0]
            theta2_grad += (1 / m) * (theta0 + theta1 * x_data[j, 0] + theta2 * x_data[j, 1] - y_data[j]) * x_data[j, 1]
        # 更新b和k
        theta0 = theta0 - lr * theta0_grad
        theta1 = theta1 - lr * theta1_grad
        theta2 = theta2 - lr * theta2_grad
    return theta0, theta1, theta2


theta0, theta1, theta2 = gradient_descent_runner(x_data, y_data, theta0, theta1, theta2, lr, epochs)

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(x_data[:, 0], x_data[:, 1], y_data, c='r', marker='o', s=100)
x0 = x_data[:, 0]
x1 = x_data[:, 1]
# 生成网格矩阵
x0, x1 = np.meshgrid(x0, x1)
z = theta0 + x0 * theta1 + theta2 * x1
# 画3D图
ax.plot_surface(x0, x1, z)
# 设置坐标轴
ax.set_xlabel('Miles')
ax.set_ylabel('Num of Deliveries')
ax.set_zlabel('Time')
plt.show()
