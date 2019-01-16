import numpy as np


def sigmoid(Z):
    """
    使用sigmoid函数做激活函数
    :param Z:
    :return: a 前一个神经元的输出，下一个神经元的输入
    """
    return 1 / (1 + np.exp(Z))


a = np.array([1, 2, 3])
print(a)
b = list(map(sigmoid, a))

print(np.reshape(b, 3))

w, e, r = [], [], []
w.append(2)
print('w', w)
print('e', e)
print('r', r)
w[0] = 1
print('w2',w)


for i in range(6):
    print('forward', i)

for j in range(5, -1, -1):
    print('backword', j)
