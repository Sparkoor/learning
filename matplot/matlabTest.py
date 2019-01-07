import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# x = [-1, 2, 3, 4]
# y = [-1, 2, 3, 4]
# np.linspace(-3, 3, 100)
# plt.plot(x[:3], y[:3])
# plt.xlabel('$x$', fontsize=16)
# plt.ylabel('$y$', fontsize=16)
# plt.show()
# # convert = lambda s: int(s.strip() or -999)
# # # 使用必须字段一样多
# # data = np.genfromtxt('global.1751_2014.csv', delimiter=',', skip_header=2,encoding='utf-8')
# #
# # print(data[:, 1])
#
# #
# data = (1, 2, 3, 3)
# print(type(data))
# # i是位置，d是数据
# # for i, d in enumerate(data):
# #     print('这是i', i)
# #     print('这是d', d)
#
# for a_prior, c, b in zip(data, ('b', 'r', 'g'), ('L', 'M')):
#     print('a_prior', a_prior)
#     # print('b_prior', b_prior)
#     print(c)
#     print(b)

def seaborn_test():
    s = [1, 2, 3, 4, 5, 6]
    # 是在概率论中用来估计未知的密度函数
    sns.kdeplot(s)
    plt.show()


if __name__ == "__main__":
    seaborn_test()
