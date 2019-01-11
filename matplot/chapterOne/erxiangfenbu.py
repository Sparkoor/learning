"""
画柱状图
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

n_params = [1, 2, 4]
p_params = [0.25, 0.5, 0.75]
# 设置x轴范围,因为没用到概率分布所以没用linespace
x = np.arange(0, max(n_params) + 1)
# 第一个参数和第二个参数是设置了有每行每列有几个图，后两个参数是设置，是否共享轴的参数
f, ax = plt.subplots(len(n_params), len(p_params), sharex=True, sharey=True)
print(type(ax))
for i in range(3):
    for j in range(3):
        n = n_params[i]
        p = p_params[j]
        # pmf是做什么的，计算二项分布的公式
        y = stats.binom(n=n, p=p).pmf(x)
        # 画柱状图，还填颜色
        ax[i, j].vlines(x, 0, y, colors='b', lw=5)
        ax[i, j].set_ylim(0, 1)
        ax[i, j].plot(0, 0, label="n={:3.2f}\np={:3.2f}".format(n, p), alpha=0)
        ax[i, j].legend(fontsize=10)
        print(type(ax[i, j]))
ax[2, 1].set_xlabel('$\\theta$', fontsize=14)
ax[1, 0].set_ylabel('$p(y|\\theta)$', fontsize=14)
ax[0, 0].set_xticks(x)
plt.show()
