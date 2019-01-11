"""
beta分布
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 这里的值放的是alpha和beta
params = [0.5, 1, 2, 3]
x = np.linspace(0, 1, 100)
f, ax = plt.subplots(len(params), len(params), sharex=True, sharey=True)
for i in range(4):
    for j in range(4):
        a = params[i]
        b = params[j]
        # pdf概率分布相关
        y = stats.beta(a, b).pdf(x)
        ax[i, j].plot(x, y)
        ax[i, j].plot(0, 0, label="$\\alpha$={:3.2f}\n$\\beta$={:3.2f}".format(a, b), alpha=0)
        ax[i, j].legend(fontsize=8)
ax[3, 0].set_xlabel('$\\theta$', fontsize=16)
ax[0, 0].set_ylabel('$p(\\theta)$', fontsize=16)
# 保存为图片
# plt.savefig('bata.png', dpi=300, figsize=(5.5, 5.5))
plt.show()
