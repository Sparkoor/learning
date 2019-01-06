# 提供类似matlab的画图方式
import matplotlib.pyplot as plt

# 数组之类的操作
import numpy as np

# stats统计功能
from scipy import stats

mu_params = [-1, 0, 1]
sd_params = [0.5, 1, 1.5]
# 设置x轴的，行间距
x = np.linspace(-7, 7, 100)
# subplots返回的是两个值，一个是主图f，副图ax
f, ax = plt.subplots(len(mu_params), len(sd_params), sharex=True, sharey=True)
# 超过九个图还不够大
for i in range(3):
    for j in range(3):
        mu = mu_params[i]
        sd = sd_params[j]
        # pdf概率密度函数，根据x求y的函数，这句还是不懂
        y = stats.norm(mu, sd).pdf(x)
        # ppt百分点函数
        # y = stats.norm(mu, sd).ppf(x)
        ax[i, j].plot(x, y)
        # 设置弹出框的 控制纵轴的，alpha是做什么的？如果不等于0可以带线
        ax[i, j].plot(0, 0, label="$\\mu$={:3.2f}\n$\\sigma$={:3.2f}\n$\\alpha$={:3.2f}".format(mu, sd,sd), alpha=0)
        # 显示弹出框
        ax[i, j].legend(fontsize=8)

# 设置这个的x周的说明
ax[2, 1].set_xlabel('$x$', fontsize=16)
# 设置y轴的说明
ax[1, 0].set_ylabel('$pdf(x)$', fontsize=16)
ax[2, 2].set_xlabel('$t$', fontsize=16)
ax[2, 2].set_ylabel('$s$', fontsize=16)
# 这个没看出来有啥效果
plt.tight_layout()
# 把图像显示出来
plt.show()
