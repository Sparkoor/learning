"""
后验
"""
import matplotlib.pyplot as plt
import numpy as np
# state统计函数
from scipy import stats

# 真theta的值
theta_real = 0.35
# 实验次数
trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
#  bate先验的数据的值
beta_params = [(1, 1), (0.5, 0.5), (20, 20)]
# 初始化beta算法
dist = stats.beta
# 设置x轴 数据的第一个元素值为a，最后一个元素为b，n是总采样点数
x = np.linspace(0, 1, 100)
# 同时取游标和数组的值
for idx, N in enumerate(trials):
    # 这个判断保证了图的位置
    if idx == 0:
        # 需要添加几个图，4是指具有几行3，只具有几列，2是指该图所在的位置
        plt.subplot(4, 3, 2)
    else:
        plt.subplot(4, 3, idx + 3)
        print(type(plt))
    # 获取本次实验的数据
    y = data[idx]
    # zip是用来组合元素的，变量c是颜色变量
    for (a_prior, b_prior), c in zip(beta_params, ('b', 'r', 'g')):
        # 根据公式计算出y的值
        p_theta_given_y = dist.pdf(x, a_prior + y, b_prior + N - y)
        # 将x，y的值带入图中，以及颜色
        plt.plot(x, p_theta_given_y, c)
        # 填充两条线之间的颜色 0 应该是x轴那根线
        plt.fill_between(x, 0, p_theta_given_y, color=c, alpha=0.6)
    # 在图的x轴画一个竖线
    plt.axvline(theta_real, ymax=0.3, color='k')
    # 写
    plt.plot(0, 0, label="{:d} experiment\n{:d}heads".format(N, y), alpha=0)
    # 设置x轴的长度
    plt.xlim(0, 1)
    # 相当于设置y轴高度了
    plt.ylim(0, 12)
    # 设置x轴下的提示
    plt.xlabel(r"$\theta$")
    plt.legend(fontsize=5)
    # 取到y轴然后再设置，y轴的显示属性
    plt.gca().axes.get_yaxis().set_visible(False)
# 没看出有什么效果
plt.tight_layout()
# plt.savefig('image.png', dpi=300, figsize=(5.5, 5.5))
plt.show()
