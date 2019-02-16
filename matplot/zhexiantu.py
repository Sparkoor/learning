"""
折现图
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset = pd.read_csv('dataset.csv', encoding='utf-8')
print(dataset)
# 获取的数据带了一个\t需要进行分割
data2 = pd.DataFrame()
data2['year'], data2['num'] = dataset['year\tnum'].str.split('\t', 1).str
print(data2)
dt = data2[0:12]
print(dt)

plt.plot(dt['year'], dt['num'])
# x轴的数倾斜
plt.xticks(rotation=45)
# 汉字存在乱码
plt.xlabel('year')
plt.show()

"""
画子图
"""
# 设置指定大小 fig做设置
ax = plt.figure(figsize=(3, 3))
ax1 = ax.add_subplot(2, 2, 1)
ax2 = ax.add_subplot(2, 2, 2)
ax4 = ax.add_subplot(2, 2, 4)
ax1.plot(np.random.randn(2), np.arange(2), c='red', label="aaaa")
ax1.plot(np.random.randn(2), np.arange(2), c='blue')
# loc的参数可以是 upper left right lower center best
ax1.legend(fontsize=8, loc='best')
plt.show()

# 柱形图
# 使用bar这个 hist可以根据不同的范围设置不同的颜色
plt.bar(dt['year'], dt['num'], 0.3)
plt.barh(dt['year'], dt['num'], 0.3)
plt.show()

# 散点图
plt.scatter(dt['year'], dt['num'], c='red')
plt.show()
col = ['year', 'num']
# print(dt[col])
# 柱形图没画出来
fig, ab = plt.subplots()
ab.boxplot(dt[col].values)
plt.show()

