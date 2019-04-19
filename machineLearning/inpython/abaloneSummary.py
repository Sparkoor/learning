"""
使用pandas分析数据
"""
from urllib import request
import sys
import numpy as np

target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/' \
             'undocumented/connectionist-bench/sonar/sonar.all-data'
data = request.urlopen(target_url, data=None, timeout=10)

x_list = []
labels = []

for line in data:
    row = line.split(',')
    x_list.append(row)
n_row = len(x_list)
n_col = len(x_list[1])
type = [0] * 3
colcount = []
col = 3
col_data = []

# 计算某一列的值
for row in x_list:
    col_data.append(float(row(col)))

col_array = np.array(col_data)
col_mean = np.mean(col_array)
colsd = np.std(col_array)

print('mean:{},std:{}'.format(col_mean, colsd))

ntiles = 4

percentBdry = []
for i in range(ntiles + 1):
    # 这个不懂是做什么的
    percentBdry.append(np.percentile(col_array, i * (100) / ntiles))

print(percentBdry)

ntiles=10
#
# for i in range(ntiles+1):

