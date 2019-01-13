"""
向量化
"""
import numpy as np
import time

row = np.random.rand(100000)
col = np.random.rand(100000)
t1 = time.time()
res = np.dot(row, col)
t2 = time.time()
print("cost time", str(t2 - t1))
print(res)
sum = 0
for i in range(100000):
    sum += col[i] * row[i]
t3 = time.time()
print("cost time", str(t3 - t2))
print(sum)
# 声明几维向量
np.zeros()
