# 测试tile
import numpy as np

c = np.array([[1, 3], [3, 4]])
print(c)
a = np.array([1, 2])
size = c.shape[0]
b = np.tile(a, [size, 1]) - c
f = b ** 2
print(b)
print(f)
b = f.argsort()
print(type(b))
print(b)
labels = ['A', 'A', 'B', 'B']
dd = b ** 0.5
print('dd', dd)
print(b[0])
# d = labels[dd]
# print(d)

print(type(labels))
