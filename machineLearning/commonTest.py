import numpy as np

# import matplotlib.pyplot as plt

x = np.arange(-5, 5, 1)
y = np.arange(-5, 3, 2)
xx, yy = np.meshgrid(x, y)
# print(xx)
# print(yy)
aa = np.mat(np.nonzero((5, 2)))
print(aa.A)
# print(aa[0, :].A)

# print(x)
# print(y)
# print(yy.ravel())

a = 9 // 2
print(a)

arr = np.array([2, 3, 4, 5, 6, 7])
p = np.concatenate([arr[:3], arr[5:9]], axis=0)
print(arr)
print(arr[:0])
print(p)
# note:初始化是一定是一个矩阵
rr = np.array([[2, 3, 4, 5, 6, 7],
               [1, 2, 3, 4, 5, 1]])

print(rr[1])
