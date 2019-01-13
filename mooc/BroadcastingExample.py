import numpy as np

# 矩阵是一个二维数组
A = np.array([[56.0, 0.0, 4.4, 68.0],
              [1.2, 104.0, 52.0, 8.0],
              [1.8, 135.0, 99.0, 0.9]])
cal = A.sum(axis=0)
print(cal)
# reshape是确保矩阵形状
percentage = 100 * A / (cal.reshape(1, 4))
print(percentage)

B = np.random.randn(5, 1)
print(B)
assert (B.shape == (5, 1))
