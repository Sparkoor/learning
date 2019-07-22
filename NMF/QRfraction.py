"""
矩阵分解
"""
import numpy as np

a = np.mat([[1, 2, 3],
            [4, 5, 6]])

print(a)

q, r = np.linalg.qr(a)
print(q, r, np.dot(q, r), sep='\n\n')
