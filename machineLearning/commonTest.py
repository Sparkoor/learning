import numpy as np

import matplotlib.pyplot as plt

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
