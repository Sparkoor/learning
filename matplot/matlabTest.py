import matplotlib.pyplot as plt
import numpy as np

x = [-1, 2, 3, 4]
y = [-1, 2, 3, 4]
np.linspace(-3, 3, 100)
plt.plot(x[:3], y[:3])
plt.xlabel('$x$', fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.show()
convert = lambda s: int(s.strip() or -999)
# 使用必须字段一样多
data = np.genfromtxt('global.1751_2014.csv', delimiter=',', skip_header=2,encoding='utf-8')

print(data[:, 1])
