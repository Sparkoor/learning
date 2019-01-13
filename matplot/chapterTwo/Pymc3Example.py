"""
手动构造抛硬币的实验数据进行实验
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 构造数据
np.random.seed(123)
n_experiment = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=n_experiment)
print('data', data)
# array([1, 0, 0, 0])
