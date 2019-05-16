"""
使用scikit-Learn训练并运行线性模型
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as sk

from commonUtils.Loggings import Logger

logger = Logger().getLogger()

# 加载数据
oecd_bli = pd.read_csv(r"D:/work/learning/machineLearning/scikitlrarn/sample/oecd_bli_2015.csv", thousands=',')
# note：这两个读的数据
gdp_per_capita = pd.read_csv(r"D:/work/learning/machineLearning/scikitlrarn/sample/oecd_bli_2015.csv", thousands=',',
                             delimiter='\t', encoding='latin1',
                             na_values='n/a')
# 准备数据
# print(oecd_bli.head())
# print(gdp_per_capita.head())
# print(oecd_bli.ndim)
# print(type(oecd_bli))
# print(oecd_bli.loc[2].tolist())
# print(oecd_bli.head(0))
print(list(oecd_bli.columns.tolist()))
print(list(gdp_per_capita))
