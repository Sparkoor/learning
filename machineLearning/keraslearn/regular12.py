"""
防止过拟合使用，正则化方法
"""
from keras import models
from keras import layers
from keras import regularizers


def build_model():
    model = models.Sequential()
    # 12(0.0001)该层权重矩阵的每个系数都会是网络的总损失增加0.001*weith
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.0001), activation='relu', input_shape=(1000,)))
    # 添加dropout层降低过拟合。其核心思想是在层中引入噪声，打破不显然的偶然模式，如果没有噪声，网络将会记住这些偶然模式，0.5是舍弃一半的单元
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

