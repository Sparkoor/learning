"""
使用正则化处理过拟合问题
在损失函数中加入刻画模型复杂度的指标
j(s)+lam*R(w)
"""
import tensorflow as tf

# 使用L2正则化损失函数
"""
w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y = tf.matmul(x, w)
# 正则化损失函数
loss = tf.reduce_mean(tf.square(y_ - y) + tf.contrib.layers.l2 _regularizer(0.5)(w))
"""

