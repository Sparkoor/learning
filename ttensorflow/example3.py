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
loss = tf.reduce_mean(tf.square(y_ - y) + tf.contrib.layers.l2_regularizer(0.5)(w))
"""


def getWeight(shape, lamba):
    """
    获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为losses的集合中
    :param shanpe:
    :param lamba:
    :return:
    """
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection函数将这个新变量的L2正则化损失项加入集合
    # 这个函数的第一个参数’losses'是集合的名字，第二个是要加入集合的内容
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamba)(var))
    return var


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
# 定义每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)
# 这个变量维护前向传播时最深层的节点，开始的时候就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]
# 通过一个循环来生成5层全连接的神经网络结构
for i in range(1, n_layers):
    # 下一层的节点数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
    weight = getWeight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用relu激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点个数更新为当前层的节点个数
    in_dimension = layer_dimension[i]
# 在定义神经网络向前传播的同时已经将所有的L2正则化损失加入了图上的集合
# 这里只需要计算刻画模型在训练数据上表现的损失函数
mes_lose = tf.reduce_mean(tf.square(y_ - cur_layer))
# 将均方误差函数加入损失函数集合
tf.add_to_collection('losses', mes_lose)
# get_collection返回一个列表，这个列表是所有这个集合中的元素，在这个样例中
# 这些元素就是损失函数不同的部分，将他们加起来就可以得到最终的损失函数
loss = tf.add_n(tf.get_collection('losses'))
