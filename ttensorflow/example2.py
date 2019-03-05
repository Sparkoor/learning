"""
完整的神经网络样例程序
网络解决二分类问题
"""
import tensorflow as tf
# 生成模拟数据集
from numpy.random import RandomState

# 定义训练数据的batch的大小
batch_size = 8
# 定义神经网络参数，
# note：Variable更新保存变量 stddev方差的设置，也可以通过函数里面是另一个参数来初始化变量，seed设置随机种子保证每次运行的结果一样
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))

# 在shape的一个维度使用None可以方便使用不大的batch大小，在训练时需要把数据分成比较小
# 的batch，但在测试时，可以一次性的使用全部的数据，当数据集大时，有可能会溢出
# note：placeholder存放数据，占位符？？？，以后赋值，相当于声明变量
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
# 定义网络神经前置传播过程
# note：matmul的功能是矩阵乘法
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 定义损失函数和反向传播算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# note：AdamOptimizer 常见的优化方法，有三个
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)

# 定义规则来给出样本的标签，在这里所有x1+x2<1的样例都被认为是正样例
# 而其他为负样本，在这里使用0来表示负样本，1为正样例
# 大部分解决分类问题的神经网络都会采用0和1的表示方法
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
# 初始化变量等
with tf.Session() as sess:
    # note:初始化所有变量并赋值
    init_op = tf.initialize_all_variables()
    # 初始化变量
    # note：初始化变量
    sess.run(init_op)
    print(sess.run(w1.initializer))
    print(sess.run(w2))
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch个样本进行训练
        start = (i * batch_size) % dataSet_size
        end = min(start + batch_size, dataSet_size)
        # 通过选取的样本训练神经网络并更新参数
        # note：通过feed_dict传递x的值，和placeholder配合
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("经过多少次训练。。。。")
# 得出的结果
print(w1)
print(w2)
