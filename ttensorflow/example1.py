"""
计算图的使用
"""
import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.3, 2.3], name='b')
print(a + b)
# 查看张量所属计算图
print(a.graph is tf.get_default_graph())
# 通过tf.graph生成计算图
g1 = tf.Graph()
with g1.as_default():
    # 在计算图中定义变量v，并设置初始值为零
    # note:有些接口名和书中的不一样
    v = tf.get_variable('v', shape=[1], initializer=tf.ones_initializer())

# 在计算图中读取变量v的值
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        # 输出
        print(sess.run(tf.get_variable("v")))
g = tf.Graph()
with g.device('/gpu:0'):
    result = a + b
