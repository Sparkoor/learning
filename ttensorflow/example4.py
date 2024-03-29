"""
滑动平均模型，可以使模型在测试数据上更健壮
"""
import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量的初始值为0，这里指定了变量类型，因为计算滑动平均的变量必须是实数型
v1 = tf.Variable(0, dtype=tf.float32)
# step模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类，初始化时给定了衰减率和控制衰减率的变量
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时，这个列表中的变量都会被更新
# note：列表内容执行更新操作时会被更新
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # 通过ema.average(v1)获取滑动平均之后的变量的取值，在初始化之后变量v1的值和v1的滑动平均都为0
    # note:什么意思
    print(sess.run([v1, ema.average(v1)]))
    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值，衰减率为min(0.99,(1+step)/(10+step)=0.1}=0.1
    # 所以v1的滑动平均会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新step的值，note：tf.assign(var,value) ,用来更新变量
    sess.run(tf.assign(step, 1000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新v1的平均滑动值，衰减率为min{0.99，(1+step)/(10+step)}
    # v1的滑动平均值更新为 0.99*4.5+0.01*10=4.555
    # note:这是执行操作，执行更新列表的内容
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
    # 再次更新滑动平均值 0.99*4.555+0.1*10=4.60945
    sess.run(maintain_average_op)
    # note:在第五章会有具体的应用
    print(sess.run(v1, ema.average(v1)))


