from numpy.random import RandomState
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    #
    rdm = RandomState(1)
    dataSet_size = 128
    X = rdm.rand(dataSet_size, 2)
    # note:int强转bool类型时True为1，False为0
    Y = [[int(x1 + x2)] for (x1, x2) in X]
    Y2 = [[int(x1 + x2 < 1)] for (x1, x2) in X]
    Y3 = [[x1 + x2 < 1] for (x1, x2) in X]
    print(Y)
    print(Y2)
    print(Y3)
    print(int(True))
    print(int(False))
    mat = tf.Variable([[1, 2], [3, 4]])
    h = tf.placeholder(dtype=tf.float32)
    k = tf.placeholder(dtype=tf.float32)
    m = tf.reduce_mean(h)
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        print(sess.run(h, feed_dict={h: 0.5}))
    f = np.array([2, 3, 4, 1, 5, 7, 8])
    b = tf.nn.embedding_lookup(f, [1, 3])
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print(sess.run(b))
