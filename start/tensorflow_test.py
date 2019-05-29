"""
我要测试
"""
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np

l = [1, 2, 3, 4]
n = np.array([[1], [2], [3], [4]])
one = tf.one_hot(n, 5)
sess = tf.InteractiveSession()
y = OneHotEncoder().fit_transform(n.reshape(-1, 1), 5).todense()
b = np.zeros((1, 1))
y = np.insert(y, y.shape[1], axis=1, values=b)
print(y)

# print(y)
# print(sess.run(one))
