"""
卷积神经网络
"""
from keras import models
from keras import layers
import numpy as np
from keras.utils.np_utils import to_categorical
from commonUtils.Loggings import *
import time

logger = Logger().getLogger()


def load_data_mnist():
    logger.info("load mnist dataset ....")
    data = np.load(
        r"D:\work\learning\machineLearning\keraslearn\sample\mnist.npz")
    train_images, train_labels = data['x_train'], data['y_train']
    test_images, test_labels = data['x_test'], data['y_test']
    logger.info("加载训练数据：{}，测试数据：{}".format(len(train_labels), len(test_labels)))
    logger.info('训练集形状高：{},行：{},列：{}'.format(train_images.shape[0], train_images.shape[1], train_images.shape[2]))
    logger.info('训练集的维数:{}'.format(train_images.ndim))
    return (train_images, train_labels), (test_images, test_labels)


def build_conv_model():
    """
    密集连接层和卷积层的区别就是，Dense层从特征空间中学习到的是全局模式，
    卷积层学习到的是局部模式
    :return:
    """
    model = models.Sequential()
    # 卷积层32表示有那么多的过滤器，输出的会有32个通道
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # note:池化是做什么的了，通过池化选出最大的特征
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # 展平
    model.add(layers.Flatten())
    # 神经网络隐藏层
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def train():
    (train_data, train_labels), (test_data, test_labels) = load_data_mnist()
    # 变成一个四维的
    train_data = train_data.reshape((60000, 28, 28, 1))
    # 标准化
    train_data = train_data.astype('float32') / 255
    logger.warning('训练集的维数：{}'.format(train_data.ndim))

    test_data = test_data.reshape((10000, 28, 28, 1))
    test_data = test_data.astype('float32') / 255

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    start = time.time()
    model = build_conv_model()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=5, batch_size=64)
    end = time.time()
    logger.warning('用时：{}'.format(end - start))


if __name__ == '__main__':
    train()
