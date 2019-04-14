"""
手写识别
"""
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

import numpy as np

from commonUtils.Loggings import *

logger = Logger().getLogger()


def load_data_mnist():
    """
    加载数据集
    :return:
    """
    logger.info("starting load dataset...")
    (train_image, trian_label), (test_image, test_label) = mnist.load_data()
    logger.info('train size:{}'.format(len(trian_label)))
    logger.info('test size:{}'.format(len(test_label)))
    return (train_image, trian_label), (test_image, test_label)


def load_from_local():
    """
    从本地加载数据集
    :return:
    """
    logger.info('开始加载数据')
    data = np.load(r"/workspace/learning/machineLearning/keraslearn/sample/mnist.npz")
    train_image, train_label = data['x_train'], data['y_train']
    test_image, test_label = data['x_test'], data['y_test']
    logger.info('train size:{}'.format(len(train_label)))
    logger.info('test size:{}'.format(len(test_label)))
    return (train_image, train_label), (test_image, test_label)


def build_model():
    """
    构建模型
    :return:
    """
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    logger.info('构建模型结束')
    return model


def train():
    """
    训练模型
    :return:
    """
    (train_image, train_label), (test_image, test_label) = load_from_local()
    logger.info('图片数据集的是{}行{}列的矩阵'.format(train_image.shape[0], test_image.shape[1]))
    # 准备图像
    train_image = train_image.reshape((60000, 28 * 28))
    # 缩放数据
    train_image = train_image.astype('float32') / 255

    test_image = test_image.reshape((10000, 28 * 28))
    test_image = test_image.astype('float32') / 255
    # to hot
    logger.info('没有to hot前的类型{}'.format(type(train_label)))
    train_label = to_categorical(train_label)
    logger.info('to hot 之后:{}'.format(type(train_label)))
    test_label = to_categorical(test_label)

    model = build_model()
    model.fit(train_image, train_label, epochs=5, batch_size=128)
    prediction = model.predict(test_image[300].reshape(784, 1))
    logger.critical("prediction {}".format(prediction))
    history = model.evaluate(test_image, test_label)
    logger.info("history : {}".format(history))
    logger.info("end...")


if __name__ == '__main__':
    train()
