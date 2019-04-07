"""
第一种构建方法
Sequential() 构建神经网络
"""
from keras import models
from keras import layers
# 符号
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from commonUtils.Loggings import *

logger = Logger().getLogger()


def initModel():
    # 初始化一个模型
    model = models.Sequential()
    # 第一层 输入层Dense(输出维数，激活函数，输入维数)
    model.add(layers.Dense(32, activation='rule', input_shape=(784,)))
    # 第二层也是输出层
    model.add(layers.Dense(10, activation='softmax'))


def initModelByAPI():
    # 使用函数式API初始化，用起来比较灵活
    input_tensor = layers.Input(shape=(784,))
    # 第一层
    x = layers.Dense(32, activation='relu')(input_tensor)
    out_tensor = layers.Dense(10, activation='softmax')(x)
    # 定义模型完成
    model = models.Model(input_tensor, out_tensor)
    # 设置训练模型 optimizers是优化器 metrics是优化器？？？？
    model.compile(optimizers=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])
    # 给模型喂数据 epoch是轮次
    model.fit(input_tensor, out_tensor, batch_size=128, epochs=10)


def dataInit():
    data = np.load(
        r'D:\work\learning\machineLearning\keras\sample\imdb.npz')
    # 这样查看数据名称
    print("{}".format(data.files))
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    logger.critical('拿数据出来。。。')
    # print(x_train[:100])
    print(max([max(sequence) for sequence in x_train]))
    return (x_train, y_train), (x_test, y_test)


def vectorize_sequence(sequence, dimension=100000):
    # note:初始数组要注意括号,dtype要给出不然内存容易爆
    result = np.zeros((len(sequence), dimension), dtype='float32')
    for i, sequen in enumerate(sequence):
        result[i, sequen] = 1
    return result


def init_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(100000,)))
    model.add(layers.Dense(16, activation='relu', ))
    model.add(layers.Dense(1, activation='sigmoid'))
    # 模型编译
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    (x_trian, y_train), (x_test, y_test) = dataInit()
    logger.error("x_train type:{}".format(type(x_trian)))
    x_trian = vectorize_sequence(x_trian)
    y_train = np.asarray(y_train).astype('float32')
    x_val = x_trian[:10000]
    partial_x_trian = x_trian[10000:]
    y_val = y_train[:10000]
    partial_y_trian = y_train[10000:]
    logger.warning('训练数据集x_train:{}'.format(len(x_val)))
    logger.warning('y_val size{}'.format(len(y_val)))
    logger.warning('partial_y_val size{}'.format(len(partial_y_trian)))
    logger.warning('part_x_val size{}'.format(len(partial_x_trian)))
    history = model.fit(partial_x_trian, partial_y_trian, epochs=20, batch_size=512, validation_data=(x_val, y_val))
    print("训练结束")
    history_dict = history.history
    print(history_dict.keys())
    loss_value = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_value) + 1)
    plt.plot(epochs, loss_value, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='validation loss')
    plt.ylabel('Epoch')
    plt.xlabel('loss')
    plt.legend()
    plt.show()


def loadData_reuter():
    """
    加载路透社数据集
    :return:
    """
    data = np.load(r"D:\work\learning\machineLearning\keras\sample\reuters.npz")
    x = data['x']
    y = data['y']
    logger.info('load data type:{}'.format(type(x)))
    return x, y


def to_one_hot(labels, dimension=46):
    """
    one-hot编码也叫分类编码
    :param labels:
    :param dimension:
    :return:
    """
    # note:这个函数也可以实现to_one_hot
    from keras.utils.np_utils import to_categorical
    result = np.zeros((len(labels), dimension), dtype='float32')
    for i, label in enumerate(labels):
        result[i, label] = 1
    return result


if __name__ == '__main__':
    loadData_reuter()
