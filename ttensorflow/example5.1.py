"""
获取MNIST数据源
"""
from tensorflow.examples.tutorials.mnist import input_data
from commonUtils.Loggings import *

logger = Logger().getLogger()


def test1():
    # 获取数据集但现在过期了
    mnist = input_data.read_data_sets("D:/workspace/pproject/ttensorflow/sample", one_hot=True)

    batch_size = 100
    # 分批次访问
    xs, ys = mnist.train.next_batch(batch_size)
    print(xs.shape)

    logger.info("{}".format(mnist.train.num_examples))
    logger.info("{}".format(mnist.validation.num_examples))
    logger.info("{}".format(mnist.test.num_examples))
    logger.warning("{}".format(mnist.train.images[0]))
    logger.warning("{}".format(mnist.train.labels[0]))


def test2():
    # mnist数据集相关的常数
    # 输入层的节点数，这个就等于图片的像素
    INPUT_NODE = 784
    # 输出层节点数，这里需要区分0-9这是个数
    OUT_PUT = 10
    # 隐藏层节点数
    LAYER1_NODE = 500
    # 训练的个数
    BATCH_SIZE = 100

    # 基础学习率
    LEARNING_RATE_BASE = 0.8
    # 学习率的衰减率
    LEARNING_RATE_DECAY = 0.99
    # 描述模型复杂度的正则化项在损失函数中的系数
    REGULARIZATION = 0.00001
    # 训练次数
    TRAINING_STEP = 30000
    # 滑动平均衰减率
    MOVING_AVERAGE_DECAY = 0.99

    def inference(input_tensor, avg_class, weights1, biass1, weights2, biases2):
        """
        一个辅助函数，给定神经网络的输入和所有参数，计算神经网络向前传播的结果
        :param input_tensor: 输入张量
        :param avg_class: 滑动平均类
        :param weights1: 权重1
        :param biass1: 偏移量1
        :param weights2:
        :param biases2:
        :return:
        """


if __name__ == '__main__':
    test1()
