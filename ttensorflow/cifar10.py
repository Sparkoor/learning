"""
from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
"""
import re
import tensorflow as tf

import cifar10_input

# todo:未知
FLAGS = tf.app.flags.FLAGS

# 基本模型参数,todo:??????
tf.app.flags.DEFINE_integer('batch_size', 128, """number of images to process in a batch""")
tf.app.flags.DEFINE_boolean('use_fp16', True, """Train the model using fp16""")

# 全局变量
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
# 全局参数
# 用于移动平均衰减率
MOVING_AVERAGE_DECAY = 0.9999
# 每批次减少的数量
NUM_EPOCHS_PER_DECAY = 350.0
# 学习率下降因子
LEARNING_RATE_DECAY_FACTOR = 0.1
# 初始学习率
INITAL_LEARNING_RATE = 0.1
# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _activation_summary(x):
    """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    # todo:感觉是在设置路径
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name, '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initalizer):
    """
    创建存储在cpu的变量
    :param name:
    :param shape:
    :param initalizer:
    :return:
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initalizer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """
    创建初始的带有权重衰减的变量
    :param name:
    :param shape:
    :param stddev:
    :param wd: 添加l2正则化项
    :return:
    """
    # todo:16和32的区别
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """
    获取带有变形的数据集
    :return:
    """
    images, labels = cifar10_input.distorted_inputs(batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data):
    pass
